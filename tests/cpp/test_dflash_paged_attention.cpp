// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace {

// Optional artifact acceptance tests: point DFLASH_TEST_MODEL at the refreshed
// Qwen3.5-9B draft XML. No deployment weights are checked into the test suite.
class DFlashPagedAttention : public testing::Test {
protected:
    std::shared_ptr<ov::Model> native;

    void SetUp() override {
        const auto path = std::getenv("DFLASH_TEST_MODEL");
        if (!path)
            GTEST_SKIP() << "Set DFLASH_TEST_MODEL to the refreshed Qwen3.5-9B draft XML";
        native = ov::Core().read_model(path);
    }

    static std::shared_ptr<ov::Node> attention(const std::shared_ptr<ov::Model>& model) {
        for (const auto& node : model->get_ordered_ops())
            if (node->get_type_name() == std::string("ScaledDotProductAttention"))
                return node;
        throw std::runtime_error("Missing SDPA in artifact");
    }

    static std::shared_ptr<ov::Node> unwrap_unsqueeze(std::shared_ptr<ov::Node> node) {
        while (node->get_type_name() == std::string("Unsqueeze"))
            node = node->input_value(0).get_node_shared_ptr();
        return node;
    }

    static void check_converted(const std::shared_ptr<ov::Model>& model) {
        const auto types = model->input("token_type_ids");
        EXPECT_EQ(types.get_element_type(), ov::element::i32);
        EXPECT_EQ(types.get_partial_shape(), ov::PartialShape{-1});
        EXPECT_EQ(model->input("inputs_embeds").get_partial_shape(), (ov::PartialShape{-1, 4096}));
        EXPECT_EQ(model->input("hidden_states").get_partial_shape(), (ov::PartialShape{-1, 32768}));
        for (const auto& name : {"position_ids", "kv_gather_indices", "query_position_indices", "candidate_indices"})
            EXPECT_EQ(model->input(name).get_partial_shape(), ov::PartialShape{-1});
        EXPECT_THROW(model->input("attention_mask"), ov::Exception);
        EXPECT_THROW(model->input("beam_idx"), ov::Exception);
        std::vector<int32_t> windows;
        for (const auto& node : model->get_ordered_ops()) {
            const std::string type = node->get_type_name();
            EXPECT_NE(type, "ScaledDotProductAttention");
            EXPECT_NE(type, "ReadValue");
            EXPECT_NE(type, "Assign");
            if (type != "PagedAttentionExtension")
                continue;
            ASSERT_EQ(node->get_input_size(), 29);
            EXPECT_EQ(node->input_value(25), types);
            EXPECT_EQ(node->input_value(28), model->input("query_subsequence_begins"));
            ov::Model scalar(ov::OutputVector{node->input_value(10)}, ov::ParameterVector{});
            ov::TensorVector output{ov::Tensor(ov::element::i32, {})};
            ASSERT_TRUE(scalar.evaluate(output, {}));
            windows.push_back(output[0].data<int32_t>()[0]);
        }
        EXPECT_EQ(windows, (std::vector<int32_t>{4096, 4096, 4096, 4096, 4096, 0}));
    }
};

TEST_F(DFlashPagedAttention, ConvertsRefreshedAppendAllArtifact) {
    EXPECT_THROW(native->input("token_type_ids"), ov::Exception);
    auto model = native->clone();
    ov::genai::utils::dflash::convert_draft_to_paged_attention(model);
    check_converted(model);
    // Runtime preparation must not mutate the model from which it was cloned.
    EXPECT_THROW(native->input("token_type_ids"), ov::Exception);
}

TEST_F(DFlashPagedAttention, NormalizesGroupedQueryAndMaskLayout) {
    auto model = native->clone();
    auto sdpa = attention(model);
    auto inputs = sdpa->input_values();
    const auto query = inputs[0];
    int64_t kv_heads = 0;
    for (size_t port : {1, 2}) {
        // The artifact already spells repeated K/V as Unsqueeze-Broadcast-Reshape.
        auto reshape = inputs[port].get_node_shared_ptr();
        ASSERT_EQ(std::string(reshape->get_type_name()), "Reshape");
        auto broadcast = reshape->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(broadcast->get_type_name()), "Broadcast");
        inputs[port] = broadcast->input_value(0);
        ASSERT_EQ(inputs[port].get_partial_shape().rank(), 5);
        kv_heads = inputs[port].get_partial_shape()[1].get_length();
    }
    const auto heads = query.get_partial_shape()[1].get_length();
    auto shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{5},
        {0, kv_heads, heads / kv_heads, -1, query.get_partial_shape()[3].get_length()});
    inputs[0] = std::make_shared<ov::op::v1::Reshape>(query, shape, true);
    inputs[3] = std::make_shared<ov::op::v0::Unsqueeze>(inputs[3],
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto grouped = sdpa->clone_with_new_inputs(inputs);
    auto restored = std::make_shared<ov::op::v1::Reshape>(grouped, std::make_shared<ov::op::v3::ShapeOf>(query), false);
    ov::replace_output_update_name(sdpa->output(0), restored->output(0));
    ov::genai::utils::dflash::convert_draft_to_paged_attention(model);
    check_converted(model);
}

TEST_F(DFlashPagedAttention, RejectsIncompatibleCacheAndMasks) {
    using Mutation = std::function<void(const std::shared_ptr<ov::Model>&)>;
    const std::vector<std::pair<Mutation, std::string>> cases{
        {[](const auto& model) {
            auto assign = model->get_sinks().front();
            assign->input(0).replace_source_output(assign->input_value(0).get_node()->input_value(0));
        }, "re-export"},
        {[](const auto& model) {
            auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(attention(model));
            sdpa->set_causal(true);
        }, "noncausal"},
        {[](const auto& model) {
            auto mask = attention(model)->input_value(3).get_node_shared_ptr();
            auto inner = mask->input_value(2).get_node_shared_ptr();
            auto comparison = unwrap_unsqueeze(inner->input_value(0).get_node_shared_ptr());
            auto distance = comparison->input_value(0).get_node_shared_ptr();
            auto inputs = distance->input_values();
            distance->set_arguments({inputs[1], inputs[0]});
        }, "Unsupported DFlash attention mask"},
        {[](const auto& model) {
            auto mask = attention(model)->input_value(3).get_node_shared_ptr();
            auto comparison = unwrap_unsqueeze(mask->input_value(0).get_node_shared_ptr());
            auto boundary = comparison->input_value(1).get_node_shared_ptr();
            boundary->input(1).replace_source_output(ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        }, "Unsupported DFlash attention mask"},
        {[](const auto& model) {
            auto mask = attention(model)->input_value(3).get_node_shared_ptr();
            auto comparison = unwrap_unsqueeze(mask->input_value(0).get_node_shared_ptr());
            auto range = comparison->input_value(0).get_node_shared_ptr();
            range->input(0).replace_source_output(ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        }, "Unsupported DFlash attention mask"},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        SCOPED_TRACE(i);
        auto model = native->clone();
        cases[i].first(model);
        try {
            ov::genai::utils::dflash::convert_draft_to_paged_attention(model);
            FAIL() << "Accepted incompatible graph";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find(cases[i].second), std::string::npos) << error.what();
        }
    }
}


template <typename T>
ov::Tensor tensor_from(const ov::element::Type& type, const std::vector<T>& values) {
    ov::Tensor tensor(type, {values.size()});
    std::copy(values.begin(), values.end(), tensor.data<T>());
    return tensor;
}

float tensor_float(const ov::Tensor& tensor, size_t index) {
    if (tensor.get_element_type() == ov::element::f16)
        return static_cast<float>(tensor.data<const ov::float16>()[index]);
    return tensor.data<const float>()[index];
}

TEST_F(DFlashPagedAttention, PackedDraftMatchesIndependentSDPA) {
    const auto device = std::getenv("DFLASH_TEST_DEVICE");
    if (!device)
        GTEST_SKIP() << "Set DFLASH_TEST_DEVICE=GPU (or TEMPLATE) to run the numerical acceptance test";
    const bool gpu = std::string(device) == "GPU";
    ASSERT_TRUE(gpu || std::string(device) == "TEMPLATE");
    ov::Core core;
    core.register_plugin("openvino_template_plugin", "TEMPLATE");
    auto model = native->clone();
    ov::genai::utils::dflash::convert_draft_to_paged_attention(model);
    ov::pass::StatefulToStateless().run_on_model(native);
    auto reference = core.compile_model(native, "TEMPLATE");
    std::vector<ov::InferRequest> refs{reference.create_infer_request(), reference.create_infer_request()};
    std::vector<std::map<std::string, ov::Tensor>> committed(2);
    size_t heads = 0, dim = 0;
    for (const auto& port : reference.inputs()) {
        if (port.get_any_name().find("past_key_values.") != 0)
            continue;
        heads = port.get_partial_shape()[1].get_length();
        dim = port.get_partial_shape()[3].get_length();
        for (auto& caches : committed)
            caches.emplace(port.get_any_name(), ov::Tensor(port.get_element_type(), {1, heads, 0, dim}));
    }
    ASSERT_GT(heads, 0);
    constexpr size_t page_size = 16, num_pages = 16;
    if (!gpu) {
        for (const auto& param : model->get_parameters()) {
            if (param->get_friendly_name().find("key_cache.") == 0 ||
                param->get_friendly_name().find("value_cache.") == 0) {
                param->set_partial_shape({-1, static_cast<int64_t>(heads), page_size, static_cast<int64_t>(dim)});
                param->set_element_type(ov::element::f32);
            }
        }
        model->validate_nodes_and_infer_types();
    }
    ov::AnyMap properties;
    if (gpu) {
        properties[ov::hint::inference_precision.name()] = ov::element::f16;
        properties[ov::hint::kv_cache_precision.name()] = ov::element::f16;
    }
    auto compiled = core.compile_model(model, device, properties);
    auto request = compiled.create_infer_request();
    std::map<std::string, ov::Tensor> pages;
    for (const auto& port : compiled.inputs()) {
        if (port.get_any_name().find("key_cache.") != 0 && port.get_any_name().find("value_cache.") != 0)
            continue;
        ov::Shape shape{num_pages};
        for (size_t i = 1; i < 4; ++i)
            shape.push_back(port.get_partial_shape()[i].get_length());
        auto tensor = gpu ? ov::Tensor(core.get_default_context(device).create_tensor(port.get_element_type(), shape, {}))
                          : ov::Tensor(port.get_element_type(), shape);
        request.set_tensor(port, tensor);
        pages.emplace(port.get_any_name(), tensor);
    }
    const size_t hidden_width = reference.input("hidden_states").get_partial_shape()[2].get_length();
    const size_t embed_width = reference.input("inputs_embeds").get_partial_shape()[2].get_length();
    std::mt19937 generator(147);
    std::normal_distribution<float> normal;
    auto random_tensor = [&](const ov::Shape& shape) {
        ov::Tensor tensor(ov::element::f32, shape);
        for (size_t i = 0; i < tensor.get_size(); ++i)
            tensor.data<float>()[i] = normal(generator);
        return tensor;
    };
    const std::vector<std::vector<int32_t>> page_ids{{7, 1, 9, 4}, {3, 8, 2, 6}};
    const std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> steps{
        {{15, 5}, {4, 3}}, {{3, 1}, {2, 5}}, {{1, 4}, {5, 2}}};
    std::vector<size_t> past(2, 0);
    for (size_t step = 0; step < steps.size(); ++step) {
        SCOPED_TRACE(step);
        const auto& delta = steps[step].first;
        const auto& block = steps[step].second;
        if (step == 2) {
            past[1] = 0;
            for (auto& item : committed[1])
                item.second = ov::Tensor(item.second.get_element_type(), {1, heads, 0, dim});
        }
        const auto total_hidden = delta[0] + delta[1], total_query = block[0] + block[1];
        ov::Tensor hidden(ov::element::f32, {total_hidden, hidden_width});
        ov::Tensor embeds(ov::element::f32, {total_query, embed_width});
        std::vector<float> expected;
        std::vector<int64_t> positions, kv_gather, query_positions, candidates;
        std::vector<int32_t> types, kv_begins{0}, q_begins{0}, indices, index_begins{0}, past_lens;
        size_t h = 0, q = 0, kv = 0, max_context = 0;
        for (size_t i = 0; i < 2; ++i) {
            const auto d = delta[i], b = block[i];
            auto hid = random_tensor({1, d, hidden_width});
            auto emb = random_tensor({1, b, embed_width});
            ov::Tensor hid_rows(ov::element::f32, {d, hidden_width}, hid.data());
            ov::Tensor emb_rows(ov::element::f32, {b, embed_width}, emb.data());
            auto hid_dest = ov::Tensor(hidden, {h, 0}, {h + d, hidden_width});
            auto emb_dest = ov::Tensor(embeds, {q, 0}, {q + b, embed_width});
            hid_rows.copy_to(hid_dest);
            emb_rows.copy_to(emb_dest);
            std::vector<int64_t> pos(d + b);
            std::iota(pos.begin(), pos.end(), past[i]);
            auto pos_tensor = tensor_from(ov::element::i64, pos);
            pos_tensor.set_shape({1, d + b});
            auto mask = tensor_from(ov::element::i64, std::vector<int64_t>(past[i] + d + b, 1));
            mask.set_shape({1, past[i] + d + b});
            refs[i].set_tensor("inputs_embeds", emb);
            refs[i].set_tensor("hidden_states", hid);
            refs[i].set_tensor("position_ids", pos_tensor);
            refs[i].set_tensor("attention_mask", mask);
            for (const auto& item : committed[i])
                refs[i].set_tensor(item.first, item.second);
            refs[i].infer();
            const auto output = refs[i].get_tensor("last_hidden_state");
            for (size_t j = 0; j < output.get_size(); ++j)
                expected.push_back(tensor_float(output, j));
            for (auto& item : committed[i]) {
                auto present = refs[i].get_tensor("present." + item.first.substr(std::string("past_key_values.").size()));
                ov::Tensor prefix(present, {0, 0, 0, 0}, {1, heads, past[i] + d, dim});
                ov::Tensor retained(present.get_element_type(), prefix.get_shape());
                prefix.copy_to(retained);
                item.second = retained;
            }
            positions.insert(positions.end(), pos.begin(), pos.end());
            for (size_t j = 0; j < d; ++j)
                kv_gather.push_back(h + j);
            for (size_t j = 0; j < b; ++j) {
                kv_gather.push_back(total_hidden + q + j);
                query_positions.push_back(kv + d + j);
                if (j)
                    candidates.push_back(q + j);
            }
            types.insert(types.end(), d, 0);
            types.insert(types.end(), b, 1);
            const auto count = (past[i] + d + b + page_size - 1) / page_size;
            indices.insert(indices.end(), page_ids[i].begin(), page_ids[i].begin() + count);
            index_begins.push_back(indices.size());
            past_lens.push_back(past[i]);
            max_context = std::max(max_context, past[i] + d + b);
            past[i] += d;
            h += d;
            q += b;
            kv += d + b;
            q_begins.push_back(q);
            kv_begins.push_back(kv);
        }
        request.set_tensor("hidden_states", hidden);
        request.set_tensor("inputs_embeds", embeds);
        for (const auto& item : std::map<std::string, std::vector<int64_t>>{
                 {"position_ids", positions}, {"kv_gather_indices", kv_gather},
                 {"query_position_indices", query_positions}, {"candidate_indices", candidates}})
            request.set_tensor(item.first, tensor_from(ov::element::i64, item.second));
        for (const auto& item : std::map<std::string, std::vector<int32_t>>{
                 {"token_type_ids", types}, {"past_lens", past_lens}, {"subsequence_begins", kv_begins},
                 {"query_subsequence_begins", q_begins}, {"block_indices", indices}, {"block_indices_begins", index_begins}})
            request.set_tensor(item.first, tensor_from(ov::element::i32, item.second));
        auto context = tensor_from(ov::element::i32, std::vector<int32_t>{static_cast<int32_t>(max_context)});
        context.set_shape({});
        request.set_tensor("max_context_len", context);
        request.infer();
        const auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_size(), expected.size());
        const float atol = gpu ? 0.02f : 1e-5f, rtol = gpu ? 0.02f : 1e-4f;
        for (size_t j = 0; j < expected.size(); ++j)
            ASSERT_NEAR(tensor_float(actual, j), expected[j], atol + rtol * std::abs(expected[j])) << "output " << j;
        if (!gpu)
            continue;  // TEMPLATE keeps reference pages internally; inspect GPU cache tensors directly.
        for (size_t i = 0; i < 2; ++i) {
            for (const auto& item : committed[i]) {
                const auto dot = item.first.rfind('.');
                const auto kind = item.first.substr(dot + 1);
                const auto layer = item.first.substr(std::string("past_key_values.").size(),
                                                    dot - std::string("past_key_values.").size());
                const auto& remote = pages.at(kind + "_cache." + layer);
                ov::Tensor host(remote.get_element_type(), remote.get_shape());
                remote.copy_to(host);
                for (size_t head = 0; head < heads; ++head) {
                    for (size_t row = 0; row < past[i]; ++row) {
                        const size_t page = page_ids[i][row / page_size], slot = row % page_size;
                        for (size_t j = 0; j < dim; ++j) {
                            const auto offset = (page * heads + head) * page_size * dim +
                                (kind == "key" ? j * page_size + slot : slot * dim + j);
                            const auto ref = tensor_float(item.second, (head * past[i] + row) * dim + j);
                            ASSERT_NEAR(tensor_float(host, offset), ref, atol + rtol * std::abs(ref))
                                << "request " << i << " " << item.first << " row " << row;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace
