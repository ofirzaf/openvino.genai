// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_stateful_sdpa_model() {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    auto embedding_weights =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 4}, std::vector<float>(16 * 4, 1.0f));
    auto embeddings =
        std::make_shared<ov::op::v8::Gather>(embedding_weights,
                                             input_ids,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).set_names({"beam_idx"});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, -1});
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto make_cache = [&](const std::string& variable_id) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(4), ov::element::f32, variable_id});
        auto initial = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 4}, {0.0f});
        auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
        auto past = std::make_shared<ov::op::v8::Gather>(read, beam_idx, gather_axis);
        auto current = std::make_shared<ov::op::v1::Reshape>(embeddings, reshape_pattern, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 1);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto transposed = std::make_shared<ov::op::v1::Transpose>(concat, transpose_order);
        return std::make_tuple(transposed, assign);
    };

    auto [key, key_assign] = make_cache("dflash_test_key_cache");
    auto [value, value_assign] = make_cache("dflash_test_value_cache");
    auto query =
        std::make_shared<ov::op::v0::Unsqueeze>(embeddings,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 3}, {0.0f});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);

    auto attention_transposed = std::make_shared<ov::op::v1::Transpose>(sdpa, transpose_order);
    auto attention_hidden =
        std::make_shared<ov::op::v1::Reshape>(attention_transposed,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1}),
                                              true);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {1.0f});
    auto decoder_layer = std::make_shared<ov::op::v1::Add>(embeddings, attention_hidden);
    decoder_layer->set_friendly_name("decoder_layer_0");
    auto final_norm = std::make_shared<ov::op::v1::Multiply>(decoder_layer, bias);
    final_norm->set_friendly_name("final_norm");
    auto lm_head_weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, {1.0f});
    auto lm_head = std::make_shared<ov::op::v0::MatMul>(final_norm, lm_head_weights);
    lm_head->set_friendly_name("lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(lm_head);
    logits_result->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{logits_result},
                                             ov::SinkVector{key_assign, value_assign},
                                             ov::ParameterVector{input_ids, beam_idx});
    model->set_rt_info(
        std::string(
            R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"final_norm","output_index":0}}})"),
        "hidden_states_decoder_layers");
    return model;
}

std::shared_ptr<ov::Model> make_eagle3_pattern_hidden_state_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    ov::Output<ov::Node> current = input->output(0);

    for (size_t layer_idx = 0; layer_idx < 5; ++layer_idx) {
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2, 4}, 1.0f);
        auto multiplied = std::make_shared<ov::op::v1::Multiply>(current, scale);
        auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 4}, 1.0f);
        auto projection = std::make_shared<ov::op::v0::MatMul>(multiplied, weight);
        auto layer = std::make_shared<ov::op::v1::Add>(current, projection);
        layer->set_friendly_name("model.layers." + std::to_string(layer_idx) + "/residual_add");
        current = layer->output(0);
    }

    auto result = std::make_shared<ov::op::v0::Result>(current);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> make_dflash_draft_hidden_states_model(const ov::PartialShape& hidden_states_shape) {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, hidden_states_shape);
    hidden_states->set_friendly_name("hidden_states");
    hidden_states->output(0).set_names({"hidden_states"});

    const auto hidden_size = static_cast<size_t>(hidden_states_shape[2].get_length());
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, hidden_size}, {1.0f});
    auto consumer = std::make_shared<ov::op::v1::Add>(hidden_states, bias);
    consumer->set_friendly_name("draft_hidden_states_consumer");
    auto result = std::make_shared<ov::op::v0::Result>(consumer);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden_states});
}

std::shared_ptr<ov::Model> make_kv_state_contract_model() {
    auto kv_delta =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, ov::Dimension::dynamic(), 1});
    kv_delta->set_friendly_name("kv_delta");
    kv_delta->output(0).set_names({"kv_delta"});

    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1, 1, ov::Dimension::dynamic(), 1},
                                   ov::element::f32,
                                   "dflash_kv_state"});
    auto initial = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 0, 1}, std::vector<float>{});
    auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{read, kv_delta}, 2);
    auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{assign}, ov::ParameterVector{kv_delta});
}

void infer_kv_delta(ov::InferRequest& request, size_t length) {
    ov::Tensor delta(ov::element::f32, ov::Shape{1, 1, length, 1});
    std::iota(delta.data<float>(), delta.data<float>() + delta.get_size(), 0.0f);
    request.set_tensor("kv_delta", delta);
    request.infer();
}

void expect_kv_state_length(ov::InferRequest& request, size_t expected_length) {
    const auto states = request.query_state();
    ASSERT_EQ(states.size(), 1);
    const auto shape = states.front().get_state().get_shape();
    ASSERT_EQ(shape, ov::Shape({1, 1, expected_length, 1}));
}

ov::Tensor make_token_major_hidden_delta(size_t seq_len, size_t hidden_size, float start = 0.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{seq_len, 1, hidden_size});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), start);
    return tensor;
}

std::vector<float> tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<int64_t> int64_tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
}

ov::Tensor make_embeddings(size_t rows, size_t hidden_size, float start = 0.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{1, rows, hidden_size});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), start);
    return tensor;
}

ov::Tensor make_position_ids(const std::vector<int64_t>& values) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{1, values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int64_t>());
    return tensor;
}

ov::Tensor make_per_layer_inputs(const std::vector<float>& values) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{1, values.size(), 1, 1});
    std::copy(values.begin(), values.end(), tensor.data<float>());
    return tensor;
}

ov::genai::SequenceGroup::Ptr make_embedding_group(uint64_t request_id,
                                                   const ov::Tensor& embeddings,
                                                   const std::vector<int64_t>& position_ids,
                                                   const std::vector<int64_t>& token_type_ids = {},
                                                   const std::vector<float>& per_layer_values = {},
                                                   bool with_deepstack = false) {
    std::unordered_map<std::string, ov::Tensor> extra_inputs;
    if (!token_type_ids.empty()) {
        extra_inputs.emplace("token_type_ids", make_position_ids(token_type_ids));
    }
    if (!per_layer_values.empty()) {
        extra_inputs.emplace("per_layer_inputs", make_per_layer_inputs(per_layer_values));
    }
    if (with_deepstack) {
        extra_inputs.emplace("deepstack_visual_embeds", ov::Tensor(ov::element::f32, ov::Shape{1, 1, 1}));
    }
    std::optional<std::unordered_map<std::string, ov::Tensor>> optional_extra_inputs;
    if (!extra_inputs.empty()) {
        optional_extra_inputs = std::move(extra_inputs);
    }

    return std::make_shared<ov::genai::SequenceGroup>(request_id,
                                                      embeddings,
                                                      ov::genai::GenerationConfig{},
                                                      optional_extra_inputs,
                                                      std::make_optional(make_position_ids(position_ids)));
}

void append_generated_embedding(const ov::genai::SequenceGroup::Ptr& group,
                                int64_t token_id,
                                const std::vector<float>& embedding,
                                int64_t position_id) {
    const auto sequence = group->get_sequences().front();
    sequence->append_token(token_id, 0.0f);
    ov::Tensor generated_embedding(ov::element::f32, ov::Shape{1, 1, embedding.size()});
    std::copy(embedding.begin(), embedding.end(), generated_embedding.data<float>());
    sequence->append_generated_ids_embeds(generated_embedding);
    sequence->append_position_ids(make_position_ids({position_id}));
}

std::shared_ptr<ov::op::v1::Add> find_dflash_draft_hidden_states_consumer(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "draft_hidden_states_consumer") {
            return ov::as_type_ptr<ov::op::v1::Add>(node);
        }
    }
    return nullptr;
}

size_t count_outputs_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    size_t count = 0;
    for (const auto& output : model->outputs()) {
        if (output.get_names().count(name) != 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST(DFlashCBHiddenDeltaBuffer, AppendsAndMaterializesSingleChunk) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto hidden_delta = make_token_major_hidden_delta(2, 3);

    buffer.append(hidden_delta);
    auto materialized = buffer.materialize();

    ASSERT_FALSE(buffer.empty());
    ASSERT_EQ(buffer.token_count(), 2);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({2, 1, 3}));
    ASSERT_EQ(tensor_values(materialized), tensor_values(hidden_delta));
}

TEST(DFlashCBHiddenDeltaBuffer, MergesChunksInOrder) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto first = make_token_major_hidden_delta(2, 2, 0.0f);
    auto second = make_token_major_hidden_delta(1, 2, 4.0f);

    buffer.append(first);
    buffer.append(second);
    auto materialized = buffer.materialize();

    ASSERT_EQ(buffer.token_count(), 3);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(materialized), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
}

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_stateful_sdpa_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);

    ASSERT_TRUE(properties.at("dflash_mode").as<bool>());
    ASSERT_EQ(properties.at("dflash_mask_token_id").as<int64_t>(), 151669);
    ASSERT_EQ(properties.at("dflash_target_layer_ids").as<std::vector<int32_t>>(),
              (std::vector<int32_t>{1, 12, 23, 34, 45}));

    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.mask_token_id, 151669);
    ASSERT_EQ(rt_info.target_layer_ids, (std::vector<int32_t>{1, 12, 23, 34, 45}));
    ASSERT_TRUE(properties.empty());
}

TEST(DFlashModelTransforms, FallsBackToEagle3LayerPatternWithoutAnnotations) {
    auto model = make_eagle3_pattern_hidden_state_model();
    const std::vector<int32_t> target_layer_ids = {0, 2, 4};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);

    ASSERT_FALSE(retained_locators);
    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 12}));
}

TEST(DFlashModelTransforms, ValidatesLocatorsAcrossPagedAttentionAndGatherBeforeAddingResult) {
    auto model = make_annotated_stateful_sdpa_model();
    const auto original_result_count = model->get_results().size();
    const std::vector<int32_t> target_layer_ids = {0, 1};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    ASSERT_EQ(locators.size(), 2);
    ASSERT_EQ(locators.front().producer, "decoder_layer_0");
    ASSERT_EQ(locators.back().producer, "final_norm");

    ov::pass::SDPAToPagedAttention(/*use_per_layer_block_indices_inputs=*/false,
                                   /*use_score_outputs=*/false,
                                   /*allow_score_aggregation=*/true,
                                   /*allow_cache_rotation=*/false)
        .run_on_model(model);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto transformed_ops = model->get_ordered_ops();
    ASSERT_TRUE(std::any_of(transformed_ops.begin(), transformed_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return std::string(node->get_type_name()).find("PagedAttention") != std::string::npos;
    }));
    for (const auto& locator : locators) {
        const auto shape = locator.output.get_partial_shape();
        ASSERT_TRUE(shape[1].is_static());
        ASSERT_EQ(shape[1].get_length(), 1);
    }

    ov::genai::utils::apply_gather_before_matmul_transformation(model);
    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto parameters = model->get_parameters();
    ASSERT_TRUE(
        std::any_of(parameters.begin(), parameters.end(), [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            return parameter->get_friendly_name() == "sampled_tokens_indices";
        }));
    const auto transformed_lm_head = std::get<0>(ov::genai::utils::find_llm_matmul(model));
    ASSERT_TRUE(transformed_lm_head);
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v8::Gather>(transformed_lm_head->input_value(0).get_node_shared_ptr()));

    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    ASSERT_EQ(model->get_results().size(), original_result_count + 1);
    ASSERT_EQ(model->output("last_hidden_state").get_partial_shape()[2].get_length(), 8);
}

TEST(DFlashModelTransforms, RejectsMalformedRequestedLocators) {
    const std::vector<std::pair<std::string, std::vector<int32_t>>> invalid_annotations = {
        {R"({"layers":{"0":{"producer":1,"output_index":0}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":-1}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0}}})", {1}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"decoder_layer_0","output_index":0}}})",
         {0, 1}},
    };
    for (const auto& [annotation, layer_ids] : invalid_annotations) {
        SCOPED_TRACE(annotation);
        auto model = make_annotated_stateful_sdpa_model();
        model->set_rt_info(annotation, "hidden_states_decoder_layers");
        EXPECT_THROW(ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, layer_ids), ov::Exception);
    }
}

TEST(DFlashModelTransforms, RejectsReplacedRetainedLocatorWhileOriginalRemainsLive) {
    auto model = make_annotated_stateful_sdpa_model();
    const std::vector<int32_t> target_layer_ids = {0};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    const auto original_output = locators.front().output;
    const auto original = original_output.get_node_shared_ptr();

    auto keep_original_live = std::make_shared<ov::op::v0::Result>(original_output);
    keep_original_live->set_friendly_name("original_decoder_layer_0_result");
    model->add_results({keep_original_live});

    auto replacement = std::make_shared<ov::op::v1::Add>(original->input_value(0), original->input_value(1));
    const auto original_name = original->get_friendly_name();
    for (auto consumer : original_output.get_target_inputs()) {
        if (consumer.get_node() != keep_original_live.get()) {
            consumer.replace_source_output(replacement->output(0));
        }
    }
    original->set_friendly_name("replaced_decoder_layer_0");
    replacement->set_friendly_name(original_name);
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(
        ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids),
        ov::Exception);
}

TEST(DFlashModelTransforms, ReshapesDynamicDraftHiddenStatesInputForCB) {
    auto model = make_dflash_draft_hidden_states_model(
        ov::PartialShape({ov::Dimension(1), ov::Dimension::dynamic(), ov::Dimension(4)}));

    ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model);

    ASSERT_EQ(model->input("hidden_states").get_partial_shape(),
              ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension(1), ov::Dimension(4)}));
    auto consumer = find_dflash_draft_hidden_states_consumer(model);
    ASSERT_TRUE(consumer);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(consumer->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(reshape);
    ASSERT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(1), ov::Dimension::dynamic(), ov::Dimension(4)}));
}

TEST(DFlashModelTransforms, ThrowsForIncompatibleDraftHiddenStatesInput) {
    auto model = make_dflash_draft_hidden_states_model(ov::PartialShape({2, 2, 4}));

    EXPECT_THROW(ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model), ov::Exception);
}

TEST(DFlashCBHiddenState, TruncatesRejectedTail) {
    auto hidden_delta = make_token_major_hidden_delta(4, 2);

    auto unchanged = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 0);
    ASSERT_EQ(unchanged.get_shape(), ov::Shape({4, 1, 2}));
    ASSERT_EQ(tensor_values(unchanged), tensor_values(hidden_delta));

    auto truncated = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 1);
    ASSERT_EQ(truncated.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(truncated), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));

    auto empty = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 4);
    ASSERT_EQ(empty.get_shape(), ov::Shape({0, 1, 2}));
    ASSERT_EQ(empty.get_size(), 0);
}

TEST(DFlashCBDraftStateContract, TrimsAndResetsKVState) {
    ov::Core core;
    auto request = core.compile_model(make_kv_state_contract_model(), "CPU").create_infer_request();

    infer_kv_delta(request, 3);
    expect_kv_state_length(request, 3);

    ov::genai::utils::CacheState state_to_trim;
    state_to_trim.num_tokens_to_trim = 1;
    state_to_trim.seq_length_axis = 2;
    ov::genai::utils::trim_kv_cache(request, state_to_trim, {});
    expect_kv_state_length(request, 2);

    infer_kv_delta(request, 1);
    expect_kv_state_length(request, 3);

    request.reset_state();
    expect_kv_state_length(request, 0);
}

TEST(DFlashCBPrefixIdentity, MatchesTokenPromptAndGeneratedPrefix) {
    auto owner =
        std::make_shared<ov::genai::SequenceGroup>(1, ov::genai::TokenIds{10, 20}, ov::genai::GenerationConfig{});
    owner->get_sequences().front()->append_token(30, 0.0f);
    owner->get_sequences().front()->append_token(40, 0.0f);
    auto current = std::make_shared<ov::genai::SequenceGroup>(2,
                                                              ov::genai::TokenIds{10, 20, 30, 99},
                                                              ov::genai::GenerationConfig{});

    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, current, 4), 3);
}

TEST(DFlashCBPrefixIdentity, MatchesGeneratedEmbeddingPerLayerInput) {
    auto owner = make_embedding_group(1, make_embeddings(2, 2), {0, 1}, {0, 0}, {10.0f, 11.0f});
    append_generated_embedding(owner, 7, {4.0f, 5.0f}, 2);
    auto current = make_embedding_group(2, make_embeddings(3, 2), {0, 1, 2}, {0, 0, 0}, {10.0f, 11.0f, 42.0f});
    const ov::genai::dflash_cb::PerLayerEmbeddingsCallback callback = [](const ov::Tensor& input_ids) {
        EXPECT_EQ(int64_tensor_values(input_ids), (std::vector<int64_t>{7}));
        return make_per_layer_inputs({42.0f});
    };

    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, current, 3, callback), 3);
    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, current, 3), 2);
}

TEST(DFlashCBPrefixIdentity, StopsOnEmbeddingPositionAndAuxiliaryMismatches) {
    auto owner = make_embedding_group(1, make_embeddings(3, 2), {0, 1, 2}, {0, 0, 0}, {1.0f, 2.0f, 3.0f});
    auto position_mismatch = make_embedding_group(2, make_embeddings(3, 2), {0, 8, 2}, {0, 0, 0}, {1.0f, 2.0f, 3.0f});
    auto per_layer_mismatch = make_embedding_group(3, make_embeddings(3, 2), {0, 1, 2}, {0, 0, 0}, {1.0f, 9.0f, 3.0f});
    auto deepstack_input =
        make_embedding_group(4, make_embeddings(3, 2), {0, 1, 2}, {0, 0, 0}, {1.0f, 2.0f, 3.0f}, true);

    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, position_mismatch, 3), 1);
    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, per_layer_mismatch, 3), 1);
    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, deepstack_input, 3), 0);
}

TEST(DFlashCBPrefixIdentity, BacksUpAndVerifiesImageSpans) {
    auto owner = make_embedding_group(1, make_embeddings(4, 2), {0, 1, 2, 3}, {0, 1, 1, 0});
    auto changed_embeddings = make_embeddings(4, 2);
    changed_embeddings.data<float>()[4] = -1.0f;
    auto changed = make_embedding_group(2, changed_embeddings, {0, 1, 2, 3}, {0, 1, 1, 0});

    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, changed, 4), 1);
    EXPECT_EQ(ov::genai::dflash_cb::last_owner_lcp(owner, changed, 2), 1);
}

TEST(DFlashCBDraftInputs, BuildsSeedMaskBlock) {
    auto input_ids = ov::genai::dflash_cb::build_draft_input_ids(42, 99, 3);

    ASSERT_EQ(input_ids.get_shape(), ov::Shape({1, 4}));
    ASSERT_EQ(int64_tensor_values(input_ids), (std::vector<int64_t>{42, 99, 99, 99}));
}

TEST(DFlashCBDraftInputs, BuildsPositionIdsFromCommittedLength) {
    auto position_ids = ov::genai::dflash_cb::build_draft_position_ids(5, 2, 3);

    ASSERT_EQ(position_ids.get_shape(), ov::Shape({1, 6}));
    ASSERT_EQ(int64_tensor_values(position_ids), (std::vector<int64_t>{5, 6, 7, 8, 9, 10}));
}

TEST(DFlashCBDraftInputs, BuildsAttentionMaskForFullDraftContext) {
    auto attention_mask = ov::genai::dflash_cb::build_draft_attention_mask(5, 2, 3);

    ASSERT_EQ(attention_mask.get_shape(), ov::Shape({1, 11}));
    ASSERT_EQ(int64_tensor_values(attention_mask), (std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(DFlashCBVlmPromptIds, BuildsPlaceholderPromptIds) {
    ASSERT_EQ(ov::genai::dflash_cb::build_placeholder_prompt_ids(3, 42), (std::vector<int64_t>{42, 42, 42}));
}

TEST(DFlashCBVlmPromptIds, RejectsInvalidPromptLengthAndPlaceholderId) {
    EXPECT_THROW(ov::genai::dflash_cb::build_placeholder_prompt_ids(0, 42), ov::Exception);
    EXPECT_THROW(ov::genai::dflash_cb::build_placeholder_prompt_ids(1, -1), ov::Exception);
}

TEST(DFlashCBGenerationConfig, DefaultsAssistantTokensToFive) {
    ov::genai::GenerationConfig config;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS);
}

TEST(DFlashCBGenerationConfig, PreservesExplicitAssistantTokens) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, 2);
}

TEST(DFlashCBGenerationConfig, AcceptsSupportedVlmOptions) {
    ov::genai::GenerationConfig config;

    EXPECT_NO_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config));
}

TEST(DFlashCBGenerationConfig, RejectsUnsupportedVlmOptions) {
    {
        ov::genai::GenerationConfig config;
        config.pruning_ratio = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.echo = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.logprobs = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.return_omni_outputs = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
}

TEST(DFlashCBLinearAttentionCheckpointing, ComputesRequiredBlockCount) {
    ASSERT_EQ(ov::genai::dflash_cb::linear_attention_checkpoint_block_count(5), 7);
    ASSERT_EQ(
        ov::genai::dflash_cb::linear_attention_checkpoint_block_count(
            ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS),
        ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS + 2);
}

TEST(DFlashCBLinearAttentionCheckpointing, AdjustsBlockCountOnlyForLinearAttentionTargets) {
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/0,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/false),
              0);
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/0,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/true),
              7);
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/9,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/true),
              9);
}

TEST(DFlashCBCandidatePlanning, KeepsDraftWindowStableUntilGenerationEnds) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 8, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 9, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, ClampsValidationWindowToTargetLength) {
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 9, 10), 0);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, SupportsSingleAssistantToken) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 0, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 9, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 9, 10), 0);
}

TEST(DFlashCBValidationAccounting, ComputesAcceptedAndRejected) {
    auto full_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 5);
    ASSERT_TRUE(full_accept.target_extended);
    ASSERT_EQ(full_accept.accepted, 3);
    ASSERT_EQ(full_accept.rejected, 0);

    auto partial_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 3);
    ASSERT_TRUE(partial_accept.target_extended);
    ASSERT_EQ(partial_accept.accepted, 1);
    ASSERT_EQ(partial_accept.rejected, 2);

    auto full_reject = ov::genai::dflash_cb::validation_accounting(3, 1, 2);
    ASSERT_TRUE(full_reject.target_extended);
    ASSERT_EQ(full_reject.accepted, 0);
    ASSERT_EQ(full_reject.rejected, 3);

    auto no_target_extension = ov::genai::dflash_cb::validation_accounting(3, 1, 1);
    ASSERT_FALSE(no_target_extension.target_extended);
    ASSERT_EQ(no_target_extension.accepted, 0);
    ASSERT_EQ(no_target_extension.rejected, 0);
}
