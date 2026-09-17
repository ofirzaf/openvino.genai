// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace ov::genai::utils::dflash {
namespace {
using namespace ov::op;
using ov::pass::paged_attention::PaParams;

std::shared_ptr<v0::Parameter> get_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->inputs()) {
        const auto& names = param.get_names();
        if (names.count(name)) {
            if (auto casted_param = ov::as_type_ptr<v0::Parameter>(param.get_node_shared_ptr())) {
                return casted_param;
            } else {
                OPENVINO_THROW("The model is in the inconsistent state. Found input '",
                               name,
                               "', but couldn't cast it to v0::Parameter.");
            }
        }
    }

    return nullptr;
}

bool depends_on_parameter(const ov::Output<ov::Node>& value, const ov::Node* parameter) {
    std::unordered_set<const ov::Node*> seen;
    std::vector<const ov::Node*> pending{value.get_node()};
    while (!pending.empty()) {
        auto node = pending.back();
        pending.pop_back();
        if (node == parameter)
            return true;
        if (!seen.insert(node).second)
            continue;
        // Shape-only dependencies do not make a RoPE table depend on token
        // values (the exporter derives its batch dimension from embeddings).
        if (ov::is_type<v0::ShapeOf>(node) || ov::is_type<v3::ShapeOf>(node))
            continue;
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node());
    }
    return false;
}

// Packed requests contain only valid rows. Prove that the padding-only part
// of the exported mask is zero when attention_mask is one, without evaluating
// any weights or depending on a particular dynamic sequence length.
std::optional<double> uniform_unpadded_mask_value(const ov::Output<ov::Node>& value,
                                                 const ov::Node* attention_mask) {
    const auto node = value.get_node_shared_ptr();
    if (node.get() == attention_mask)
        return 1.;
    if (auto constant = ov::as_type_ptr<v0::Constant>(node)) {
        const auto values = constant->cast_vector<double>();
        if (!values.empty() && std::all_of(values.begin(), values.end(), [&](double v) { return v == values.front(); }))
            return values.front();
        return {};
    }
    const std::string type = node->get_type_name();
    if (type == "Slice" || type == "StridedSlice" || type == "Reshape" || type == "Broadcast" ||
        type == "Unsqueeze" || type == "Squeeze" || type == "Convert")
        return uniform_unpadded_mask_value(node->input_value(0), attention_mask);
    if (type == "Add" || type == "Subtract" || type == "Multiply") {
        auto a = uniform_unpadded_mask_value(node->input_value(0), attention_mask);
        auto b = uniform_unpadded_mask_value(node->input_value(1), attention_mask);
        if (a && b) {
            if (type == "Add") return *a + *b;
            if (type == "Subtract") return *a - *b;
            return *a * *b;
        }
    }
    return {};
}

void validate_dflash_mask(const std::shared_ptr<v13::ScaledDotProductAttention>& sdpa,
                          const ov::Node* attention_mask) {
    OPENVINO_ASSERT(!sdpa->get_causal() && sdpa->get_input_size() >= 4,
                    "DFlash PA requires the native noncausal attention mask; re-export the draft.");
    auto mask = sdpa->input_value(3);
    if (mask.get_partial_shape().rank() == 5) {
        const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(mask.get_node_shared_ptr());
        const auto axis = unsqueeze ? ov::as_type_ptr<v0::Constant>(unsqueeze->get_input_node_shared_ptr(1)) : nullptr;
        OPENVINO_ASSERT(axis && axis->cast_vector<int64_t>() == std::vector<int64_t>{0},
                        "Unsupported DFlash attention mask layout; re-export the draft.");
        mask = unsqueeze->input_value(0);
    }
    if (uniform_unpadded_mask_value(mask, attention_mask) == std::optional<double>(0.))
        return;
    // Native SWA: Select(proposal_keys, 0, Select(distance >= window, -inf, padding)).
    auto scalar_is = [](const ov::Output<ov::Node>& value, int64_t expected) {
        const auto c = ov::as_type_ptr<v0::Constant>(value.get_node_shared_ptr());
        return c && c->cast_vector<int64_t>() == std::vector<int64_t>{expected};
    };
    auto unwrap_axes = [&](ov::Output<ov::Node> value, std::initializer_list<int64_t> axes) -> ov::Output<ov::Node> {
        for (const auto axis : axes) {
            if (!value.get_node() || !ov::is_type<v0::Unsqueeze>(value.get_node()) ||
                !scalar_is(value.get_node()->input_value(1), axis))
                return {};
            value = value.get_node()->input_value(0);
        }
        return value;
    };
    auto length_of = [&](const ov::Output<ov::Node>& length, const ov::Output<ov::Node>& data, bool shared_past = false) {
        const auto gather = ov::as_type_ptr<v8::Gather>(length.get_node_shared_ptr());
        if (!gather || !scalar_is(gather->input_value(1), 2) || !scalar_is(gather->input_value(2), 0))
            return false;
        const auto shape = gather->input_value(0).get_node_shared_ptr();
        if (!ov::is_type<v0::ShapeOf>(shape) && !ov::is_type<v3::ShapeOf>(shape))
            return false;
        if (shape->input_value(0) == data)
            return true;
        // Export shape folding shares the past length across layer caches.
        // PA likewise requires all layers to use the same per-request past.
        auto is_cache_read = [](ov::Output<ov::Node> value) {
            if (ov::is_type<v8::Gather>(value.get_node()))
                value = value.get_node()->input_value(0);
            return ov::is_type<ov::op::util::ReadValueBase>(value.get_node());
        };
        return shared_past && is_cache_read(data) && is_cache_read(shape->input_value(0)) &&
               data.get_partial_shape().compatible(shape->get_input_partial_shape(0));
    };
    auto is_range = [&](const ov::Output<ov::Node>& value, const ov::Output<ov::Node>& length) {
        const auto* node = value.get_node();
        return node && std::string(node->get_type_name()) == "Range" && scalar_is(node->input_value(0), 0) &&
               node->input_value(1) == length && scalar_is(node->input_value(2), 1);
    };
    auto outer = mask.get_node_shared_ptr();
    bool supported = false;
    if (std::string(outer->get_type_name()) == "Select") {
        auto proposal_value = unwrap_axes(outer->input_value(0), {2, 1, 0});
        auto proposal = proposal_value.get_node_shared_ptr();
        auto inner = outer->input_value(2).get_node_shared_ptr();
        if (proposal && std::string(proposal->get_type_name()) == "GreaterEqual" &&
            std::string(proposal->get_input_node_ptr(0)->get_type_name()) == "Range" &&
            std::string(proposal->get_input_node_ptr(1)->get_type_name()) == "Subtract" &&
            uniform_unpadded_mask_value(outer->input_value(1), attention_mask) == std::optional<double>(0.) &&
            std::string(inner->get_type_name()) == "Select") {
            auto distance = unwrap_axes(inner->input_value(0), {1, 0}).get_node_shared_ptr();
            auto excluded = uniform_unpadded_mask_value(inner->input_value(1), attention_mask);
            if (distance && std::string(distance->get_type_name()) == "GreaterEqual" &&
                std::string(distance->get_input_node_ptr(0)->get_type_name()) == "Subtract") {
                auto window = uniform_unpadded_mask_value(distance->input_value(1), attention_mask);
                supported = window && *window > 0 && excluded && *excluded < -1e4 &&
                            uniform_unpadded_mask_value(inner->input_value(2), attention_mask) == std::optional<double>(0.);
                // Prove the actual positions, not just the operator names:
                // K positions = [0, P+D+B), Q positions = P+D + [0, B).
                auto start = proposal->input_value(1).get_node_shared_ptr();
                auto total = start->input_value(0);
                auto q_length = start->input_value(1);
                auto keys = sdpa->input_value(1);
                while (std::string(keys.get_node()->get_type_name()) == "Reshape" ||
                       std::string(keys.get_node()->get_type_name()) == "Broadcast" ||
                       std::string(keys.get_node()->get_type_name()) == "Unsqueeze")
                    keys = keys.get_node()->input_value(0);
                const auto concat = ov::as_type_ptr<v0::Concat>(keys.get_node_shared_ptr());
                auto q = sdpa->input_value(0);
                if (q.get_partial_shape().rank() == 5 && ov::is_type<v1::Reshape>(q.get_node()))
                    q = q.get_node()->input_value(0);
                const auto total_node = total.get_node_shared_ptr();
                supported &= concat && (concat->get_axis() == 2 || concat->get_axis() == -2) &&
                             concat->get_input_size() == 2 &&
                             std::string(total_node->get_type_name()) == "Add" &&
                             length_of(total_node->input_value(0), concat->input_value(0), true) &&
                             length_of(total_node->input_value(1), concat->input_value(1)) &&
                             length_of(q_length, q) && is_range(proposal->input_value(0), total);
                const auto diff = distance->input_value(0).get_node_shared_ptr();
                const auto q_pos = diff->input_value(0).get_node_shared_ptr();
                supported &= std::string(q_pos->get_type_name()) == "Add" &&
                             q_pos->input_value(1) == start->output(0) &&
                             is_range(unwrap_axes(q_pos->input_value(0), {1}), q_length) &&
                             unwrap_axes(diff->input_value(1), {0}) == proposal->input_value(0);
            }
        }
    }
    OPENVINO_ASSERT(supported, "Unsupported DFlash attention mask; expected native full or sliding-window block attention. ",
                    "Layer: ", sdpa->get_friendly_name(), ". Re-export the draft with the current exporter.");
}

// PA replaces the cache update with an append of all current K/V. Check the
// state outputs as well as the attention inputs before discarding Assigns:
// older DFlash exports persist only the target-hidden prefix.
void validate_dflash_cache_updates(const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<std::string, ov::Output<ov::Node>> updates;
    for (const auto& sink : model->get_sinks()) {
        if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink))
            updates.emplace(assign->get_variable_id(), assign->input_value(0));
    }
    auto unwrap_layout = [](ov::Output<ov::Node> value) {
        while (true) {
            const std::string type = value.get_node()->get_type_name();
            if (type != "Reshape" && type != "Transpose" && type != "Unsqueeze" &&
                type != "Broadcast" && type != "Convert")
                return value;
            value = value.get_node()->input_value(0);
        }
    };
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(node);
        if (!sdpa)
            continue;
        validate_dflash_mask(sdpa, get_parameter(model, "attention_mask").get());
        for (size_t port : {1, 2}) {
            auto concat = ov::as_type_ptr<v0::Concat>(unwrap_layout(node->input_value(port)).get_node_shared_ptr());
            OPENVINO_ASSERT(concat && concat->get_input_size() == 2,
                            "DFlash PA requires append-all K/V. Re-export the draft with the current exporter.");
            auto past = unwrap_layout(concat->input_value(0));
            if (auto gather = ov::as_type_ptr<v8::Gather>(past.get_node_shared_ptr()))
                past = gather->input_value(0);
            auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(past.get_node_shared_ptr());
            OPENVINO_ASSERT(rv && updates.count(rv->get_variable_id()) &&
                                updates.at(rv->get_variable_id()) == concat->output(0),
                            "DFlash PA requires the complete attention K/V update in state. "
                            "Committed-prefix exports require re-export with the current exporter.");
        }
    }
}

// Some exports group queries as [1, KV heads, groups, Q, dim] and let SDPA
// broadcast K/V's singleton group axis. Normalize that export convention to
// ordinary 4D GQA before invoking the model-independent PA matcher.
void normalize_dflash_attention_layouts(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(node);
        if (!sdpa)
            continue;
        const auto query = sdpa->input_value(0);
        if (query.get_partial_shape().rank() == 4)
            continue;

        auto reshape = ov::as_type_ptr<v1::Reshape>(query.get_node_shared_ptr());
        const auto& grouped_shape = query.get_partial_shape();
        OPENVINO_ASSERT(grouped_shape.rank() == 5 && reshape &&
                            reshape->get_input_partial_shape(0).rank() == 4 &&
                            grouped_shape[0].compatible(1) && grouped_shape[1].is_static() && grouped_shape[2].is_static() &&
                            grouped_shape[4].is_static() &&
                            reshape->get_input_partial_shape(0)[1] == grouped_shape[1] * grouped_shape[2],
                        "Unsupported grouped DFlash query layout; re-export the draft.");
        const auto heads = grouped_shape[1].get_length() * grouped_shape[2].get_length();
        auto inputs = sdpa->input_values();
        inputs[0] = reshape->input_value(0);
        for (size_t port : {1, 2}) {
            const auto& shape = inputs[port].get_partial_shape();
            OPENVINO_ASSERT(shape.rank() == 5 && shape[0].compatible(1) && shape[1] == grouped_shape[1] &&
                                shape[2] == 1 && shape[4].is_static() && shape[4] == grouped_shape[4],
                            "Unsupported grouped DFlash K/V layout; re-export the draft.");
            auto repeats = v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{5},
                                                        {1, 1, grouped_shape[2].get_length(), 1, 1});
            auto broadcast_shape = std::make_shared<v1::Multiply>(std::make_shared<v3::ShapeOf>(inputs[port]), repeats);
            auto broadcast = std::make_shared<v3::Broadcast>(inputs[port], broadcast_shape);
            auto flat_shape = v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{4},
                                                          {1, heads, -1, shape[4].get_length()});
            inputs[port] = std::make_shared<v1::Reshape>(broadcast, flat_shape, false);
        }
        if (inputs[3].get_partial_shape().rank() == 5) {
            const auto mask = ov::as_type_ptr<v0::Unsqueeze>(inputs[3].get_node_shared_ptr());
            const auto axis = mask ? ov::as_type_ptr<v0::Constant>(mask->get_input_node_shared_ptr(1)) : nullptr;
            OPENVINO_ASSERT(axis && axis->cast_vector<int64_t>() == std::vector<int64_t>{0},
                            "Unsupported grouped DFlash mask layout; re-export the draft.");
            inputs[3] = mask->input_value(0);
        }
        auto normalized = sdpa->clone_with_new_inputs(inputs);
        normalized->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, normalized);
        auto output = std::make_shared<v1::Reshape>(normalized, std::make_shared<v3::ShapeOf>(query), false);
        ov::replace_output_update_name(sdpa->output(0), output->output(0));
    }
}

// Keep a singleton batch inside the exported draft. Q and target-hidden rows
// are independently packed; the runtime supplies gathers to interleave each
// request's context and proposal rows and to select its proposal positions.
void prepare_dflash_packing(const std::shared_ptr<ov::Model>& model, PaParams& params) {
    const auto embeds = get_parameter(model, "inputs_embeds");
    const auto hidden = get_parameter(model, "hidden_states");
    const auto positions = get_parameter(model, "position_ids");
    OPENVINO_ASSERT(embeds && hidden && positions, "DFlash PA requires embeddings, hidden states and position IDs");
    auto kv_gather = params.add("kv_gather_indices", ov::element::i64, ov::PartialShape{-1});
    auto query_positions = params.add("query_position_indices", ov::element::i64, ov::PartialShape{-1});
    auto candidates = params.add("candidate_indices", ov::element::i64, ov::PartialShape{-1});
    params.add("token_type_ids", ov::element::i32, ov::PartialShape{-1});
    size_t concats = 0, rope_slices = 0, candidate_slices = 0, attention_layers = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<v13::ScaledDotProductAttention>(node))
            ++attention_layers;
        if (auto concat = ov::as_type_ptr<v0::Concat>(node)) {
            if (concat->get_axis() != 1 || concat->get_input_size() != 2 ||
                concat->get_output_partial_shape(0).rank() != 3 ||
                !depends_on_parameter(concat->input_value(0), hidden.get()) ||
                depends_on_parameter(concat->input_value(0), embeds.get()) ||
                !depends_on_parameter(concat->input_value(1), embeds.get()))
                continue;
            auto consumers = concat->output(0).get_target_inputs();
            auto gather = std::make_shared<v8::Gather>(concat, kv_gather,
                                                       v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
            for (auto& consumer : consumers)
                consumer.replace_source_output(gather);
            ++concats;
        }
        if (auto slice = ov::as_type_ptr<v8::Slice>(node)) {
            if (slice->get_input_size() != 5)
                continue;
            const auto axes = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(4));
            const auto steps = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(3));
            if (!axes || !steps || steps->cast_vector<int64_t>() != std::vector<int64_t>{1})
                continue;
            const auto axis = axes->cast_vector<int64_t>();
            const auto data = slice->input_value(0);
            if (axis == std::vector<int64_t>{2} && data.get_partial_shape().rank() == 4 &&
                depends_on_parameter(data, positions.get()) && !depends_on_parameter(data, embeds.get()) &&
                !depends_on_parameter(data, hidden.get())) {
                auto gather = std::make_shared<v8::Gather>(data, query_positions,
                                                           v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
                ov::replace_node(slice, gather);
                ++rope_slices;
            } else if (axis == std::vector<int64_t>{1} && data.get_partial_shape().rank() == 3) {
                const auto start = ov::as_type_ptr<v0::Constant>(slice->get_input_node_shared_ptr(1));
                if (start && start->cast_vector<int64_t>() == std::vector<int64_t>{1} &&
                    depends_on_parameter(data, hidden.get()) && depends_on_parameter(data, embeds.get())) {
                    auto gather = std::make_shared<v8::Gather>(data, candidates,
                                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
                    ov::replace_node(slice, gather);
                    ++candidate_slices;
                }
            }
        }
    }
    OPENVINO_ASSERT(attention_layers > 0 && concats == attention_layers && rope_slices == 2 * attention_layers &&
                        candidate_slices == 1,
                    "Unsupported DFlash packing graph: attention=", attention_layers, ", context concatenations=", concats,
                    ", query RoPE slices=", rope_slices, ", candidate slices=", candidate_slices);
    for (const auto& name : {"inputs_embeds", "hidden_states", "position_ids"}) {
        auto param = get_parameter(model, name);
        OPENVINO_ASSERT(param, "DFlash PA is missing input ", name);
        auto shape = param->get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() >= 2 && shape[0].compatible(1),
                        "Unexpected DFlash input rank for ", name);
        shape = ov::PartialShape(std::vector<ov::Dimension>(shape.begin() + 1, shape.end()));
        shape[0] = ov::Dimension::dynamic();
        auto consumers = param->output(0).get_target_inputs();
        param->set_partial_shape(shape);
        auto expanded = std::make_shared<v0::Unsqueeze>(param,
                                                        v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        for (auto& consumer : consumers)
            consumer.replace_source_output(expanded);
    }
}

}  // namespace

void convert_draft_to_paged_attention(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash draft model cannot be null.");
    validate_dflash_cache_updates(model);
    normalize_dflash_attention_layouts(model);
    PaParams params(model->get_parameters());
    prepare_dflash_packing(model, params);
    model->add_parameters(params.items());
    model->validate_nodes_and_infer_types();

    ov::pass::paged_attention::Options options{};
    options.use_asymmetric_inputs = true;
    ov::pass::SDPAToPagedAttention(options).run_on_model(model);

    for (const auto& name : {"beam_idx", "attention_mask"}) {
        if (auto param = get_parameter(model, name))
            model->remove_parameter(param);
    }
    model->validate_nodes_and_infer_types();
    for (const auto& node : model->get_ops()) {
        OPENVINO_ASSERT(!ov::is_type<v13::ScaledDotProductAttention>(node) &&
                            !ov::is_type<ov::op::util::ReadValueBase>(node) &&
                            !ov::is_type<ov::op::util::AssignBase>(node),
                        "DFlash PA conversion left an unmanaged attention/cache node: ", node->get_friendly_name());
    }
}

}  // namespace ov::genai::utils::dflash
