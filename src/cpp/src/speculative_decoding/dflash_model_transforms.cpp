// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

constexpr const char* DFLASH_HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers";
constexpr const char* DFLASH_LAST_HIDDEN_STATE_OUTPUT_NAME = "last_hidden_state";

std::optional<ov::Output<ov::Node>> find_dflash_output_by_tensor_name(const std::shared_ptr<ov::Model>& model,
                                                                      const std::string& tensor_name) {
    for (const auto& node : model->get_ordered_ops()) {
        for (const auto& output : node->outputs()) {
            if (output.get_names().count(tensor_name) != 0) {
                return output;
            }
        }
    }
    return std::nullopt;
}

void add_dflash_hidden_state_result(std::shared_ptr<ov::Model>& model,
                                    const std::vector<ov::Output<ov::Node>>& hidden_state_outputs) {
    std::shared_ptr<ov::Node> node_to_operate;
    if (hidden_state_outputs.size() > 1) {
        auto concat = std::make_shared<ov::op::v0::Concat>(hidden_state_outputs, -1);
        concat->set_friendly_name("dflash_hidden_states_concat");
        node_to_operate = concat;
    } else {
        node_to_operate = hidden_state_outputs[0].get_node_shared_ptr();
    }

    auto result = std::make_shared<ov::op::v0::Result>(node_to_operate);
    result->output(0).set_names({DFLASH_LAST_HIDDEN_STATE_OUTPUT_NAME});
    result->set_friendly_name(DFLASH_LAST_HIDDEN_STATE_OUTPUT_NAME);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

std::vector<ov::Output<ov::Node>> get_dflash_annotated_hidden_state_outputs(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids) {
    if (!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY)) {
        return {};
    }

    const auto annotation = nlohmann::json::parse(model->get_rt_info<std::string>(DFLASH_HIDDEN_STATES_RT_INFO_KEY));
    OPENVINO_ASSERT(annotation.contains("layers") && annotation["layers"].is_object(),
                    "Invalid hidden-state annotation metadata in model rt_info.");

    std::vector<ov::Output<ov::Node>> outputs;
    outputs.reserve(target_layer_ids.size());
    for (const auto layer_idx : target_layer_ids) {
        const auto key = std::to_string(layer_idx);
        OPENVINO_ASSERT(annotation["layers"].contains(key),
                        "Missing hidden-state annotation for decoder layer ",
                        layer_idx,
                        ".");
        const auto tensor_name = annotation["layers"].at(key).get<std::string>();
        auto output = find_dflash_output_by_tensor_name(model, tensor_name);
        OPENVINO_ASSERT(output.has_value(),
                        "Hidden-state tensor annotated for decoder layer ",
                        layer_idx,
                        " was not found in the model graph: ",
                        tensor_name);
        outputs.push_back(*output);
    }
    return outputs;
}

template <typename T>
std::optional<T> get_rt_info_value(const std::shared_ptr<ov::Model>& model, const std::vector<std::string>& path) {
    if (!model->has_rt_info(path)) {
        return std::nullopt;
    }
    return model->get_rt_info<T>(path);
}

std::vector<int32_t> parse_layer_ids(const std::string& raw) {
    std::vector<int32_t> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            result.push_back(static_cast<int32_t>(std::stoi(item)));
        }
    }
    return result;
}

std::shared_ptr<ov::op::v0::Parameter> find_hidden_states_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "hidden_states" ||
            parameter->output(0).get_names().count("hidden_states") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Result> find_dflash_hidden_state_result(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v0::Result> result;
    for (const auto& candidate : model->get_results()) {
        if (candidate->get_friendly_name() == DFLASH_LAST_HIDDEN_STATE_OUTPUT_NAME ||
            candidate->output(0).get_names().count(DFLASH_LAST_HIDDEN_STATE_OUTPUT_NAME) != 0) {
            OPENVINO_ASSERT(!result, "DFlash target model has multiple last_hidden_state results.");
            result = candidate;
        }
    }
    return result;
}

bool output_depends_on(const ov::Output<ov::Node>& output,
                       const ov::Output<ov::Node>& dependency,
                       std::unordered_set<const ov::Node*>& visited) {
    if (output.get_node() == dependency.get_node() && output.get_index() == dependency.get_index()) {
        return true;
    }

    const auto* node = output.get_node();
    if (!visited.insert(node).second) {
        return false;
    }

    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        if (output_depends_on(node->get_input_source_output(input_idx), dependency, visited)) {
            return true;
        }
    }
    return false;
}

bool output_depends_on(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& dependency) {
    std::unordered_set<const ov::Node*> visited;
    return output_depends_on(output, dependency, visited);
}

bool node_depends_on(const std::shared_ptr<ov::Node>& node, const ov::Output<ov::Node>& dependency) {
    for (const auto& output : node->outputs()) {
        if (output_depends_on(output, dependency)) {
            return true;
        }
    }
    return false;
}

bool partial_shapes_have_same_rank_and_last_dim(const ov::PartialShape& lhs, const ov::PartialShape& rhs) {
    if (lhs.rank().is_dynamic() || rhs.rank().is_dynamic() ||
        lhs.rank().get_length() != rhs.rank().get_length() ||
        lhs.rank().get_length() == 0) {
        return false;
    }

    const auto last_idx = static_cast<size_t>(lhs.rank().get_length() - 1);
    return lhs[last_idx].compatible(rhs[last_idx]);
}

int64_t require_static_dimension(const ov::Dimension& dimension, const std::string& message) {
    OPENVINO_ASSERT(dimension.is_static(), message);
    return dimension.get_length();
}

std::shared_ptr<ov::op::v0::MatMul> find_dflash_projection_fc(const std::shared_ptr<ov::Model>& model,
                                                              const ov::Output<ov::Node>& hidden_states) {
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> candidates;
    for (const auto& node : model->get_ordered_ops()) {
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (!matmul || matmul->get_input_size() < 2) {
            continue;
        }

        const bool data_depends_on_hidden = output_depends_on(matmul->input_value(0), hidden_states);
        const bool weights_depend_on_hidden = output_depends_on(matmul->input_value(1), hidden_states);
        if (!data_depends_on_hidden || weights_depend_on_hidden) {
            continue;
        }

        const auto input_shape = matmul->input_value(0).get_partial_shape();
        const auto output_shape = matmul->output(0).get_partial_shape();
        if (input_shape.rank().is_static() && output_shape.rank().is_static() &&
            input_shape.rank().get_length() == output_shape.rank().get_length() &&
            input_shape.rank().get_length() > 0) {
            const auto last_idx = static_cast<size_t>(input_shape.rank().get_length() - 1);
            if (input_shape[last_idx].is_static() && output_shape[last_idx].is_static() &&
                input_shape[last_idx].get_length() <= output_shape[last_idx].get_length()) {
                continue;
            }
        }

        candidates.push_back(matmul);
    }

    OPENVINO_ASSERT(!candidates.empty(),
                    "Failed to locate DFlash draft hidden_states projection MatMul.");
    if (candidates.size() == 1) {
        return candidates.front();
    }

    std::vector<std::shared_ptr<ov::op::v0::MatMul>> named_candidates;
    for (const auto& candidate : candidates) {
        if (candidate->get_friendly_name().find("fc") != std::string::npos) {
            named_candidates.push_back(candidate);
        }
    }
    OPENVINO_ASSERT(named_candidates.size() == 1,
                    "DFlash draft hidden_states projection MatMul is ambiguous.");
    return named_candidates.front();
}

std::pair<int64_t, int64_t> get_dflash_projection_dimensions(const std::shared_ptr<ov::op::v0::MatMul>& fc) {
    const auto weights_shape = fc->input_value(1).get_partial_shape();
    OPENVINO_ASSERT(weights_shape.rank().is_static() && weights_shape.rank().get_length() == 2,
                    "DFlash hidden_states projection MatMul weights must have rank 2.");

    const auto input_dim_idx = fc->get_transpose_b() ? 1 : 0;
    const auto output_dim_idx = fc->get_transpose_b() ? 0 : 1;
    const auto raw_hidden_dim =
        require_static_dimension(weights_shape[input_dim_idx],
                                 "DFlash hidden_states projection input feature dimension must be static.");
    const auto projected_hidden_dim =
        require_static_dimension(weights_shape[output_dim_idx],
                                 "DFlash hidden_states projection output feature dimension must be static.");

    OPENVINO_ASSERT(raw_hidden_dim > projected_hidden_dim,
                    "DFlash hidden_states projection must reduce the target hidden-state feature dimension, got ",
                    raw_hidden_dim,
                    " -> ",
                    projected_hidden_dim,
                    ".");
    return {raw_hidden_dim, projected_hidden_dim};
}

std::shared_ptr<ov::Node> find_dflash_hidden_norm_output(const std::shared_ptr<ov::Model>& model,
                                                         const ov::Output<ov::Node>& fc_output) {
    std::vector<std::shared_ptr<ov::Node>> candidates;
    const auto fc_shape = fc_output.get_partial_shape();

    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name().find("hidden_norm") == std::string::npos ||
            !node_depends_on(node, fc_output)) {
            continue;
        }
        for (const auto& output : node->outputs()) {
            if (partial_shapes_have_same_rank_and_last_dim(output.get_partial_shape(), fc_shape)) {
                candidates.push_back(node);
                break;
            }
        }
    }

    OPENVINO_ASSERT(!candidates.empty(),
                    "Failed to locate DFlash draft hidden_norm output after hidden_states projection.");
    return candidates.back();
}

std::shared_ptr<ov::Node> extend_through_layout_preserving_ops(std::shared_ptr<ov::Node> node) {
    while (node && node->get_output_size() == 1) {
        const auto targets = node->output(0).get_target_inputs();
        if (targets.size() != 1) {
            break;
        }

        auto consumer = targets.begin()->get_node()->shared_from_this();
        if (consumer->get_input_size() != 1 ||
            consumer->get_type_name() != std::string("Convert") ||
            !partial_shapes_have_same_rank_and_last_dim(consumer->output(0).get_partial_shape(),
                                                        node->output(0).get_partial_shape())) {
            break;
        }
        node = consumer;
    }
    return node;
}

std::shared_ptr<ov::Node> clone_subgraph_with_replacement(
    const std::shared_ptr<ov::Node>& node,
    const std::unordered_map<const ov::Node*, ov::Output<ov::Node>>& replacements,
    std::unordered_map<const ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) {
    if (auto replacement = replacements.find(node.get()); replacement != replacements.end()) {
        return replacement->second.get_node_shared_ptr();
    }
    if (auto cloned = cloned_nodes.find(node.get()); cloned != cloned_nodes.end()) {
        return cloned->second;
    }

    OPENVINO_ASSERT(!ov::is_type<ov::op::v0::Parameter>(node),
                    "DFlash projection subgraph unexpectedly depends on input parameter '",
                    node->get_friendly_name(),
                    "'. Only hidden_states may be replaced.");

    ov::OutputVector cloned_inputs;
    cloned_inputs.reserve(node->get_input_size());
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        const auto source_output = node->get_input_source_output(input_idx);
        const auto source_node = source_output.get_node_shared_ptr();
        if (auto replacement = replacements.find(source_node.get()); replacement != replacements.end()) {
            OPENVINO_ASSERT(source_output.get_index() == replacement->second.get_index(),
                            "DFlash projection replacement output index mismatch.");
            cloned_inputs.push_back(replacement->second);
            continue;
        }

        auto cloned_input_node = clone_subgraph_with_replacement(source_node, replacements, cloned_nodes);
        cloned_inputs.push_back(cloned_input_node->output(source_output.get_index()));
    }

    auto cloned = node->clone_with_new_inputs(cloned_inputs);
    cloned->set_friendly_name(node->get_friendly_name() + "_dflash_target_projection");
    cloned_nodes[node.get()] = cloned;
    return cloned;
}

}  // namespace

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_mode") || !model->get_rt_info<bool>("dflash_mode")) {
        return;
    }

    properties["dflash_mode"] = true;
    if (auto block_size = get_rt_info_value<std::string>(model, {"dflash", "block_size"})) {
        properties["dflash_block_size"] = static_cast<int64_t>(std::stoll(*block_size));
    }
    if (auto mask_token_id = get_rt_info_value<std::string>(model, {"dflash", "mask_token_id"})) {
        properties["dflash_mask_token_id"] = static_cast<int64_t>(std::stoll(*mask_token_id));
    }
    if (auto target_layer_ids = get_rt_info_value<std::string>(model, {"dflash", "target_layer_ids"})) {
        properties["dflash_target_layer_ids"] = parse_layer_ids(*target_layer_ids);
    }
}

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config) {
    DFlashRTInfo info;
    auto mode_it = config.find("dflash_mode");
    if (mode_it == config.end()) {
        return info;
    }

    info.dflash_mode = mode_it->second.as<bool>();
    config.erase(mode_it);

    auto block_it = config.find("dflash_block_size");
    OPENVINO_ASSERT(block_it != config.end(), "DFlash draft model is missing dflash/block_size RT info.");
    info.block_size = static_cast<size_t>(block_it->second.as<int64_t>());
    config.erase(block_it);

    auto mask_it = config.find("dflash_mask_token_id");
    OPENVINO_ASSERT(mask_it != config.end(), "DFlash draft model is missing dflash/mask_token_id RT info.");
    info.mask_token_id = mask_it->second.as<int64_t>();
    config.erase(mask_it);

    auto layers_it = config.find("dflash_target_layer_ids");
    OPENVINO_ASSERT(layers_it != config.end(), "DFlash draft model is missing dflash/target_layer_ids RT info.");
    info.target_layer_ids = layers_it->second.as<std::vector<int32_t>>();
    config.erase(layers_it);
    OPENVINO_ASSERT(!info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    return info;
}

void expose_target_hidden_states(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    auto annotated_outputs = get_dflash_annotated_hidden_state_outputs(model, target_layer_ids);
    OPENVINO_ASSERT(!annotated_outputs.empty(),
                    "DFlash requires hidden-state annotations in the target model. "
                    "Export the target model with Optimum hidden-state annotations.");
    add_dflash_hidden_state_result(model, annotated_outputs);
}

void move_hidden_state_projection_to_target(const std::shared_ptr<ov::Model>& draft_model,
                                            const std::shared_ptr<ov::Model>& target_model) {
    OPENVINO_ASSERT(draft_model, "DFlash draft model cannot be null.");
    OPENVINO_ASSERT(target_model, "DFlash target model cannot be null.");

    auto hidden_states = find_hidden_states_parameter(draft_model);
    OPENVINO_ASSERT(hidden_states, "DFlash draft model must have 'hidden_states' input.");
    auto target_hidden_result = find_dflash_hidden_state_result(target_model);
    OPENVINO_ASSERT(target_hidden_result,
                    "DFlash target model must expose last_hidden_state before moving hidden projection.");

    auto fc = find_dflash_projection_fc(draft_model, hidden_states->output(0));
    const auto [raw_hidden_dim, projected_hidden_dim] = get_dflash_projection_dimensions(fc);
    auto projected_hidden = extend_through_layout_preserving_ops(
        find_dflash_hidden_norm_output(draft_model, fc->output(0)));

    const auto draft_input_shape = hidden_states->get_partial_shape();
    const auto target_hidden_shape = target_hidden_result->input_value(0).get_partial_shape();
    const auto projected_shape = projected_hidden->output(0).get_partial_shape();
    OPENVINO_ASSERT(draft_input_shape.rank().is_static() && draft_input_shape.rank().get_length() == 3,
                    "DFlash draft hidden_states input must have rank 3 before moving projection.");
    OPENVINO_ASSERT(target_hidden_shape.rank().is_static() && target_hidden_shape.rank().get_length() == 3,
                    "DFlash target last_hidden_state output must have rank 3 before moving projection.");
    OPENVINO_ASSERT(target_hidden_shape[1].compatible(1),
                    "DFlash target last_hidden_state must use CB layout [seq_len, 1, hidden].");
    OPENVINO_ASSERT(target_hidden_shape[2].compatible(draft_input_shape[2]),
                    "DFlash target hidden-state feature dimension must match draft hidden_states input before projection.");
    OPENVINO_ASSERT(target_hidden_shape[2].compatible(raw_hidden_dim),
                    "DFlash target hidden-state feature dimension must match projection input dimension, got ",
                    target_hidden_shape[2],
                    " and ",
                    raw_hidden_dim,
                    ".");
    OPENVINO_ASSERT(projected_shape.rank().is_static() && projected_shape.rank().get_length() == 3,
                    "DFlash projected hidden_states output must have rank 3.");
    OPENVINO_ASSERT(projected_shape[0].compatible(1),
                    "DFlash projected hidden_states output must use draft layout [1, seq_len, hidden].");
    OPENVINO_ASSERT(projected_shape[2].compatible(projected_hidden_dim),
                    "DFlash projected hidden-state feature dimension must match projection output dimension, got ",
                    projected_shape[2],
                    " and ",
                    projected_hidden_dim,
                    ".");

    auto target_hidden_cb_layout = target_hidden_result->input_value(0);
    std::unordered_map<const ov::Node*, ov::Output<ov::Node>> replacements;
    replacements.emplace(hidden_states.get(), target_hidden_cb_layout);
    std::unordered_map<const ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_projection = clone_subgraph_with_replacement(projected_hidden, replacements, cloned_nodes);

    const auto cloned_projection_shape = cloned_projection->output(0).get_partial_shape();
    OPENVINO_ASSERT(cloned_projection_shape.rank().is_static() && cloned_projection_shape.rank().get_length() == 3,
                    "DFlash target-side projected hidden_states output must have rank 3.");
    OPENVINO_ASSERT(cloned_projection_shape[1].compatible(1),
                    "DFlash target-side projected hidden_states output must keep CB layout [seq_len, 1, hidden].");
    OPENVINO_ASSERT(cloned_projection_shape[2].compatible(projected_hidden_dim),
                    "DFlash target-side projected hidden-state feature dimension must match projection output dimension.");
    target_hidden_result->input(0).replace_source_output(cloned_projection->output(0));

    auto original_projected_consumers = projected_hidden->output(0).get_target_inputs();
    OPENVINO_ASSERT(!original_projected_consumers.empty(),
                    "DFlash draft projected hidden_states output has no consumers.");
    hidden_states->set_partial_shape(projected_shape);
    hidden_states->set_element_type(projected_hidden->output(0).get_element_type());
    hidden_states->validate_and_infer_types();
    OPENVINO_ASSERT(hidden_states->get_partial_shape()[2].compatible(projected_hidden_dim),
                    "DFlash draft hidden_states input was not compacted to the projected hidden size.");
    for (auto consumer : original_projected_consumers) {
        consumer.replace_source_output(hidden_states->output(0));
    }

    target_model->validate_nodes_and_infer_types();
    draft_model->validate_nodes_and_infer_types();
}

void reshape_draft_hidden_states_input_for_cb(std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash draft model cannot be null.");

    auto hidden_states = find_hidden_states_parameter(model);
    OPENVINO_ASSERT(hidden_states, "DFlash draft model must have 'hidden_states' input.");

    const auto draft_shape = hidden_states->get_partial_shape();
    OPENVINO_ASSERT(draft_shape.rank().is_static() && draft_shape.rank().get_length() == 3,
                    "DFlash draft hidden_states input must have rank 3.");
    OPENVINO_ASSERT(draft_shape[0].is_dynamic() || draft_shape[0].get_length() == 1,
                    "DFlash draft hidden_states input must use exported shape [1, seq_len, hidden].");

    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }
    std::vector<ov::Input<ov::Node>> original_consumers;
    for (auto consumer : hidden_states->output(0).get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            original_consumers.push_back(consumer);
        }
    }
    OPENVINO_ASSERT(!original_consumers.empty(),
                    "DFlash draft hidden_states input has no live consumers.");
    hidden_states->set_partial_shape(ov::PartialShape({draft_shape[1], ov::Dimension(1), draft_shape[2]}));

    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                     ov::Shape{3},
                                                     std::vector<int64_t>{1, -1, 0});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, reshape_shape, true);
    reshape->set_friendly_name("dflash_hidden_states_cb_to_draft_layout");

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(reshape->output(0));
    }
    model->validate_nodes_and_infer_types();
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
