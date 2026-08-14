// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

#include "eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

constexpr const char* DFLASH_HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers";
constexpr const char* LAST_HIDDEN_STATE_OUTPUT_NAME = "last_hidden_state";
constexpr const char* MUSE_TEXT_EMBEDDINGS_OUTPUT_NAME = "inputs_embeds";
constexpr const char* MUSE_VISION_EMBEDDINGS_OUTPUT_NAME = "last_hidden_state";

void add_dflash_hidden_state_result(std::shared_ptr<ov::Model>& model,
                                    const std::vector<ov::Output<ov::Node>>& hidden_state_outputs) {
    ov::Output<ov::Node> output_to_operate;
    if (hidden_state_outputs.size() > 1) {
        auto concat = std::make_shared<ov::op::v0::Concat>(hidden_state_outputs, -1);
        concat->set_friendly_name("dflash_hidden_states_concat");
        output_to_operate = concat->output(0);
    } else {
        output_to_operate = hidden_state_outputs[0];
    }

    auto result = std::make_shared<ov::op::v0::Result>(output_to_operate);
    result->output(0).set_names({LAST_HIDDEN_STATE_OUTPUT_NAME});
    result->set_friendly_name(LAST_HIDDEN_STATE_OUTPUT_NAME);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

bool is_rank3_hidden_output(const ov::Output<ov::Node>& output,
                            std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto shape = output.get_partial_shape();
    return shape.rank().is_static() && shape.rank().get_length() == 3 && shape[2].is_static() &&
           (!expected_hidden_size || static_cast<size_t>(shape[2].get_length()) == *expected_hidden_size);
}

std::shared_ptr<ov::Node> resolve_unique_live_node(const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
                                                   const std::string& producer) {
    std::shared_ptr<ov::Node> match;
    size_t node_count = 0;
    for (const auto& node : live_nodes) {
        if (node->get_friendly_name() == producer) {
            match = node;
            ++node_count;
        }
    }
    OPENVINO_ASSERT(node_count == 1, "Hidden-state producer ", producer, " resolved to ", node_count, " live nodes.");
    return match;
}

ov::Output<ov::Node> resolve_live_hidden_state_output(
    const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
    const std::string& producer,
    nlohmann::json::number_unsigned_t output_index,
    std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto node = resolve_unique_live_node(live_nodes, producer);
    OPENVINO_ASSERT(output_index < node->get_output_size(),
                    "Hidden-state producer ",
                    producer,
                    " has no output ",
                    output_index,
                    ".");
    const auto output = node->output(static_cast<size_t>(output_index));
    OPENVINO_ASSERT(is_rank3_hidden_output(output, expected_hidden_size),
                    "Hidden-state producer ",
                    producer,
                    " output ",
                    output_index,
                    " must be rank-3 with a static hidden width",
                    expected_hidden_size ? " matching the retained hidden width." : ".");
    return output;
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
        if (item.empty()) {
            continue;
        }
        std::istringstream item_stream(item);
        int32_t layer_id{};
        if (!(item_stream >> layer_id) || (item_stream >> std::ws && !item_stream.eof())) {
            OPENVINO_THROW("Malformed dflash_target_layer_ids RT info value: '",
                           raw,
                           "'. Invalid layer id: '",
                           item,
                           "'.");
        }
        result.push_back(layer_id);
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

std::shared_ptr<ov::op::v0::Parameter> find_inputs_embeds_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "inputs_embeds" ||
            parameter->output(0).get_names().count("inputs_embeds") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

bool outputs_match(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    return lhs.get_node_shared_ptr() == rhs.get_node_shared_ptr() && lhs.get_index() == rhs.get_index();
}

template <typename Op>
std::shared_ptr<Op> expect_op(const ov::Output<ov::Node>& output, const std::string& description) {
    auto op = ov::as_type_ptr<Op>(output.get_node_shared_ptr());
    OPENVINO_ASSERT(op, "Muse Glimmer embedding norm must contain ", description, ".");
    return op;
}

std::shared_ptr<ov::op::v0::Constant> expect_constant(const ov::Output<ov::Node>& output,
                                                       const std::string& description) {
    return expect_op<ov::op::v0::Constant>(output, description);
}

float get_single_f32_constant(const std::shared_ptr<ov::op::v0::Constant>& constant,
                              const std::string& description) {
    OPENVINO_ASSERT(constant->get_element_type() == ov::element::f32,
                    "Muse Glimmer embedding norm ",
                    description,
                    " must be FP32.");
    const auto values = constant->get_vector<float>();
    OPENVINO_ASSERT(values.size() == 1,
                    "Muse Glimmer embedding norm ",
                    description,
                    " must contain exactly one value.");
    return values.front();
}

void validate_last_axis(const std::shared_ptr<ov::op::v0::Constant>& axes) {
    std::vector<int64_t> values;
    if (axes->get_element_type() == ov::element::i64) {
        values = axes->get_vector<int64_t>();
    } else if (axes->get_element_type() == ov::element::i32) {
        const auto values_i32 = axes->get_vector<int32_t>();
        values.assign(values_i32.begin(), values_i32.end());
    } else {
        OPENVINO_THROW("Muse Glimmer embedding norm ReduceMean axes must be i32 or i64.");
    }
    OPENVINO_ASSERT(values == std::vector<int64_t>{-1},
                    "Muse Glimmer embedding norm ReduceMean must reduce only the last axis.");
}

struct MuseRmsNorm {
    std::shared_ptr<ov::op::v0::Result> result;
    ov::Output<ov::Node> raw_activation;
    std::shared_ptr<ov::op::v1::Power> square;
    std::shared_ptr<ov::op::v1::ReduceMean> reduce_mean;
    std::shared_ptr<ov::op::v1::Add> epsilon_add;
    std::shared_ptr<ov::op::v1::Power> inverse_rms;
    std::shared_ptr<ov::op::v1::Multiply> multiply;
    std::shared_ptr<ov::op::v0::Constant> square_exponent;
    std::shared_ptr<ov::op::v0::Constant> reduce_axes;
    std::shared_ptr<ov::op::v0::Constant> epsilon;
    std::shared_ptr<ov::op::v0::Constant> inverse_exponent;
    size_t multiply_raw_input_index = 0;
    size_t epsilon_add_reduce_input_index = 0;
    size_t hidden_size = 0;
    ov::PartialShape public_shape;
    ov::element::Type public_type;
    float square_exponent_value = 0.0f;
    float epsilon_value = 0.0f;
    float inverse_exponent_value = 0.0f;
};

std::shared_ptr<ov::op::v0::Result> find_unique_result(const std::shared_ptr<ov::Model>& model,
                                                        const std::string& output_name) {
    std::shared_ptr<ov::op::v0::Result> result;
    for (const auto& candidate : model->get_results()) {
        if (candidate->output(0).get_names().count(output_name) == 0) {
            continue;
        }
        OPENVINO_ASSERT(!result,
                        "Muse Glimmer component has multiple public outputs named '",
                        output_name,
                        "'.");
        result = candidate;
    }
    OPENVINO_ASSERT(result, "Muse Glimmer component must expose public output '", output_name, "'.");
    return result;
}

MuseRmsNorm match_muse_terminal_rms_norm(const std::shared_ptr<ov::Model>& model,
                                         const std::string& output_name,
                                         size_t expected_rank) {
    OPENVINO_ASSERT(model, "Muse Glimmer embedding component model cannot be null.");

    MuseRmsNorm pattern;
    pattern.result = find_unique_result(model, output_name);
    pattern.public_shape = pattern.result->output(0).get_partial_shape();
    pattern.public_type = pattern.result->output(0).get_element_type();

    pattern.multiply = expect_op<ov::op::v1::Multiply>(
        pattern.result->input_value(0), "terminal Multiply(raw_activation, inverse_rms)");
    OPENVINO_ASSERT(pattern.multiply->get_input_size() == 2,
                    "Muse Glimmer terminal RMSNorm Multiply must have two inputs.");

    const auto first_inverse = ov::as_type_ptr<ov::op::v1::Power>(
        pattern.multiply->input_value(0).get_node_shared_ptr());
    const auto second_inverse = ov::as_type_ptr<ov::op::v1::Power>(
        pattern.multiply->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(static_cast<bool>(first_inverse) != static_cast<bool>(second_inverse),
                    "Muse Glimmer terminal RMSNorm Multiply must have exactly one inverse-RMS Power input.");
    const size_t inverse_input_index = first_inverse ? 0 : 1;
    pattern.multiply_raw_input_index = inverse_input_index == 0 ? 1 : 0;
    pattern.inverse_rms = first_inverse ? first_inverse : second_inverse;
    pattern.raw_activation = pattern.multiply->input_value(pattern.multiply_raw_input_index);

    OPENVINO_ASSERT(pattern.inverse_rms->get_input_size() == 2,
                    "Muse Glimmer inverse-RMS Power must have two inputs.");
    pattern.epsilon_add =
        expect_op<ov::op::v1::Add>(pattern.inverse_rms->input_value(0), "inverse-RMS Power input Add");
    pattern.inverse_exponent =
        expect_constant(pattern.inverse_rms->input_value(1), "inverse-RMS Power exponent Constant");

    OPENVINO_ASSERT(pattern.epsilon_add->get_input_size() == 2,
                    "Muse Glimmer RMSNorm epsilon Add must have two inputs.");
    const auto first_reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(
        pattern.epsilon_add->input_value(0).get_node_shared_ptr());
    const auto second_reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(
        pattern.epsilon_add->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(static_cast<bool>(first_reduce) != static_cast<bool>(second_reduce),
                    "Muse Glimmer RMSNorm epsilon Add must have exactly one ReduceMean input.");
    pattern.epsilon_add_reduce_input_index = first_reduce ? 0 : 1;
    pattern.reduce_mean = first_reduce ? first_reduce : second_reduce;
    pattern.epsilon = expect_constant(
        pattern.epsilon_add->input_value(pattern.epsilon_add_reduce_input_index == 0 ? 1 : 0),
        "RMSNorm epsilon Constant");

    OPENVINO_ASSERT(pattern.reduce_mean->get_input_size() == 2,
                    "Muse Glimmer RMSNorm ReduceMean must have data and axes inputs.");
    OPENVINO_ASSERT(pattern.reduce_mean->get_keep_dims(),
                    "Muse Glimmer RMSNorm ReduceMean must preserve reduced dimensions.");
    pattern.square = expect_op<ov::op::v1::Power>(
        pattern.reduce_mean->input_value(0), "RMSNorm square Power");
    pattern.reduce_axes = expect_constant(pattern.reduce_mean->input_value(1), "RMSNorm ReduceMean axes Constant");

    OPENVINO_ASSERT(pattern.square->get_input_size() == 2,
                    "Muse Glimmer RMSNorm square Power must have two inputs.");
    OPENVINO_ASSERT(outputs_match(pattern.square->input_value(0), pattern.raw_activation),
                    "Muse Glimmer RMSNorm square Power must consume the terminal Multiply raw activation.");
    pattern.square_exponent = expect_constant(pattern.square->input_value(1), "RMSNorm square Power exponent Constant");

    pattern.square_exponent_value = get_single_f32_constant(pattern.square_exponent, "square exponent");
    pattern.epsilon_value = get_single_f32_constant(pattern.epsilon, "epsilon");
    pattern.inverse_exponent_value = get_single_f32_constant(pattern.inverse_exponent, "inverse exponent");
    OPENVINO_ASSERT(pattern.square_exponent_value == 2.0f,
                    "Muse Glimmer RMSNorm square exponent must equal 2.");
    OPENVINO_ASSERT(pattern.inverse_exponent_value == -0.5f,
                    "Muse Glimmer RMSNorm inverse exponent must equal -0.5.");
    validate_last_axis(pattern.reduce_axes);

    const auto raw_shape = pattern.raw_activation.get_partial_shape();
    OPENVINO_ASSERT(raw_shape.rank().is_static() && raw_shape.rank().get_length() == expected_rank,
                    "Muse Glimmer component '",
                    output_name,
                    "' raw activation must have rank ",
                    expected_rank,
                    ".");
    OPENVINO_ASSERT(raw_shape[expected_rank - 1].is_static(),
                    "Muse Glimmer component '",
                    output_name,
                    "' raw activation hidden width must be static.");
    pattern.hidden_size = static_cast<size_t>(raw_shape[expected_rank - 1].get_length());

    OPENVINO_ASSERT(pattern.raw_activation.get_element_type() == ov::element::f32 &&
                        pattern.square->output(0).get_element_type() == ov::element::f32 &&
                        pattern.reduce_mean->output(0).get_element_type() == ov::element::f32 &&
                        pattern.epsilon_add->output(0).get_element_type() == ov::element::f32 &&
                        pattern.inverse_rms->output(0).get_element_type() == ov::element::f32 &&
                        pattern.multiply->output(0).get_element_type() == ov::element::f32 &&
                        pattern.public_type == ov::element::f32,
                    "Muse Glimmer terminal RMSNorm must use FP32 activations and statistics.");
    OPENVINO_ASSERT(pattern.public_shape == raw_shape,
                    "Muse Glimmer component '",
                    output_name,
                    "' terminal RMSNorm must preserve its public output shape.");
    return pattern;
}

void validate_equivalent_norms(const MuseRmsNorm& text, const MuseRmsNorm& vision) {
    OPENVINO_ASSERT(text.hidden_size == vision.hidden_size,
                    "Muse Glimmer text and vision embedding hidden widths must match.");
    OPENVINO_ASSERT(text.square_exponent_value == vision.square_exponent_value &&
                        text.epsilon_value == vision.epsilon_value &&
                        text.inverse_exponent_value == vision.inverse_exponent_value,
                    "Muse Glimmer text and vision terminal RMSNorm constants must match.");
    OPENVINO_ASSERT(text.reduce_mean->get_keep_dims() == vision.reduce_mean->get_keep_dims(),
                    "Muse Glimmer text and vision terminal RMSNorm ReduceMean attributes must match.");
}

std::shared_ptr<ov::op::v0::Parameter> find_unique_inputs_embeds_parameter(
    const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v0::Parameter> parameter;
    for (const auto& candidate : model->get_parameters()) {
        if (candidate->output(0).get_names().count(MUSE_TEXT_EMBEDDINGS_OUTPUT_NAME) == 0) {
            continue;
        }
        OPENVINO_ASSERT(!parameter,
                        "Muse Glimmer language model has multiple public inputs named 'inputs_embeds'.");
        parameter = candidate;
    }
    OPENVINO_ASSERT(parameter, "Muse Glimmer language model must expose public input 'inputs_embeds'.");
    return parameter;
}

std::vector<ov::Input<ov::Node>> snapshot_live_consumers(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }

    std::vector<ov::Input<ov::Node>> consumers;
    for (auto consumer : parameter->output(0).get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            consumers.push_back(consumer);
        }
    }
    OPENVINO_ASSERT(!consumers.empty(),
                    "Muse Glimmer language-model inputs_embeds input has no live consumers.");
    return consumers;
}

std::shared_ptr<ov::op::v0::Constant> clone_constant(const std::shared_ptr<ov::op::v0::Constant>& constant) {
    return std::make_shared<ov::op::v0::Constant>(
        constant->get_element_type(), constant->get_shape(), constant->get_data_ptr());
}

ov::Output<ov::Node> clone_operation(const std::shared_ptr<ov::Node>& operation,
                                     const ov::OutputVector& inputs,
                                     const std::string& friendly_name) {
    auto clone = operation->clone_with_new_inputs(inputs);
    clone->set_friendly_name(friendly_name);
    return clone->output(0);
}

ov::Output<ov::Node> clone_muse_rms_norm_on_language_input(
    const MuseRmsNorm& norm,
    const std::shared_ptr<ov::op::v0::Parameter>& language_inputs_embeds) {
    const auto raw_input = language_inputs_embeds->output(0);

    ov::OutputVector square_inputs{norm.square->input_value(0), norm.square->input_value(1)};
    square_inputs[0] = raw_input;
    square_inputs[1] = clone_constant(norm.square_exponent)->output(0);
    const auto square = clone_operation(
        norm.square, square_inputs, "dflash_muse_embedding_norm_square");

    ov::OutputVector reduce_inputs{norm.reduce_mean->input_value(0), norm.reduce_mean->input_value(1)};
    reduce_inputs[0] = square;
    reduce_inputs[1] = clone_constant(norm.reduce_axes)->output(0);
    const auto reduce_mean = clone_operation(
        norm.reduce_mean, reduce_inputs, "dflash_muse_embedding_norm_reduce_mean");

    ov::OutputVector add_inputs{norm.epsilon_add->input_value(0), norm.epsilon_add->input_value(1)};
    add_inputs[norm.epsilon_add_reduce_input_index] = reduce_mean;
    add_inputs[norm.epsilon_add_reduce_input_index == 0 ? 1 : 0] = clone_constant(norm.epsilon)->output(0);
    const auto epsilon_add = clone_operation(
        norm.epsilon_add, add_inputs, "dflash_muse_embedding_norm_add_epsilon");

    ov::OutputVector inverse_inputs{norm.inverse_rms->input_value(0), norm.inverse_rms->input_value(1)};
    inverse_inputs[0] = epsilon_add;
    inverse_inputs[1] = clone_constant(norm.inverse_exponent)->output(0);
    const auto inverse_rms = clone_operation(
        norm.inverse_rms, inverse_inputs, "dflash_muse_embedding_norm_inverse_rms");

    ov::OutputVector multiply_inputs{norm.multiply->input_value(0), norm.multiply->input_value(1)};
    multiply_inputs[norm.multiply_raw_input_index] = raw_input;
    multiply_inputs[norm.multiply_raw_input_index == 0 ? 1 : 0] = inverse_rms;
    return clone_operation(norm.multiply, multiply_inputs, "dflash_muse_embedding_norm");
}

}  // namespace

std::optional<std::vector<DFlashHiddenStateLocator>> resolve_target_hidden_state_locators(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    if (!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY)) {
        return std::nullopt;
    }

    nlohmann::json annotation;
    try {
        annotation = nlohmann::json::parse(model->get_rt_info<std::string>(DFLASH_HIDDEN_STATES_RT_INFO_KEY));
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata in model rt_info: ", error.what());
    }

    std::set<std::pair<std::string, nlohmann::json::number_unsigned_t>> producer_outputs;
    std::vector<DFlashHiddenStateLocator> resolved;
    resolved.reserve(target_layer_ids.size());
    try {
        const auto& layers = annotation.at("layers");
        const auto live_nodes = model->get_ordered_ops();
        for (const auto layer_id : target_layer_ids) {
            const auto key = std::to_string(layer_id);
            const auto& locator = layers.at(key);
            const auto producer = locator.at("producer").get<std::string>();
            const auto output_index =
                locator.at("output_index").get_ref<const nlohmann::json::number_unsigned_t&>();
            OPENVINO_ASSERT(producer_outputs.emplace(producer, output_index).second,
                            "Duplicate producer/output locator for requested decoder layers: ",
                            producer,
                            "[",
                            output_index,
                            "].");

            const auto output = resolve_live_hidden_state_output(live_nodes, producer, output_index);
            resolved.push_back({producer, output, static_cast<size_t>(output.get_partial_shape()[2].get_length())});
        }
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata: ", error.what());
    }
    return resolved;
}

void expose_target_hidden_states(std::shared_ptr<ov::Model>& model,
                                 const std::optional<std::vector<DFlashHiddenStateLocator>>& retained_locators,
                                 const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    std::vector<ov::Output<ov::Node>> outputs;

    if (retained_locators) {
        OPENVINO_ASSERT(!retained_locators->empty(), "DFlash hidden-state locators cannot be empty.");
        OPENVINO_ASSERT(retained_locators->size() == target_layer_ids.size(),
                        "DFlash hidden-state locator count does not match target_layer_ids.");
        const auto live_nodes = model->get_ordered_ops();
        outputs.reserve(retained_locators->size());
        for (const auto& retained : *retained_locators) {
            const auto current_output = resolve_live_hidden_state_output(
                live_nodes,
                retained.producer,
                retained.output.get_index(),
                retained.hidden_size);
            OPENVINO_ASSERT(current_output.get_node_shared_ptr() == retained.output.get_node_shared_ptr(),
                            "DFlash hidden-state producer ",
                            retained.producer,
                            " no longer resolves to the retained output identity.");
            outputs.push_back(retained.output);
        }
    } else {
        OPENVINO_ASSERT(!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY),
                        "DFlash hidden-state fallback cannot be used when target metadata is present.");
        outputs = eagle3::find_decoder_layer_hidden_state_outputs(model, target_layer_ids);
        OPENVINO_ASSERT(outputs.size() == target_layer_ids.size(),
                        "DFlash target model has no hidden-state RT info and Eagle3 fallback found ",
                        outputs.size(),
                        " outputs for ",
                        target_layer_ids.size(),
                        " requested target layers.");
    }
    add_dflash_hidden_state_result(model, outputs);
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_mode") || !model->get_rt_info<bool>("dflash_mode")) {
        return;
    }

    properties["dflash_mode"] = true;
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
    if (!info.dflash_mode) {
        return info;
    }

    auto mask_it = config.find("dflash_mask_token_id");
    OPENVINO_ASSERT(mask_it != config.end(), "DFlash draft model is missing dflash_mask_token_id RT info.");
    info.mask_token_id = mask_it->second.as<int64_t>();
    config.erase(mask_it);

    auto layers_it = config.find("dflash_target_layer_ids");
    OPENVINO_ASSERT(layers_it != config.end(), "DFlash draft model is missing dflash_target_layer_ids RT info.");
    info.target_layer_ids = layers_it->second.as<std::vector<int32_t>>();
    config.erase(layers_it);
    OPENVINO_ASSERT(!info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    return info;
}

void hoist_muse_glimmer_embedding_norms(const std::shared_ptr<ov::Model>& language_model,
                                        const std::shared_ptr<ov::Model>& text_embeddings_model,
                                        const std::shared_ptr<ov::Model>& vision_embeddings_model,
                                        float scale_emb) {
    OPENVINO_ASSERT(language_model && text_embeddings_model && vision_embeddings_model,
                    "Muse Glimmer DFlash embedding-norm relocation requires language, text, and vision models.");
    OPENVINO_ASSERT(scale_emb == 1.0f,
                    "Muse Glimmer DFlash embedding-norm relocation requires scale_emb == 1, got ",
                    scale_emb,
                    ".");

    // Inspect every graph before changing any of them.
    const auto text_norm =
        match_muse_terminal_rms_norm(text_embeddings_model, MUSE_TEXT_EMBEDDINGS_OUTPUT_NAME, 3);
    const auto vision_norm =
        match_muse_terminal_rms_norm(vision_embeddings_model, MUSE_VISION_EMBEDDINGS_OUTPUT_NAME, 2);
    validate_equivalent_norms(text_norm, vision_norm);

    const auto language_inputs_embeds = find_unique_inputs_embeds_parameter(language_model);
    const auto language_shape = language_inputs_embeds->output(0).get_partial_shape();
    OPENVINO_ASSERT(language_shape.rank().is_static() && language_shape.rank().get_length() == 3,
                    "Muse Glimmer language-model inputs_embeds input must be rank 3.");
    OPENVINO_ASSERT(language_shape[2].is_static() &&
                        static_cast<size_t>(language_shape[2].get_length()) == text_norm.hidden_size,
                    "Muse Glimmer language-model inputs_embeds hidden width must match component embeddings.");
    OPENVINO_ASSERT(language_inputs_embeds->output(0).get_element_type() == ov::element::f32,
                    "Muse Glimmer language-model inputs_embeds input must be FP32.");
    const auto original_language_consumers = snapshot_live_consumers(language_model, language_inputs_embeds);

    // Only begin rewiring after all structural checks above have passed.
    const auto hoisted_norm = clone_muse_rms_norm_on_language_input(text_norm, language_inputs_embeds);
    text_norm.result->input(0).replace_source_output(text_norm.raw_activation);
    vision_norm.result->input(0).replace_source_output(vision_norm.raw_activation);
    for (auto consumer : original_language_consumers) {
        consumer.replace_source_output(hoisted_norm);
    }

    text_embeddings_model->validate_nodes_and_infer_types();
    vision_embeddings_model->validate_nodes_and_infer_types();
    language_model->validate_nodes_and_infer_types();

    OPENVINO_ASSERT(text_norm.result->output(0).get_partial_shape() == text_norm.public_shape &&
                        text_norm.result->output(0).get_element_type() == text_norm.public_type,
                    "Muse Glimmer text embedding public output changed during norm relocation.");
    OPENVINO_ASSERT(vision_norm.result->output(0).get_partial_shape() == vision_norm.public_shape &&
                        vision_norm.result->output(0).get_element_type() == vision_norm.public_type,
                    "Muse Glimmer vision embedding public output changed during norm relocation.");
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

void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash lm_head graft requires both target and draft models.");

    auto target_head = std::get<0>(ov::genai::utils::find_llm_matmul(main_model));
    auto target_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(target_head);
    OPENVINO_ASSERT(target_matmul,
                    "DFlash: could not locate the target lm_head MatMul to graft onto the draft.");

    // Clone ONLY the weight side (input(1)); input(0) is the target activation and is not reused.
    // Cloning recursively carries any INT4 decompression subgraph intact.
    auto weight_source = target_matmul->input_value(1);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target lm_head weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    std::shared_ptr<ov::op::v0::Result> hidden_result;
    for (const auto& result : draft_model->get_results()) {
        if (result->output(0).get_names().count(LAST_HIDDEN_STATE_OUTPUT_NAME) != 0 ||
            result->get_friendly_name() == LAST_HIDDEN_STATE_OUTPUT_NAME) {
            hidden_result = result;
            break;
        }
    }
    OPENVINO_ASSERT(hidden_result,
                    "DFlash draft model must expose a 'last_hidden_state' output to graft the target lm_head.");

    // Feed the draft post-norm hidden into a bare MatMul with the target head weight + transpose flags.
    auto draft_hidden = hidden_result->input_value(0);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(draft_hidden,
                                                              cloned_weight_out,
                                                              target_matmul->get_transpose_a(),
                                                              target_matmul->get_transpose_b());
    logits_matmul->set_friendly_name("dflash_grafted_lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");
    logits_result->get_rt_info()["manually_added_output"] = true;

    draft_model->add_results({logits_result});
    draft_model->remove_result(hidden_result);
    draft_model->validate_nodes_and_infer_types();
}

void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash embedding attach requires both target and draft models.");

    auto gather = eagle3::find_embedding_gather(main_model);
    OPENVINO_ASSERT(gather,
                    "DFlash: could not find the target token-embedding Gather to attach onto the draft.");

    // Clone the embedding weight (input(0)). The recursive clone is precision-agnostic: it copies a
    // plain f16/f32 Constant or an int8/int4 dequant chain intact, so the draft owns its embedding.
    auto weight_source = gather->input_value(0);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target embedding weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    auto inputs_embeds_param = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds_param,
                    "DFlash draft model must expose an 'inputs_embeds' input to attach the target embedding.");

    // Snapshot the live consumers of inputs_embeds before rewiring them onto the new Gather.
    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : draft_model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }
    std::vector<ov::Input<ov::Node>> original_consumers;
    for (auto consumer : inputs_embeds_param->output(0).get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            original_consumers.push_back(consumer);
        }
    }
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash draft inputs_embeds input has no live consumers.");

    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto embed_gather = std::make_shared<ov::op::v8::Gather>(cloned_weight_out, input_ids, axis);
    embed_gather->set_friendly_name("dflash_draft_embedding");

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(embed_gather->output(0));
    }
    draft_model->add_parameters({input_ids});
    draft_model->remove_parameter(inputs_embeds_param);
    draft_model->validate_nodes_and_infer_types();
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
