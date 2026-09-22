// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"

namespace ov::genai::dflash_cb {

inline constexpr size_t DEFAULT_NUM_ASSISTANT_TOKENS = 5;

inline void copy_tensor_bytes(const ov::Tensor& src, ov::Tensor& dst) {
    OPENVINO_ASSERT(src.get_element_type() == dst.get_element_type(),
                    "DFlash hidden state copy requires matching tensor element types.");
    OPENVINO_ASSERT(src.get_byte_size() == dst.get_byte_size(),
                    "DFlash hidden state copy requires matching tensor byte sizes.");
    src.copy_to(dst);
}

inline ov::Tensor truncate_normalized_hidden_state_from_end(const ov::Tensor& hidden_state, size_t tokens_to_remove) {
    if (!hidden_state || hidden_state.get_size() == 0 || tokens_to_remove == 0) {
        return hidden_state;
    }

    auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                    "DFlash hidden_states delta must have shape [seq_len, 1, hidden].");
    const size_t current_seq_len = shape[0];
    if (tokens_to_remove >= current_seq_len) {
        shape[0] = 0;
        return ov::Tensor(hidden_state.get_element_type(), shape);
    }

    ov::Coordinate start_coord(shape.size(), 0);
    ov::Coordinate end_coord(shape.begin(), shape.end());
    end_coord[0] = current_seq_len - tokens_to_remove;
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

class HiddenDeltaBuffer {
public:
    static constexpr size_t INITIAL_CHUNK_CAPACITY = 10;

    HiddenDeltaBuffer() {
        m_chunks.reserve(INITIAL_CHUNK_CAPACITY);
    }

    void append(const ov::Tensor& hidden_delta, bool copy_data = false) {
        if (!hidden_delta || hidden_delta.get_size() == 0) {
            return;
        }
        const auto shape = hidden_delta.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                        "DFlash hidden delta buffer expects [seq_len, 1, hidden] chunks.");
        const size_t token_count = shape[0];
        if (token_count == 0) {
            return;
        }

        if (copy_data) {
            ov::Tensor owned(hidden_delta.get_element_type(), shape);
            copy_tensor_bytes(hidden_delta, owned);
            m_chunks.push_back(owned);
        } else {
            m_chunks.push_back(hidden_delta);
        }
        m_token_count += token_count;
    }

    bool empty() const {
        return m_token_count == 0;
    }

    size_t token_count() const {
        return m_token_count;
    }

    ov::Tensor materialize() const {
        OPENVINO_ASSERT(m_token_count > 0, "Cannot materialize empty DFlash hidden deltas.");
        OPENVINO_ASSERT(!m_chunks.empty(), "DFlash hidden delta chunks are empty.");

        if (m_chunks.size() == 1) {
            const auto& chunk = m_chunks.front();
            OPENVINO_ASSERT(chunk && chunk.get_size() > 0,
                            "DFlash single hidden delta chunk is empty.");
            return chunk;
        }

        auto merged_shape = m_chunks.front().get_shape();
        OPENVINO_ASSERT(merged_shape.size() == 3 && merged_shape[1] == 1,
                        "DFlash hidden delta buffer expects [seq_len, 1, hidden] chunks.");
        merged_shape[0] = m_token_count;
        ov::Tensor merged(m_chunks.front().get_element_type(), merged_shape);
        size_t offset = 0;
        for (const auto& chunk : m_chunks) {
            const auto chunk_shape = chunk.get_shape();
            OPENVINO_ASSERT(chunk_shape.size() == 3 && chunk_shape[1] == 1 && chunk_shape[2] == merged_shape[2],
                            "Cannot merge DFlash hidden deltas with incompatible shape.");
            const size_t chunk_tokens = chunk_shape[0];
            ov::Tensor dst(merged,
                           ov::Coordinate{offset, 0, 0},
                           ov::Coordinate{offset + chunk_tokens, 1, merged_shape[2]});
            copy_tensor_bytes(chunk, dst);
            offset += chunk_tokens;
        }
        OPENVINO_ASSERT(offset == m_token_count, "DFlash hidden delta token count mismatch.");
        return merged;
    }

    void clear() {
        m_chunks.clear();
        m_token_count = 0;
    }

private:
    std::vector<ov::Tensor> m_chunks;
    size_t m_token_count = 0;
};

using PerLayerEmbeddingsCallback = std::function<ov::Tensor(const ov::Tensor&)>;

namespace detail {

inline bool tensors_equal(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!lhs || !rhs) {
        return !lhs && !rhs;
    }
    if (lhs.get_element_type() != rhs.get_element_type() || lhs.get_shape() != rhs.get_shape() ||
        lhs.get_byte_size() != rhs.get_byte_size()) {
        return false;
    }
    return lhs.get_byte_size() == 0 || std::memcmp(lhs.data(), rhs.data(), lhs.get_byte_size()) == 0;
}

inline bool embedding_rows_equal(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

inline std::optional<int64_t> token_type_at(const std::optional<std::vector<int64_t>>& token_types,
                                            size_t index,
                                            size_t prompt_length) {
    if (index >= prompt_length || !token_types) {
        return int64_t{0};
    }
    if (index >= token_types->size()) {
        return std::nullopt;
    }
    return (*token_types)[index];
}

inline bool is_visual_row(const std::optional<std::vector<int64_t>>& token_types, size_t index, size_t prompt_length) {
    const auto token_type = token_type_at(token_types, index, prompt_length);
    return token_type.has_value() && *token_type != 0;
}

inline size_t visual_span_start(const std::optional<std::vector<int64_t>>& token_types,
                                size_t index,
                                size_t prompt_length) {
    if (!is_visual_row(token_types, index, prompt_length)) {
        return index;
    }
    while (index > 0 && is_visual_row(token_types, index - 1, prompt_length)) {
        --index;
    }
    return index;
}

inline size_t visual_span_end(const std::optional<std::vector<int64_t>>& token_types,
                              size_t index,
                              size_t prompt_length) {
    while (index < prompt_length && is_visual_row(token_types, index, prompt_length)) {
        ++index;
    }
    return index;
}

inline bool has_per_layer_inputs(const ov::Tensor& tensor) {
    return tensor && tensor.get_size() != 0;
}

inline bool has_valid_per_layer_layout(const ov::Tensor& tensor) {
    if (!tensor) {
        return false;
    }
    const auto shape = tensor.get_shape();
    return tensor.get_element_type() == ov::element::f32 && shape.size() == 4 && shape[0] == 1;
}

inline bool per_layer_layouts_match(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!has_valid_per_layer_layout(lhs) || !has_valid_per_layer_layout(rhs)) {
        return false;
    }
    const auto lhs_shape = lhs.get_shape();
    const auto rhs_shape = rhs.get_shape();
    return lhs_shape[2] == rhs_shape[2] && lhs_shape[3] == rhs_shape[3];
}

inline bool per_layer_rows_equal(const ov::Tensor& lhs, size_t lhs_row, const ov::Tensor& rhs, size_t rhs_row) {
    if (!per_layer_layouts_match(lhs, rhs)) {
        return false;
    }
    const auto lhs_shape = lhs.get_shape();
    const auto rhs_shape = rhs.get_shape();
    if (lhs_row >= lhs_shape[1] || rhs_row >= rhs_shape[1]) {
        return false;
    }
    const size_t row_elements = lhs_shape[2] * lhs_shape[3];
    const auto* lhs_data = lhs.data<const float>() + lhs_row * row_elements;
    const auto* rhs_data = rhs.data<const float>() + rhs_row * row_elements;
    return row_elements == 0 || std::memcmp(lhs_data, rhs_data, row_elements * sizeof(float)) == 0;
}

}  // namespace detail

// Returns the reusable logical prefix shared by the retained target request and a new target request.
// DFlash keeps a single draft InferRequest state; it does not maintain a draft hash table.
inline size_t last_owner_lcp(const SequenceGroup::CPtr& owner,
                             const SequenceGroup::CPtr& current,
                             size_t limit,
                             const PerLayerEmbeddingsCallback& per_layer_embeddings_callback = {}) {
    if (!owner || !current || limit == 0 || owner->get_sequence_group_type() != current->get_sequence_group_type() ||
        owner->get_sequences().size() != 1 || current->get_sequences().size() != 1) {
        return 0;
    }

    const auto owner_sequence = owner->get_sequences().front();
    const auto current_sequence = current->get_sequences().front();
    const size_t owner_prompt_length = owner->get_prompt_len();
    const size_t current_prompt_length = current->get_prompt_len();
    const size_t owner_content_length = owner_prompt_length + owner_sequence->get_generated_len();
    const size_t max_length = std::min({limit, owner_content_length, current_prompt_length});
    if (max_length == 0) {
        return 0;
    }

    if (owner->get_sequence_group_type() == SequenceGroupType::TOKENS) {
        const auto& owner_prompt_ids = owner->get_prompt_ids();
        const auto& current_prompt_ids = current->get_prompt_ids();
        const auto& owner_generated_ids = owner_sequence->get_generated_ids();
        if (owner_prompt_ids.size() != owner_prompt_length || current_prompt_ids.size() != current_prompt_length) {
            return 0;
        }
        for (size_t index = 0; index < max_length; ++index) {
            const int64_t owner_id = index < owner_prompt_length ? owner_prompt_ids[index]
                                                                 : owner_generated_ids[index - owner_prompt_length];
            if (owner_id != current_prompt_ids[index]) {
                return index;
            }
        }
        return max_length;
    }

    if (owner->get_sequence_group_type() != SequenceGroupType::EMBEDDINGS) {
        return 0;
    }
    if (owner->get_deepstack_visual_embeds() || current->get_deepstack_visual_embeds() ||
        owner->get_visual_pos_masks() || current->get_visual_pos_masks()) {
        return 0;
    }

    const auto& owner_prompt_embeds = owner->get_input_embeds();
    const auto& current_prompt_embeds = current->get_input_embeds();
    const auto& owner_generated_embeds = owner_sequence->get_generated_ids_embeds();
    const auto& owner_positions = owner_sequence->get_position_ids_list();
    const auto& current_positions = current_sequence->get_position_ids_list();
    const auto owner_token_types = owner->get_token_type_ids();
    const auto current_token_types = current->get_token_type_ids();
    const auto& owner_per_layer_inputs = owner->get_per_layer_inputs();
    const auto& current_per_layer_inputs = current->get_per_layer_inputs();
    const bool has_owner_per_layer_inputs = detail::has_per_layer_inputs(owner_per_layer_inputs);
    const bool has_current_per_layer_inputs = detail::has_per_layer_inputs(current_per_layer_inputs);

    if (owner_prompt_embeds.size() != owner_prompt_length || current_prompt_embeds.size() != current_prompt_length ||
        owner_positions.size() < max_length || current_positions.size() < max_length ||
        has_owner_per_layer_inputs != has_current_per_layer_inputs ||
        (has_owner_per_layer_inputs &&
         !detail::per_layer_layouts_match(owner_per_layer_inputs, current_per_layer_inputs))) {
        return 0;
    }

    const auto row_matches = [&](size_t index) {
        const std::vector<float>* owner_embed = nullptr;
        if (index < owner_prompt_length) {
            owner_embed = &owner_prompt_embeds[index];
        } else {
            const size_t generated_index = index - owner_prompt_length;
            if (generated_index >= owner_generated_embeds.size()) {
                return false;
            }
            owner_embed = &owner_generated_embeds[generated_index];
        }
        if (!detail::embedding_rows_equal(*owner_embed, current_prompt_embeds[index]) ||
            !detail::tensors_equal(owner_positions[index], current_positions[index])) {
            return false;
        }

        const auto owner_type = detail::token_type_at(owner_token_types, index, owner_prompt_length);
        const auto current_type = detail::token_type_at(current_token_types, index, current_prompt_length);
        if (!owner_type || !current_type || *owner_type != *current_type || !has_owner_per_layer_inputs) {
            return owner_type && current_type && *owner_type == *current_type;
        }

        if (index < owner_prompt_length) {
            return detail::per_layer_rows_equal(owner_per_layer_inputs, index, current_per_layer_inputs, index);
        }
        if (!per_layer_embeddings_callback) {
            return false;
        }

        const size_t generated_index = index - owner_prompt_length;
        const auto& owner_generated_ids = owner_sequence->get_generated_ids();
        if (generated_index >= owner_generated_ids.size()) {
            return false;
        }
        ov::Tensor input_id(ov::element::i64, {1, 1});
        input_id.data<int64_t>()[0] = owner_generated_ids[generated_index];
        const ov::Tensor generated_per_layer_inputs = per_layer_embeddings_callback(input_id);
        return detail::per_layer_rows_equal(generated_per_layer_inputs, 0, current_per_layer_inputs, index);
    };

    size_t match_length = 0;
    while (match_length < max_length && row_matches(match_length)) {
        ++match_length;
    }
    if (match_length < max_length) {
        return std::min(detail::visual_span_start(owner_token_types, match_length, owner_prompt_length),
                        detail::visual_span_start(current_token_types, match_length, current_prompt_length));
    }

    // A prefix may end in the middle of an image span. Check the entire span before accepting it.
    const size_t last_index = match_length - 1;
    if (!detail::is_visual_row(owner_token_types, last_index, owner_prompt_length) &&
        !detail::is_visual_row(current_token_types, last_index, current_prompt_length)) {
        return match_length;
    }

    const size_t owner_span_start = detail::visual_span_start(owner_token_types, last_index, owner_prompt_length);
    const size_t current_span_start = detail::visual_span_start(current_token_types, last_index, current_prompt_length);
    const size_t span_start = std::min(owner_span_start, current_span_start);
    const size_t owner_span_end = detail::visual_span_end(owner_token_types, last_index, owner_prompt_length);
    const size_t current_span_end = detail::visual_span_end(current_token_types, last_index, current_prompt_length);
    if (owner_span_start != current_span_start || owner_span_end != current_span_end ||
        owner_span_end > owner_content_length || current_span_end > current_prompt_length ||
        owner_positions.size() < owner_span_end || current_positions.size() < current_span_end) {
        return span_start;
    }
    for (size_t index = match_length; index < owner_span_end; ++index) {
        if (!row_matches(index)) {
            return span_start;
        }
    }
    return match_length;
}

inline void ensure_num_assistant_tokens_is_set(GenerationConfig& config) {
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "DFlash CB/PA only supports num_assistant_tokens; assistant_confidence_threshold must be 0.f.");
    OPENVINO_ASSERT(config.max_ngram_size == 0,
                    "DFlash CB/PA does not support prompt lookup decoding; max_ngram_size must be 0.");
    if (!config.num_assistant_tokens.has_value() || config.num_assistant_tokens.value() == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
}

inline void ensure_vlm_generation_config(const GenerationConfig& config) {
    // Keep generic DFlash decoding behavior unchanged. These checks cover only
    // VLM representation/output modes that cannot be expressed faithfully with
    // row-aligned multimodal prompt IDs.
    OPENVINO_ASSERT(config.pruning_ratio == 0,
                    "DFlash VLM does not support visual token pruning.");
    OPENVINO_ASSERT(!config.echo && config.logprobs == 0,
                    "DFlash VLM does not support echo or prompt logprobs for row-aligned multimodal prompt IDs.");
    OPENVINO_ASSERT(!config.return_omni_outputs,
                    "DFlash VLM does not support Omni intermediate outputs.");
}

inline std::vector<int64_t> build_placeholder_prompt_ids(size_t prompt_length, int64_t placeholder_id) {
    OPENVINO_ASSERT(prompt_length > 0, "DFlash VLM logical prompt length must be greater than 0.");
    OPENVINO_ASSERT(placeholder_id >= 0, "DFlash VLM requires a valid placeholder token ID.");
    return std::vector<int64_t>(prompt_length, placeholder_id);
}

inline size_t linear_attention_checkpoint_block_count(size_t num_assistant_tokens) {
    OPENVINO_ASSERT(num_assistant_tokens <= std::numeric_limits<size_t>::max() - 2,
                    "DFlash num_assistant_tokens is too large for linear attention checkpoint block count.");
    return num_assistant_tokens + 2;
}

inline size_t adjusted_linear_attention_block_count(size_t current_block_count,
                                                    size_t num_assistant_tokens,
                                                    bool target_has_linear_attention) {
    if (!target_has_linear_attention) {
        return current_block_count;
    }
    return std::max(current_block_count, linear_attention_checkpoint_block_count(num_assistant_tokens));
}

inline ov::Tensor build_draft_input_ids(int64_t seed_token, int64_t mask_token_id, size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t draft_input_length = candidate_count + 1;
    ov::Tensor input_ids(ov::element::i64, {1, draft_input_length});
    auto* data = input_ids.data<int64_t>();
    data[0] = seed_token;
    std::fill(data + 1, data + draft_input_length, mask_token_id);
    return input_ids;
}

inline ov::Tensor build_draft_position_ids(size_t committed_context_length,
                                           size_t hidden_delta_length,
                                           size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    ov::Tensor position_ids(ov::element::i64, {1, hidden_delta_length + candidate_count + 1});
    auto* data = position_ids.data<int64_t>();
    std::iota(data, data + position_ids.get_size(), static_cast<int64_t>(committed_context_length));
    return position_ids;
}

inline ov::Tensor build_draft_attention_mask(size_t committed_context_length,
                                             size_t hidden_delta_length,
                                             size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t attention_mask_length = committed_context_length + hidden_delta_length + candidate_count + 1;
    ov::Tensor attention_mask(ov::element::i64, {1, attention_mask_length});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
    return attention_mask;
}

inline size_t draft_candidate_count(size_t num_assistant_tokens, size_t generated_len, size_t max_new_tokens) {
    OPENVINO_ASSERT(num_assistant_tokens > 0, "DFlash num_assistant_tokens must be greater than 0.");
    if (generated_len >= max_new_tokens) {
        return 0;
    }
    return num_assistant_tokens;
}

inline size_t validation_candidate_count(size_t draft_count, size_t generated_len, size_t max_new_tokens) {
    if (generated_len >= max_new_tokens) {
        return 0;
    }
    const size_t remaining = max_new_tokens - generated_len;
    if (remaining <= 1) {
        return 0;
    }
    return std::min(draft_count, remaining - 1);
}

struct ValidationAccounting {
    size_t accepted = 0;
    size_t rejected = 0;
    bool target_extended = false;
};

inline ValidationAccounting validation_accounting(size_t draft_generated,
                                                  size_t generated_before_draft,
                                                  size_t target_generated_len) {
    if (draft_generated == 0 || target_generated_len <= generated_before_draft) {
        return {};
    }

    const size_t produced_by_target = target_generated_len - generated_before_draft;
    const size_t accepted = produced_by_target > 0 ? std::min(draft_generated, produced_by_target - 1) : 0;
    return {accepted, draft_generated - accepted, true};
}

}  // namespace ov::genai::dflash_cb
