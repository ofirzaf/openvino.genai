// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/pass/sdpa_to_paged_attention.hpp>

#include "continuous_batching/paged_attention_transformations.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace {

struct DraftCandidateToken {
    int64_t token_id;
    float log_prob;
};

std::vector<float> zero_log_probs(size_t count) {
    return std::vector<float>(count, 0.0f);
}

bool dflash_draft_profiling_enabled() {
    const auto* value = std::getenv("DFLASH_PROFILE_DRAFT");
    return value && std::string(value) != "0";
}

void print_dflash_top_profile(const std::vector<ov::ProfilingInfo>& profiling_info) {
    std::vector<const ov::ProfilingInfo*> rows;
    rows.reserve(profiling_info.size());
    for (const auto& info : profiling_info) {
        if (info.real_time.count() > 0 || info.cpu_time.count() > 0)
            rows.push_back(&info);
    }
    std::sort(rows.begin(), rows.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->real_time > rhs->real_time;
    });

    std::ostringstream stream;
    stream << "DFlash draft GPU profile (top operations):";
    for (size_t i = 0; i < std::min<size_t>(rows.size(), 10); ++i) {
        const auto& info = *rows[i];
        stream << "\n  " << std::fixed << std::setprecision(3)
               << static_cast<double>(info.real_time.count()) / 1000.0 << " ms  "
               << info.node_type << "  " << info.node_name;
    }
    std::cout << stream.str() << std::endl;
}

void print_dflash_paged_attention_profile(const std::vector<ov::ProfilingInfo>& profiling_info) {
    std::ostringstream stream;
    stream << "DFlash draft PagedAttention operations:";
    size_t count = 0;
    for (const auto& info : profiling_info) {
        const bool is_paged_attention = info.node_type.find("PagedAttention") != std::string::npos ||
                                        info.node_name.find("paged_attention") != std::string::npos ||
                                        info.node_name.find("PagedAttention") != std::string::npos;
        if (!is_paged_attention)
            continue;
        stream << "\n  " << std::fixed << std::setprecision(3)
               << static_cast<double>(info.real_time.count()) / 1000.0 << " ms  "
               << info.node_type << "  " << info.node_name;
        ++count;
    }
    if (!count)
        stream << "\n  <not reported by the GPU profiling interface>";
    std::cout << stream.str() << std::endl;
}


bool model_has_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto outputs = model->outputs();
    return std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != outputs.end();
}

void validate_target_has_no_unmanaged_state(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    std::vector<std::string> unmanaged_state_ops;
    for (const auto& op : model->get_ordered_ops()) {
        const std::string type_name = op->get_type_name();
        if (type_name == "ReadValue" || type_name == "Assign") {
            unmanaged_state_ops.push_back(op->get_friendly_name());
        }
    }
    OPENVINO_ASSERT(unmanaged_state_ops.empty(),
                    "DFlash CB/PA target model contains unmanaged state op(s) after paging conversion. ",
                    "First unmanaged op: ", unmanaged_state_ops.empty() ? std::string{} : unmanaged_state_ops.front(),
                    ". Convert KV, conv, and GatedDeltaNet state to managed paging before enabling DFlash.");
}

}  // namespace

class ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashCBDraftRunner
    : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    struct Work {
        uint64_t request_id;
        ov::Tensor hidden_delta;
        int64_t seed;
        size_t candidate_count;
        size_t validation_count;
        bool final_chunk;
    };
    struct Result {
        size_t consumed;
        std::vector<DraftCandidateToken> candidates;
    };

    DFlashCBDraftRunner(const ModelDesc& desc, const Tokenizer& tokenizer,
                       const utils::dflash::DFlashRTInfo& info, EmbeddingsModel::Ptr embedding)
        : m_embedding_model(std::move(embedding)),
          m_mask_token_id(info.mask_token_id),
          m_profile_draft(dflash_draft_profiling_enabled()) {
        OPENVINO_ASSERT(desc.device.find("GPU") == 0,
                        "DFlash draft PA currently requires a GPU with micro-SDPA support");
        OPENVINO_ASSERT(!desc.scheduler_config.enable_prefix_caching && !desc.scheduler_config.use_cache_eviction &&
                            !desc.scheduler_config.use_sparse_attention,
                        "DFlash draft PA does not yet support prefix caching, eviction, or sparse attention");
        m_tokenizer = tokenizer;
        m_generation_config = desc.generation_config;
        auto properties = desc.properties;
        properties.emplace(ov::hint::kv_cache_precision.name(), ov::element::f16);
        if (m_profile_draft)
            properties.insert_or_assign(ov::enable_profiling.name(), true);
        initialize_pipeline(desc.model, desc.scheduler_config, desc.device, properties);
        OPENVINO_ASSERT(utils::get_compiled_kv_cache_precision(m_model_runner->get_infer_request().get_compiled_model()) ==
                            ov::element::f16, "DFlash draft requires FP16 KV caches");
        reset_metrics();
    }

    void initialize_sequence(uint64_t id, TokenIds prompt, const GenerationConfig& config) {
        OPENVINO_ASSERT(!m_sampler_groups.count(id), "Duplicate DFlash request ID");
        if (m_requests.empty())
            reset_metrics();
        m_sampler->clear_request_info(id);
        m_sampler_groups[id] = std::make_shared<SequenceGroup>(id, prompt, config);
        m_sampler_groups[id]->update_processed_tokens_num(prompt.size());
        // This group's counters describe cache rows only. Sampling has its own
        // group so rejecting every proposal cannot advance the cache ledger.
        m_requests.push_back(std::make_shared<SequenceGroup>(id, std::move(prompt), config));
    }

    void release(uint64_t id) {
        for (auto it = m_requests.begin(); it != m_requests.end(); ++it) {
            if ((*it)->get_request_id() != id)
                continue;
            for (const auto& seq : (*it)->get_sequences())
                m_scheduler->free_sequence(seq->get_id());
            m_requests.erase(it);
            break;
        }
        m_sampler_groups.erase(id);
        m_sampler->clear_request_info(id);
    }

    void sync_generated_tokens(uint64_t id, const std::vector<int64_t>& tokens) {
        auto& group = m_sampler_groups.at(id);
        auto seq = (*group)[0];
        if (seq->get_generated_len())
            seq->remove_last_tokens(seq->get_generated_len());
        for (auto token : tokens)
            seq->append_token(token, 0.f);
        group->update_processed_tokens_num(group->get_prompt_len() + tokens.size());
        seq->set_status(SequenceStatus::RUNNING);
        m_sampler->clear_request_info(id);
    }

    size_t committed(uint64_t id) const {
        for (const auto& group : m_requests)
            if (group->get_request_id() == id)
                return group->get_num_processed_tokens();
        OPENVINO_THROW("Missing DFlash cache request ", id);
    }

    size_t budget() const { return m_scheduler->get_config().max_num_batched_tokens; }

    std::map<uint64_t, Result> infer_batch(const std::vector<Work>& work) {
        std::map<uint64_t, const Work*> by_id;
        std::map<uint64_t, size_t> rows;
        for (const auto& item : work) {
            by_id.emplace(item.request_id, &item);
            rows.emplace(item.request_id, item.hidden_delta.get_shape()[0] + item.candidate_count + 1);
        }
        auto schedule = m_scheduler->schedule_append(m_requests, rows);
        if (utils::env_setup_for_print_debug_info()) {
            std::cout << "DFlash draft PA: requests=" << schedule.m_scheduled_sequence_groups_ids.size()
                      << " cache_rows=" << schedule.m_total_num_scheduled_tokens
                      << " cache_bytes=" << schedule.m_cache_size_in_bytes
                      << " cache_usage=" << schedule.m_cache_usage << std::endl;
        }
        if (schedule.m_scheduled_sequence_groups_ids.empty())
            return {};
        size_t total_hidden = 0, total_query = 0;
        for (auto i : schedule.m_scheduled_sequence_groups_ids) {
            const auto& item = *by_id.at(m_requests[i]->get_request_id());
            total_hidden += item.hidden_delta.get_shape()[0];
            total_query += item.candidate_count + 1;
        }
        const auto& first = *by_id.at(m_requests[schedule.m_scheduled_sequence_groups_ids.front()]->get_request_id());
        const size_t width = first.hidden_delta.get_shape()[2];
        ov::Tensor hidden(first.hidden_delta.get_element_type(), {total_hidden, width});
        std::vector<int64_t> ids, positions, kv_gather, query_positions, candidates;
        std::vector<int32_t> query_begins{0};
        std::vector<int32_t> token_types;
        size_t h = 0, q = 0, kv = 0;
        for (auto i : schedule.m_scheduled_sequence_groups_ids) {
            const auto& group = m_requests[i];
            const auto& item = *by_id.at(group->get_request_id());
            const size_t d = item.hidden_delta.get_shape()[0], b = item.candidate_count + 1;
            ov::Tensor destination(hidden, {h, 0}, {h + d, width});
            ov::Tensor normalized(item.hidden_delta.get_element_type(), {d, width}, item.hidden_delta.data());
            normalized.copy_to(destination);
            ids.push_back(item.seed);
            ids.insert(ids.end(), b - 1, m_mask_token_id);
            token_types.insert(token_types.end(), d, 0);
            token_types.insert(token_types.end(), b, 1);
            for (size_t j = 0; j < d + b; ++j)
                positions.push_back(group->get_num_processed_tokens() + j);
            for (size_t j = 0; j < d; ++j)
                kv_gather.push_back(h + j);
            for (size_t j = 0; j < b; ++j) {
                kv_gather.push_back(total_hidden + q + j);
                query_positions.push_back(kv + d + j);
                if (j)
                    candidates.push_back(q + j);
            }
            h += d;
            q += b;
            kv += d + b;
            query_begins.push_back(static_cast<int32_t>(q));
        }
        auto i64 = [](const std::vector<int64_t>& values) {
            ov::Tensor tensor(ov::element::i64, {values.size()});
            std::copy(values.begin(), values.end(), tensor.data<int64_t>());
            return tensor;
        };
        ov::Tensor query_bounds(ov::element::i32, {query_begins.size()});
        std::copy(query_begins.begin(), query_begins.end(), query_bounds.data<int32_t>());
        ov::Tensor types(ov::element::i32, {token_types.size()});
        std::copy(token_types.begin(), token_types.end(), types.data<int32_t>());
        std::map<std::string, ov::Tensor> inputs{{"hidden_states", hidden}, {"position_ids", i64(positions)},
            {"kv_gather_indices", i64(kv_gather)}, {"query_position_indices", i64(query_positions)},
            {"candidate_indices", i64(candidates)}, {"query_subsequence_begins", query_bounds},
            {"token_type_ids", types}};
        auto input_ids = i64(ids);
        const auto start = std::chrono::steady_clock::now();
        ov::Tensor logits;
        double embedding_ms = 0.0;
        double forward_ms = 0.0;
        if (m_embedding_model) {
            input_ids.set_shape({1, ids.size()});
            CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding_model->get_request_queue().get());
            const auto embedding_start = std::chrono::steady_clock::now();
            auto embedding = m_embedding_model->infer(guard.get(), input_ids);
            embedding_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embedding_start).count();
            ov::Tensor packed(embedding.get_element_type(), {total_query, embedding.get_shape().back()}, embedding.data());
            inputs.emplace("inputs_embeds", packed);
            const auto forward_start = std::chrono::steady_clock::now();
            logits = m_model_runner->forward_asymmetric(m_requests, schedule, inputs);
            forward_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - forward_start).count();
        } else {
            inputs.emplace("input_ids", input_ids);
            const auto forward_start = std::chrono::steady_clock::now();
            logits = m_model_runner->forward_asymmetric(m_requests, schedule, inputs);
            forward_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - forward_start).count();
        }
        const auto duration = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
        if (m_profile_draft && schedule.m_total_num_scheduled_tokens <= 16 && m_profile_dump_count++ < 6) {
            std::cout << "DFlash draft profile: cache_rows=" << schedule.m_total_num_scheduled_tokens
                      << " query_rows=" << total_query << " hidden_delta_rows=" << total_hidden
                      << " embedding_ms=" << std::fixed << std::setprecision(3) << embedding_ms
                      << " forward_ms=" << forward_ms
                      << " total_ms=" << static_cast<double>(duration) / 1000.0 << std::endl;
            const auto profiling_info = m_model_runner->get_infer_request().get_profiling_info();
            print_dflash_top_profile(profiling_info);
            print_dflash_paged_attention_profile(profiling_info);
        }
        m_raw_metrics.m_durations.emplace_back(duration);
        m_raw_metrics.m_inference_durations[0] += MicroSeconds(duration);
        m_raw_metrics.m_batch_sizes.push_back(schedule.m_total_num_scheduled_tokens);
        m_pipeline_metrics.scheduled_requests = schedule.m_scheduled_sequence_groups_ids.size();
        m_pipeline_metrics.cache_usage = schedule.m_cache_usage;
        m_pipeline_metrics.cache_size_in_bytes = schedule.m_cache_size_in_bytes;

        std::map<uint64_t, Result> results;
        size_t output_offset = 0;
        for (auto i : schedule.m_scheduled_sequence_groups_ids) {
            const auto& group = m_requests[i];
            const auto id = group->get_request_id();
            const auto& item = *by_id.at(id);
            const size_t d = item.hidden_delta.get_shape()[0];
            // All proposal rows are temporary, regardless of eventual acceptance.
            group->clear_scheduled_tokens();
            group->update_processed_tokens_num(group->get_num_processed_tokens() + d);
            auto& result = results[id];
            result.consumed = d;
            if (item.final_chunk) {
                auto& sampler_group = m_sampler_groups.at(id);
                for (size_t j = 0; j < item.validation_count; ++j) {
                    const auto before = (*sampler_group)[0]->get_generated_len();
                    ov::Tensor row(logits, {0, output_offset + j, 0},
                                   {1, output_offset + j + 1, logits.get_shape()[2]});
                    sampler_group->schedule_tokens(1);
                    sampler_group->set_output_seq_len(1);
                    sampler_group->set_num_validated_tokens(0);
                    m_sampler->sample({sampler_group}, row, false);
                    sampler_group->finish_iteration();
                    const auto seq = (*sampler_group)[0];
                    OPENVINO_ASSERT(seq->get_generated_len() == before + 1, "DFlash sampler failed to produce a candidate");
                    result.candidates.push_back({seq->get_generated_ids().back(), seq->get_generated_log_probs().back()});
                }
            }
            output_offset += item.candidate_count;
        }
        // Release whole tail pages; keep the committed portion of a partial page
        // in place. The next append overwrites its old proposal slots.
        m_scheduler->clean_empty_blocks(m_requests);
        return results;
    }

    RawPerfMetrics& get_raw_perf_metrics() { return m_raw_metrics; }

private:
    void reset_metrics() {
        m_raw_metrics = {};
        m_raw_metrics.m_inference_durations = {MicroSeconds(0.f)};
    }
    EmbeddingsModel::Ptr m_embedding_model;
    int64_t m_mask_token_id;
    std::map<uint64_t, SequenceGroup::Ptr> m_sampler_groups;
    RawPerfMetrics m_raw_metrics;
    bool m_profile_draft = false;
    size_t m_profile_dump_count = 0;
};

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info)
    : m_rt_info(rt_info) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.enable_prefix_caching,
                    "DFlash CB/PA does not support scheduler_config.enable_prefix_caching.");
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.use_cache_eviction &&
                        !main_model_desc.scheduler_config.use_sparse_attention,
                    "DFlash CB/PA does not yet support cache eviction or sparse attention.");

    auto main_model = main_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model_desc.model, "DFlash requires both target and draft models.");
    const bool is_vlm_dflash = static_cast<bool>(main_model_desc.inputs_embedder);
    if (is_vlm_dflash) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_vision_registry = std::make_shared<VisionRegistry>();
    }
    const auto draft_embedding_model =
        m_inputs_embedder ? m_inputs_embedder->get_embedding_model() : nullptr;

    // Detect each draft capability independently from its I/O signature (no RT-info markers) and
    // apply only the missing transform(s); a well-formed draft has exactly one input/output per pair.
    const bool needs_embedding_attach = utils::has_input(draft_model_desc.model, "inputs_embeds");
    const bool needs_lm_head_graft = model_has_output(draft_model_desc.model, "last_hidden_state");
    OPENVINO_ASSERT(needs_embedding_attach != utils::has_input(draft_model_desc.model, "input_ids"),
                    "DFlash draft model must have exactly one of 'inputs_embeds' or 'input_ids' input.");
    OPENVINO_ASSERT(needs_lm_head_graft != model_has_output(draft_model_desc.model, "logits"),
                    "DFlash draft model must have exactly one of 'last_hidden_state' or 'logits' output.");

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, m_rt_info.target_layer_ids);

    // Convert while the exported embedding parameter is still available to
    // the DFlash packer. Token IDs can be grafted onto the packed input later.
    OPENVINO_ASSERT(needs_embedding_attach,
                    "DFlash PA requires the current embeds-in append-all export; re-export the draft.");
    utils::dflash::convert_draft_to_paged_attention(draft_model_desc.model);
    validate_target_has_no_unmanaged_state(draft_model_desc.model);
    if (!is_vlm_dflash)
        utils::dflash::attach_target_embedding_to_draft(main_model, draft_model_desc.model);
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model, draft_model_desc.model);
    }

    const bool target_has_linear_attention = utils::get_cache_types(*main_model).has_linear();

    const bool allow_score_aggregation = true;
    const bool allow_cache_rotation = false;
    const bool allow_xattention = false;
    const bool allow_adaptive_rkv = false;
    const bool allow_qq_bias = main_model_desc.properties.count("query_to_query_bias") > 0 &&
                               main_model_desc.properties.at("query_to_query_bias").as<bool>();
    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_cache_rotation,
                                   allow_xattention,
                                   allow_adaptive_rkv,
                                   allow_qq_bias)
        .run_on_model(main_model);
    utils::apply_gather_before_matmul_transformation(main_model);
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(
        main_model,
        retained_hidden_state_locators,
        m_rt_info.target_layer_ids);

    m_tokenizer = main_model_desc.tokenizer;
    auto main_generation_config = main_model_desc.generation_config;
    dflash_cb::ensure_num_assistant_tokens_is_set(main_generation_config);
    OPENVINO_ASSERT(main_model_desc.scheduler_config.max_num_batched_tokens >=
                        main_generation_config.num_assistant_tokens.value() + 1,
                    "DFlash CB/PA requires max_num_batched_tokens >= num_assistant_tokens + 1.");
    m_generation_config = main_generation_config;
    auto target_scheduler_config = main_model_desc.scheduler_config;
    if (target_scheduler_config.num_linear_attention_blocks != 0) {
        target_scheduler_config.num_linear_attention_blocks =
            dflash_cb::adjusted_linear_attention_block_count(target_scheduler_config.num_linear_attention_blocks,
                                                             main_generation_config.num_assistant_tokens.value(),
                                                             target_has_linear_attention);
    }
    auto draft_model_desc_for_runner = draft_model_desc;
    if (draft_model_desc_for_runner.device.empty()) {
        draft_model_desc_for_runner.device = main_model_desc.device;
    }

    m_draft = std::make_shared<DFlashCBDraftRunner>(draft_model_desc_for_runner,
                                                    m_tokenizer,
                                                    m_rt_info,
                                                    draft_embedding_model);

    if (is_vlm_dflash) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            main_model,
            m_inputs_embedder,
            main_model_desc.tokenizer,
            main_generation_config,
            target_scheduler_config,
            main_model_desc.device,
            main_model_desc.properties,
            true);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            main_model,
            main_model_desc.tokenizer,
            main_generation_config,
            target_scheduler_config,
            main_model_desc.device,
            main_model_desc.properties,
            true);
    }
    m_main_pipeline->set_hidden_state_export_needed(true);
    m_main_pipeline->disable_preemption();
    m_main_pipeline->defer_oom_while_draft_synchronizes();
    OPENVINO_ASSERT(m_main_pipeline->get_kv_cache_element_type() == ov::element::f16,
                    "DFlash CB/PA currently requires FP16 target KV caches");

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) const {
    auto draft_config = config;
    draft_config.ignore_eos = true;
    draft_config.stop_strings = {};
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        // The target sampler retains these options and sees verified token
        // history. The placeholder-backed draft sampler must not apply state
        // that it cannot reconstruct after a speculative rejection.
        draft_config.repetition_penalty = 1.0f;
        draft_config.presence_penalty = 0.0f;
        draft_config.frequency_penalty = 0.0f;
        draft_config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();
        draft_config.min_new_tokens = 0;
        draft_config.stop_token_ids = {};
        draft_config.include_stop_str_in_output = false;
        draft_config.structured_output_config.reset();
        draft_config.parsers.clear();
    }
    draft_config.max_new_tokens = config.max_new_tokens + config.num_assistant_tokens.value();
    draft_config.num_assistant_tokens = 0;
    return draft_config;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::append_pending_hidden_delta(RequestState& state,
                                                                                const ov::Tensor& hidden_delta,
                                                                                bool copy_data) {
    state.pending_hidden_deltas.append(hidden_delta, copy_data);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_pending_hidden_delta(const RequestState& state) {
    return !state.pending_hidden_deltas.empty();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::clear_pending_hidden_delta(RequestState& state) {
    state.pending_hidden_deltas.clear();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_hidden_prefix_length(const RequestState& state) const {
    OPENVINO_ASSERT(!state.generated_tokens.empty(),
                    "DFlash hidden prefix can only be validated after target generated a seed token.");
    const size_t expected = state.prompt_len + state.generated_tokens.size() - 1;
    const size_t actual = m_draft->committed(state.target_request->get_request_id()) + state.pending_hidden_deltas.token_count();
    OPENVINO_ASSERT(actual == expected, "DFlash hidden prefix length mismatch before draft inference.");
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_finished_request_states() {
    for (auto state_it = m_request_states.begin(); state_it != m_request_states.end();) {
        if (state_it->second.finished) {
            m_draft->release(state_it->first);
            state_it = m_request_states.erase(state_it);
        } else {
            ++state_it;
        }
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    const bool is_vlm_dflash = m_model_input_type == ModelInputType::EMBEDDINGS;
    OPENVINO_ASSERT(sampling_params.is_greedy_decoding(), "DFlash CB/PA currently supports greedy decoding only.");
    OPENVINO_ASSERT(sampling_params.num_beams == 1, "DFlash CB/PA does not support beam search.");
    OPENVINO_ASSERT(sampling_params.num_return_sequences == 1, "DFlash CB/PA supports one sequence per request.");
    OPENVINO_ASSERT(!sampling_params.adapters.has_value(),
                    "DFlash CB/PA does not support adapters until target and draft adapter parity is validated.");
    if (is_vlm_dflash) {
        dflash_cb::ensure_vlm_generation_config(sampling_params);
    }
    drop_finished_request_states();
    OPENVINO_ASSERT(!m_request_states.count(request_id), "Duplicate DFlash request ID");

    const auto input_shape = input_ids.get_shape();
    if (is_vlm_dflash) {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::f32 &&
                            input_shape.size() == 3 && input_shape[0] == 1 && input_shape[1] > 0,
                        "DFlash VLM expects main input embeddings with shape [1, rows, hidden].");
    } else {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64 &&
                            input_shape.size() == 2 && input_shape[0] == 1 && input_shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
    }
    auto sampling_params_copy = sampling_params;
    dflash_cb::ensure_num_assistant_tokens_is_set(sampling_params_copy);
    RequestState state;
    state.generation_config = sampling_params_copy;
    state.prompt_len = input_shape[1];
    OPENVINO_ASSERT(sampling_params_copy.num_assistant_tokens.value() + 2 <= m_draft->budget(),
                    "DFlash draft scheduling budget must fit a delta row, seed, and the proposal block");
    OPENVINO_ASSERT(sampling_params_copy.num_assistant_tokens.value() + 1 <= m_main_pipeline->max_batched_tokens(),
                    "DFlash target scheduling budget must fit the complete verification block");
    TokenIds draft_prompt;
    if (is_vlm_dflash) {
        draft_prompt = dflash_cb::build_placeholder_prompt_ids(state.prompt_len, m_tokenizer.get_pad_token_id());
    } else {
        const auto* data = input_ids.data<const int64_t>();
        draft_prompt.assign(data, data + input_shape[1]);
    }
    m_draft->initialize_sequence(request_id, std::move(draft_prompt),
                                 make_draft_generation_config(sampling_params_copy));

    m_request_states[request_id] = std::move(state);

    // The draft sampler and request state are initialized above. If target
    // request creation fails, erase the state so a later request is not
    // rejected as a stale active DFlash request.
    try {
        auto handle = m_main_pipeline->add_request(request_id,
                                            input_ids,
                                            sampling_params_copy,
                                            prompt_ids,
                                            lm_extra_inputs);
        m_request_states.at(request_id).target_request = m_main_pipeline->find_request(request_id);
        return handle;
    } catch (...) {
        m_draft->release(request_id);
        m_request_states.erase(request_id);
        throw;
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(
            request_id,
            prompt,
            {},
            sampling_params);
    }
    auto input_ids = m_tokenizer.encode(prompt).input_ids;
    return add_request(request_id, input_ids, sampling_params);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
    const auto step_start = std::chrono::steady_clock::now();
    m_main_pipeline->pull_awaiting_requests();
    try {
        std::vector<DFlashCBDraftRunner::Work> work;
        for (auto& [id, state] : m_request_states) {
            const auto& target = state.target_request;
            OPENVINO_ASSERT(target, "DFlash target request was not registered");
            if (target->has_finished() || target->get_sequences().front()->out_of_memory() || target->handle_cancelled() || target->handle_stopped()) {
                state.finished = true;
                state.phase = RequestState::Phase::COMPLETE;
                clear_pending_hidden_delta(state);
                m_draft->release(id);
                continue;
            }
            if (state.phase == RequestState::Phase::VERIFY || state.generated_tokens.empty())
                continue;
            const size_t validation = dflash_cb::validation_candidate_count(
                state.generation_config.num_assistant_tokens.value(), state.generated_tokens.size(),
                state.generation_config.max_new_tokens);
            if (!validation) {
                target->pause_generation(false);
                continue;
            }
            OPENVINO_ASSERT(has_pending_hidden_delta(state), "DFlash is missing committed target hidden states");
            state.phase = RequestState::Phase::SYNCHRONIZE;
            target->pause_generation(true);
            validate_hidden_prefix_length(state);
            const size_t count = std::min(state.pending_hidden_deltas.token_count(), m_draft->budget() - validation - 1);
            work.push_back({id, state.pending_hidden_deltas.prefix(count), state.generated_tokens.back(),
                            validation, validation, count == state.pending_hidden_deltas.token_count()});
        }
        const auto draft_start = std::chrono::steady_clock::now();
        const auto draft_infer_before = m_draft->get_raw_perf_metrics().m_inference_durations[0];
        auto results = work.empty() ? std::map<uint64_t, DFlashCBDraftRunner::Result>{} : m_draft->infer_batch(work);
        const auto draft_end = std::chrono::steady_clock::now();
        m_sd_metrics.draft_duration += PerfMetrics::get_microsec(draft_end - draft_start) / 1e6;
        for (auto& [id, result] : results) {
            auto& state = m_request_states.at(id);
            state.pending_hidden_deltas.consume(result.consumed);
            if (result.candidates.empty())
                continue;
            OPENVINO_ASSERT(state.pending_hidden_deltas.empty(), "DFlash produced candidates before synchronization completed");
            state.generated_before_draft = state.generated_tokens.size();
            state.draft_generated = result.candidates.size();
            auto ids = state.generated_tokens;
            auto probabilities = zero_log_probs(ids.size());
            for (const auto& candidate : result.candidates) {
                ids.push_back(candidate.token_id);
                probabilities.push_back(candidate.log_prob);
            }
            m_main_pipeline->update_request(id, {{0, GeneratedSequence(ids, probabilities)}}, false);
            state.phase = RequestState::Phase::VERIFY;
            state.target_request->pause_generation(false);
        }

        bool target_ready = false;
        for (auto& [id, state] : m_request_states) {
            const auto& target = state.target_request;
            // Clear borrowed outputs before another target call, including for
            // deferred requests. Only this call's scheduled rows may be consumed.
            for (const auto& seq : target->get_sequences())
                seq->update_hidden_state({});
            state.hidden_start = target->get_num_processed_tokens();
            target_ready |= !state.finished && !target->is_waiting();
        }
        const auto main_start = std::chrono::steady_clock::now();
        // Remove requests completed outside the target's own step before its
        // scheduler sees them. In particular an OOM sequence is already terminal.
        const bool cleanup = std::any_of(m_request_states.begin(), m_request_states.end(),
                                        [](const auto& item) { return item.second.finished; });
        if (cleanup)
            m_main_pipeline->cleanup_finished_requests();
        if (target_ready) {
            m_main_pipeline->sync_generated_embeddings();
            m_main_pipeline->step();
        }
        const auto main_end = std::chrono::steady_clock::now();
        const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
        m_sd_metrics.main_duration += main_duration / 1e6;
        const bool target_inferred = target_ready && m_main_pipeline->get_metrics().scheduled_requests > 0;
        if (!target_inferred && results.empty() && !work.empty()) {
            // Neither model made progress. No request can release capacity
            // without first admitting more cache rows, so fail one request
            // normally and release its pages instead of spinning.
            auto& state = m_request_states.at(work.front().request_id);
            state.target_request->set_out_of_memory();
            state.target_request->notify_handle();
            state.target_request->pause_generation(false);
            state.finished = true;
            m_draft->release(work.front().request_id);
            m_main_pipeline->cleanup_finished_requests();
        }
        update_draft_states_from_main();
        drop_finished_request_states();
        m_pipeline_metrics = m_main_pipeline->get_metrics();
        const auto tokens = target_inferred ? m_main_pipeline->get_processed_tokens_per_iteration() : 0;
        // Synchronization chunks and target prefill chunks can produce no
        // tokens. Their inference time still belongs to this generation.
        auto& raw = m_perf_metrics.raw_metrics;
        raw.m_inference_durations[0] += m_draft->get_raw_perf_metrics().m_inference_durations[0] - draft_infer_before;
        if (target_inferred) {
            const auto inference = MicroSeconds(m_pipeline_metrics.inference_duration);
            raw.m_inference_durations[0] += inference;
            auto& main_raw = m_perf_metrics.main_model_metrics.raw_metrics;
            main_raw.m_durations.emplace_back(main_duration);
            main_raw.m_inference_durations[0] += inference;
            main_raw.m_batch_sizes.push_back(tokens);
        } else {
            m_pipeline_metrics.scheduled_requests = 0;
            m_pipeline_metrics.inference_duration = 0;
        }
        if (tokens) {
            const auto duration = PerfMetrics::get_microsec(main_end - step_start);
            raw.m_token_infer_durations.emplace_back(duration);
            raw.m_new_token_times.emplace_back(main_end);
            raw.m_batch_sizes.push_back(tokens);
            m_sd_metrics.update_generated_len(tokens);
        }
    } catch (...) {
        m_main_pipeline->finish_request();
        for (const auto& [id, state] : m_request_states)
            m_draft->release(id);
        m_request_states.clear();
        throw;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::update_draft_states_from_main() {
    for (auto& [id, state] : m_request_states) {
        const auto& target = state.target_request;
        const auto seq = target->get_sequences().front();
        const auto hidden = seq->get_hidden_state();
        if (hidden && hidden.get_size()) {
            const auto& generated = seq->get_generated_ids();
            const auto accounting = dflash_cb::validation_accounting(
                state.draft_generated, state.generated_before_draft, generated.size());
            if (state.draft_generated) {
                OPENVINO_ASSERT(accounting.target_extended, "DFlash verification did not advance the target");
                if (utils::env_setup_for_print_debug_info()) {
                    std::cout << "DFlash verification: request=" << id << " proposed=" << state.draft_generated
                              << " accepted=" << accounting.accepted << std::endl;
                }
                m_perf_metrics.num_draft_tokens += state.draft_generated;
                m_perf_metrics.num_accepted_tokens += accounting.accepted;
                m_sd_metrics.update_draft_generated_len(id, state.draft_generated);
                m_sd_metrics.update_draft_accepted_tokens(id, accounting.accepted);
                m_sd_metrics.update_acceptance_rate(id, 100.f * accounting.accepted / state.draft_generated);
            }
            if (!state.finished && !target->has_finished() && !target->get_sequences().front()->out_of_memory() &&
                !target->handle_cancelled() && !target->handle_stopped()) {
                auto committed_hidden = dflash_cb::truncate_normalized_hidden_state_from_end(hidden, accounting.rejected);
                OPENVINO_ASSERT(state.hidden_start == m_draft->committed(id) + state.pending_hidden_deltas.token_count(),
                                "DFlash target hidden range is duplicated, missing, or out of order");
                // Target outputs are reused on its next inference. Own each
                // accepted range until the draft has consumed every row.
                append_pending_hidden_delta(state, committed_hidden, true);
                state.generated_tokens = generated;
                m_draft->sync_generated_tokens(id, generated);
                state.phase = generated.empty() ? RequestState::Phase::PREFILL : RequestState::Phase::SYNCHRONIZE;
            }
            state.draft_generated = 0;
        } else if (state.phase == RequestState::Phase::VERIFY && !state.finished &&
                   utils::env_setup_for_print_debug_info()) {
            std::cout << "DFlash verification deferred: request=" << id << std::endl;
        }
        state.finished = state.finished || target->has_finished() || target->get_sequences().front()->out_of_memory() ||
                         target->handle_cancelled() || target->handle_stopped();
        if (state.finished) {
            state.phase = RequestState::Phase::COMPLETE;
            clear_pending_hidden_delta(state);
        }
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_requests() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    if (m_main_pipeline) {
        m_main_pipeline->pull_awaiting_requests();
        m_main_pipeline->finish_request();
    }
    for (const auto& [id, state] : m_request_states)
        m_draft->release(id);
    m_request_states.clear();
}

ov::genai::RawPerfMetrics ContinuousBatchingPipeline::DFlashDecodingImpl::collect_draft_raw_metrics() {
    ov::genai::RawPerfMetrics raw_metrics;
    raw_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    if (m_draft) {
        auto& draft_metrics = m_draft->get_raw_perf_metrics();
        raw_metrics.m_durations.insert(raw_metrics.m_durations.end(),
                                       draft_metrics.m_durations.begin(),
                                       draft_metrics.m_durations.end());
        raw_metrics.m_batch_sizes.insert(raw_metrics.m_batch_sizes.end(),
                                         draft_metrics.m_batch_sizes.begin(),
                                         draft_metrics.m_batch_sizes.end());
        if (!draft_metrics.m_inference_durations.empty()) {
            raw_metrics.m_inference_durations[0] += draft_metrics.m_inference_durations[0];
        }
    }
    return raw_metrics;
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    if (position_ids.has_value()) {
        OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS,
                        "DFlash CB/PA only accepts explicit position_ids in VLM embedding mode.");
        OPENVINO_ASSERT(position_ids->size() == input_ids.size(),
                        "DFlash VLM position_ids count must match input embeddings.");
    }
    OPENVINO_ASSERT(!has_non_finished_requests(),
                    "Generate cannot be called while ContinuousBatchingPipeline is already running");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());
    OPENVINO_ASSERT(!input_ids.empty(), "DFlash requires at least one input");

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    auto start_time = std::chrono::steady_clock::now();

    auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);
    OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 && sampling_params[0].is_greedy_decoding()),
                    "DFlash CB/PA streaming only supports batch size=1 with greedy decoding.");

    std::vector<GenerationHandle> main_generations;
    try {
        for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
            OPENVINO_ASSERT(input_ids[request_id].get_shape().at(0) == 1, "Use multiple tensors to pass a batch.");
            if (position_ids.has_value()) {
                const auto& [main_position_ids, rope_delta] = (*position_ids)[request_id];
                m_inputs_embedder->set_position_ids(main_position_ids);
                m_inputs_embedder->set_rope_delta(rope_delta.value_or(compute_rope_delta(main_position_ids)));
            }
            const bool has_valid_prompt_ids = prompt_ids.has_value() && request_id < prompt_ids->size();
            const bool has_valid_lm_extra_inputs = lm_extra_inputs_list.has_value() && request_id < lm_extra_inputs_list->size();
            main_generations.push_back(add_request(
                request_id,
                input_ids[request_id],
                sampling_params[request_id],
                has_valid_prompt_ids ? std::make_optional((*prompt_ids)[request_id]) : std::nullopt,
                has_valid_lm_extra_inputs ? std::make_optional((*lm_extra_inputs_list)[request_id]) : std::nullopt));
        }
    } catch (...) {
        drop_requests();
        throw;
    }

    auto all_requests = m_main_pipeline->get_awaiting_requests();
    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();
    while (has_non_finished_requests()) {
        try {
            step();
        } catch (...) {
            drop_requests();
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }
    streamer_ptr->end();

    OPENVINO_ASSERT(m_main_pipeline->is_requests_empty(),
                    "Internal error: current request is supposed to be dropped within step() function as completed");

    m_perf_metrics.draft_model_metrics.raw_metrics = collect_draft_raw_metrics();
    uint64_t generate_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time);

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto cfg = request->get_sampling_parameters();
        const auto& seqs = request->get_finished_sequences();
        size_t num_out = std::min(cfg.num_return_sequences, seqs.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_out);
        result.m_scores.resize(num_out);
        result.m_finish_reasons.resize(num_out, GenerationFinishReason::NONE);
        result.m_status = main_generations[request_id]->get_status();

        for (size_t i = 0; i < num_out; ++i) {
            const auto& seq = seqs[i];
            float score = cfg.is_beam_search() ? seq->get_beam_search_score(cfg) : seq->get_cumulative_log_prob();
            const auto& gen_ids = seq->get_generated_ids();
            if (cfg.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(gen_ids.begin(), gen_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
            result.m_finish_reasons[i] = seq->get_finish_reason();
            if (result.m_finish_reasons[i] == GenerationFinishReason::NONE && request->handle_stopped()) {
                result.m_finish_reasons[i] = request->get_generation_stream()->get_finish_reason();
            }
        }

        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_duration_us);
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.num_prefix_cache_hit_tokens = request->get_num_prefix_cache_hit_tokens();
        m_perf_metrics.evaluate_statistics(start_time);

        result.perf_metrics = m_perf_metrics;
        result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_perf_metrics);
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

}  // namespace ov::genai
