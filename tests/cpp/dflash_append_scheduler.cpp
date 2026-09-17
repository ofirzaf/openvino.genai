// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "continuous_batching/scheduler.hpp"

namespace {
using namespace ov::genai;

// No device work is needed to test page ownership and scheduling decisions.
class AppendTestCache final : public ICacheManager {
    size_t blocks = 0;
public:
    void allocate_cache_if_needed(size_t count) override { blocks = std::max(blocks, count); }
    void copy_blocks(const std::map<size_t, std::list<size_t>>& copies) override { EXPECT_TRUE(copies.empty()); }
    void clear() override { blocks = 0; }
    size_t get_num_layers() const override { return 6; }
    size_t get_num_cache_tensors() const override { return 12; }
    size_t get_block_size() const override { return 4; }
    std::string get_device() const override { return "CPU"; }
    size_t get_block_size_in_bytes() const override { return 48; }
    size_t get_num_allocated_blocks() const override { return blocks; }
};

struct AppendSchedulerTest : testing::Test {
    SchedulerConfig config;
    std::shared_ptr<CacheOrchestrator> cache = std::make_shared<CacheOrchestrator>();
    std::unique_ptr<Scheduler> scheduler;
    std::vector<SequenceGroup::Ptr> groups;
    void SetUp() override {
        config.num_kv_blocks = 4;
        config.max_num_batched_tokens = 8;
        config.max_num_seqs = 2;
        cache->register_cache_type(CacheType::KV_CACHE, std::make_unique<AppendTestCache>(),
                                    std::make_unique<BlockManager>(4, false, 4, 1));
        scheduler = std::make_unique<Scheduler>(cache, config);
        GenerationConfig generation;
        generation.max_new_tokens = 32;
        for (uint64_t id : {19, 42})
            groups.push_back(std::make_shared<SequenceGroup>(id, TokenIds{1}, generation));
    }
    void commit(size_t index, size_t delta) {
        auto& group = groups[index];
        group->clear_scheduled_tokens();
        group->update_processed_tokens_num(group->get_num_processed_tokens() + delta);
    }
};

TEST_F(AppendSchedulerTest, RetainsPartialPageAndReusesWholeProposalPages) {
    auto first = scheduler->schedule_append(groups, {{19, 5}, {42, 3}});
    ASSERT_EQ(first.m_scheduled_sequence_groups_ids, (std::vector<uint64_t>{0, 1}));
    EXPECT_EQ(first.m_total_num_scheduled_tokens, 8);
    const auto id = groups[0]->get_sequences()[0]->get_id();
    const auto retained_page = scheduler->get_kv_block_tables(id)[0][0]->get_index();
    commit(0, 3);
    commit(1, 1);
    // Drop the schedule snapshot's shared pointers before releasing pages.
    first = {};
    scheduler->clean_empty_blocks(groups);
    EXPECT_EQ(scheduler->get_kv_block_tables(id)[0].size(), 1);
    EXPECT_EQ(cache->num_free_blocks(), 2);
    auto next = scheduler->schedule_append(groups, {{19, 3}, {42, 5}});
    EXPECT_EQ(next.m_scheduled_sequence_groups_ids.size(), 2);
    EXPECT_EQ(scheduler->get_kv_block_tables(id)[0][0]->get_index(), retained_page);
    EXPECT_EQ(groups[0]->get_num_processed_tokens(), 3);
    EXPECT_EQ(groups[1]->get_num_processed_tokens(), 1);
}

TEST_F(AppendSchedulerTest, DefersIndivisibleBlockWithoutPreemptingCommittedPages) {
    auto first = scheduler->schedule_append(groups, {{19, 8}, {42, 8}});
    ASSERT_EQ(first.m_scheduled_sequence_groups_ids, (std::vector<uint64_t>{0}));
    commit(0, 7);
    first = {};
    auto second = scheduler->schedule_append(groups, {{42, 8}});
    ASSERT_EQ(second.m_scheduled_sequence_groups_ids, (std::vector<uint64_t>{1}));
    commit(1, 7);
    second = {};
    auto deferred = scheduler->schedule_append(groups, {{19, 3}, {42, 3}});
    EXPECT_TRUE(deferred.m_scheduled_sequence_groups_ids.empty());
    EXPECT_EQ(groups[0]->get_num_processed_tokens(), 7);
    EXPECT_EQ(groups[1]->get_num_processed_tokens(), 7);
    EXPECT_EQ(groups[0]->get_num_scheduled_tokens(), 0);
    scheduler->free_sequence(groups[1]->get_sequences()[0]->get_id());
    groups.pop_back();
    auto resumed = scheduler->schedule_append(groups, {{19, 3}});
    EXPECT_EQ(resumed.m_scheduled_sequence_groups_ids, (std::vector<uint64_t>{0}));
}
}  // namespace
