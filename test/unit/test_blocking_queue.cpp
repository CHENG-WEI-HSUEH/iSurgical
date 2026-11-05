// Unit tests for src/common/blocking_queue.hpp: the bounded queue between the capture, inference, RSD and display threads.

// Code under test (header-only template)
#include "common/blocking_queue.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::atomic
#include <atomic>
// std::chrono
#include <chrono>
// std::unique_ptr
#include <memory>
// std::string
#include <string>
// std::thread
#include <thread>
// std::vector
#include <vector>

// ElapsedMs
#include "test_helpers.hpp"

// Anonymous namespace
namespace {

// Short names
using namespace std::chrono_literals;
// ElapsedMs
using test_helpers::ElapsedMs;

// Items come out in the order they went in
TEST(BlockingQueueTest, FifoOrder) {
    // Room for 4
    BlockingQueue<int> q(4);
    // Push 1, 2, 3
    for (int i = 1; i <= 3; ++i) ASSERT_TRUE(q.Push(i));
    // Output slot
    int v = 0;
    // Pop in order
    for (int i = 1; i <= 3; ++i) {
        // Must succeed
        ASSERT_TRUE(q.Pop(v));
        // Same order
        EXPECT_EQ(v, i);
    }
}

// Capacity 0 is treated as 1 (otherwise every Push would block forever)
TEST(BlockingQueueTest, ZeroCapacityBecomesOne) {
    // Capacity 0, drop-oldest so we can see the effective capacity
    BlockingQueue<int> q(0, true);
    // First item
    q.Push(1);
    // Second item replaces the first
    q.Push(2);
    // One dropped → capacity was 1
    EXPECT_EQ(q.Dropped(), 1u);
    // Output
    int v = 0;
    // Newest item
    ASSERT_TRUE(q.Pop(v));
    // Is 2
    EXPECT_EQ(v, 2);
}

// A new queue has dropped nothing
TEST(BlockingQueueTest, DroppedStartsAtZero) {
    // Any queue
    BlockingQueue<int> q(2, true);
    // Zero
    EXPECT_EQ(q.Dropped(), 0u);
}

// drop_oldest: a full queue discards the oldest items and counts them
TEST(BlockingQueueTest, DropOldestKeepsNewestItems) {
    // Room for 3
    BlockingQueue<int> q(3, true);
    // Push 1..5: 1 and 2 get dropped
    for (int i = 1; i <= 5; ++i) ASSERT_TRUE(q.Push(i));
    // Two dropped
    EXPECT_EQ(q.Dropped(), 2u);
    // Close so Pop returns false once drained
    q.Close();
    // Collected values
    std::vector<int> got;
    // Output
    int v = 0;
    // Drain
    while (q.Pop(v)) got.push_back(v);
    // The three newest
    EXPECT_EQ(got, (std::vector<int>{3, 4, 5}));
}

// Blocking mode never drops
TEST(BlockingQueueTest, BlockingModeNeverDrops) {
    // Room for 2, blocking
    BlockingQueue<int> q(2, false);
    // Fill exactly
    q.Push(1);
    // Second
    q.Push(2);
    // Nothing dropped
    EXPECT_EQ(q.Dropped(), 0u);
}

// After Close, Push fails in both modes
class BlockingQueueModeTest : public ::testing::TestWithParam<bool> {};

// Push on a closed queue returns false and adds nothing
TEST_P(BlockingQueueModeTest, PushAfterCloseFails) {
    // Mode from the parameter
    BlockingQueue<int> q(2, GetParam());
    // Close first
    q.Close();
    // Rejected
    EXPECT_FALSE(q.Push(1));
    // Output
    int v = -1;
    // Nothing to pop
    EXPECT_FALSE(q.Pop(v));
    // Output untouched
    EXPECT_EQ(v, -1);
}

// Items pushed before Close are still delivered
TEST_P(BlockingQueueModeTest, PopDrainsAfterClose) {
    // Mode from the parameter
    BlockingQueue<int> q(4, GetParam());
    // Two items
    q.Push(10);
    // Second
    q.Push(20);
    // Close
    q.Close();
    // Output
    int v = 0;
    // First still there
    ASSERT_TRUE(q.Pop(v));
    // Value
    EXPECT_EQ(v, 10);
    // Second still there
    ASSERT_TRUE(q.Pop(v));
    // Value
    EXPECT_EQ(v, 20);
    // Now empty and closed
    EXPECT_FALSE(q.Pop(v));
}

// Run both modes
INSTANTIATE_TEST_SUITE_P(
    // Prefix
    Modes,
    // Fixture
    BlockingQueueModeTest,
    // false = blocking, true = drop oldest
    ::testing::Bool(),
    // Readable names
    [](const ::testing::TestParamInfo<bool>& info) { return std::string(info.param ? "DropOldest" : "Blocking"); });

// Close wakes a consumer waiting on an empty queue
TEST(BlockingQueueTest, CloseWakesBlockedPop) {
    // Empty queue
    BlockingQueue<int> q(1);
    // Result of the blocked Pop
    std::atomic<int> result{-1};
    // Consumer thread
    std::thread consumer([&] {
        // Output
        int v = 0;
        // Blocks until Close
        result = q.Pop(v) ? 1 : 0;
    });
    // Give it time to block
    std::this_thread::sleep_for(50ms);
    // Still waiting
    EXPECT_EQ(result.load(), -1);
    // Wake it
    q.Close();
    // Thread finishes
    consumer.join();
    // Pop returned false
    EXPECT_EQ(result.load(), 0);
}

// Close wakes a producer waiting on a full queue (blocking mode)
TEST(BlockingQueueTest, CloseWakesBlockedPush) {
    // Room for 1
    BlockingQueue<int> q(1, false);
    // Fill it
    ASSERT_TRUE(q.Push(1));
    // Result of the blocked Push
    std::atomic<int> result{-1};
    // Producer thread
    std::thread producer([&] { result = q.Push(2) ? 1 : 0; });
    // Let it block
    std::this_thread::sleep_for(50ms);
    // Still blocked
    EXPECT_EQ(result.load(), -1);
    // Wake it
    q.Close();
    // Finish
    producer.join();
    // Push returned false
    EXPECT_EQ(result.load(), 0);
}

// A blocked Push continues as soon as a Pop makes room
TEST(BlockingQueueTest, BlockedPushResumesAfterPop) {
    // Room for 1
    BlockingQueue<int> q(1, false);
    // Fill
    q.Push(1);
    // Set when the second Push returns
    std::atomic<bool> pushed{false};
    // Producer
    std::thread producer([&] {
        // Blocks while full
        q.Push(2);
        // Done
        pushed = true;
    });
    // Let it block
    std::this_thread::sleep_for(50ms);
    // Still blocked
    EXPECT_FALSE(pushed.load());
    // Output
    int v = 0;
    // Make room
    ASSERT_TRUE(q.Pop(v));
    // First item
    EXPECT_EQ(v, 1);
    // Producer finishes
    producer.join();
    // Push completed
    EXPECT_TRUE(pushed.load());
    // Second item available
    ASSERT_TRUE(q.Pop(v));
    // Value
    EXPECT_EQ(v, 2);
}

// A blocked Pop continues as soon as an item is pushed
TEST(BlockingQueueTest, BlockedPopResumesAfterPush) {
    // Empty queue
    BlockingQueue<int> q(2);
    // Value seen by the consumer
    std::atomic<int> seen{0};
    // Consumer
    std::thread consumer([&] {
        // Output
        int v = 0;
        // Blocks until an item arrives
        if (q.Pop(v)) seen = v;
    });
    // Let it block
    std::this_thread::sleep_for(30ms);
    // Start time
    const auto start = std::chrono::steady_clock::now();
    // Deliver
    q.Push(99);
    // Consumer finishes
    consumer.join();
    // Got it
    EXPECT_EQ(seen.load(), 99);
    // Woke promptly (well under a second)
    EXPECT_LT(ElapsedMs(start), 1000);
}

// Many producers and consumers: nothing lost, nothing duplicated
TEST(BlockingQueueTest, ConcurrentProducersConsumersLoseNothing) {
    // Small capacity forces lots of blocking
    BlockingQueue<int> q(4, false);
    // Items per producer
    constexpr int kPerProducer = 2000;
    // Number of producers
    constexpr int kProducers = 3;
    // Number of consumers
    constexpr int kConsumers = 3;
    // Sum of everything consumed
    std::atomic<long long> sum{0};
    // Count of everything consumed
    std::atomic<int> count{0};
    // Consumer threads
    std::vector<std::thread> consumers;
    // Start consumers
    for (int c = 0; c < kConsumers; ++c) {
        // Each drains until Close + empty
        consumers.emplace_back([&] {
            // Output
            int v = 0;
            // Drain
            while (q.Pop(v)) {
                // Add up
                sum += v;
                // Count
                ++count;
            }
        });
    }
    // Producer threads
    std::vector<std::thread> producers;
    // Start producers
    for (int p = 0; p < kProducers; ++p) {
        // Each pushes 1..kPerProducer
        producers.emplace_back([&] {
            // Push all
            for (int i = 1; i <= kPerProducer; ++i) q.Push(i);
        });
    }
    // Wait for producers
    for (std::thread& t : producers) t.join();
    // Let consumers finish once the queue is empty
    q.Close();
    // Wait for consumers
    for (std::thread& t : consumers) t.join();
    // Every item seen once
    EXPECT_EQ(count.load(), kProducers * kPerProducer);
    // Sum of 1..N per producer
    EXPECT_EQ(sum.load(), static_cast<long long>(kProducers) * kPerProducer * (kPerProducer + 1) / 2);
    // Blocking mode never drops
    EXPECT_EQ(q.Dropped(), 0u);
}

// Move-only types work (frames are often held by unique_ptr)
TEST(BlockingQueueTest, MoveOnlyItems) {
    // Queue of owning pointers
    BlockingQueue<std::unique_ptr<int>> q(2);
    // Push a moved pointer
    ASSERT_TRUE(q.Push(std::make_unique<int>(7)));
    // Output
    std::unique_ptr<int> out;
    // Pop
    ASSERT_TRUE(q.Pop(out));
    // Non-null
    ASSERT_NE(out, nullptr);
    // Value preserved
    EXPECT_EQ(*out, 7);
}

// Non-trivial value types are moved intact
TEST(BlockingQueueTest, StringItemsSurvive) {
    // Queue of strings
    BlockingQueue<std::string> q(2, true);
    // A long string (heap-allocated)
    const std::string big(1000, 'x');
    // Push a copy
    q.Push(big);
    // Output
    std::string out;
    // Pop
    ASSERT_TRUE(q.Pop(out));
    // Same content
    EXPECT_EQ(out, big);
}

// Close may be called more than once
TEST(BlockingQueueTest, CloseIsIdempotent) {
    // Any queue
    BlockingQueue<int> q(1);
    // First close
    q.Close();
    // Second close: no crash, no deadlock
    q.Close();
    // Still closed
    EXPECT_FALSE(q.Push(1));
}

// Dropped() counts across many overflows
TEST(BlockingQueueTest, DroppedCountsEveryOverflow) {
    // Room for 1
    BlockingQueue<int> q(1, true);
    // 100 pushes
    for (int i = 0; i < 100; ++i) q.Push(i);
    // 99 dropped
    EXPECT_EQ(q.Dropped(), 99u);
}

}  // namespace
