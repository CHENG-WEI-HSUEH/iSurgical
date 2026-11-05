// Unit tests for src/common/isurgical_shm.hpp: the shared-memory layout, CopyName and the status seqlock.

// Code under test (header only)
#include "common/isurgical_shm.hpp"

// GoogleTest
#include <gtest/gtest.h>

// std::atomic
#include <atomic>
// offsetof
#include <cstddef>
// std::memcmp, std::strlen
#include <cstring>
// std::make_unique
#include <memory>
// std::string
#include <string>
// std::thread
#include <thread>

// Anonymous namespace
namespace {

// Short name
using namespace isurgical_shm;

// The magic number's bytes spell "ISUG" in memory (little-endian)
TEST(ShmLayoutTest, MagicSpellsIsug) {
    // The constant
    const uint32_t magic = kMagic;
    // Its bytes in memory
    EXPECT_EQ(std::memcmp(&magic, "ISUG", 4), 0);
}

// The documented limits
TEST(ShmLayoutTest, Limits) {
    // Version 1
    EXPECT_EQ(kVersion, 1u);
    // 16 phases
    EXPECT_EQ(kMaxPhases, 16);
    // 32 classes
    EXPECT_EQ(kMaxClasses, 32);
    // 64-byte names
    EXPECT_EQ(kNameSize, 64);
}

// The layout is identical in every process: fixed-size, standard-layout members
TEST(ShmLayoutTest, TypesAreTriviallyCopyableAndStandardLayout) {
    // memcpy-able
    EXPECT_TRUE(std::is_trivially_copyable_v<Tables>);
    // memcpy-able
    EXPECT_TRUE(std::is_trivially_copyable_v<Status>);
    // Same layout as a C struct
    EXPECT_TRUE(std::is_standard_layout_v<SharedState>);
    // Lock-free atomics (required across processes)
    EXPECT_TRUE(std::atomic<uint64_t>::is_always_lock_free);
}

// magic and version are the first two fields, so a reader can check them before anything else
TEST(ShmLayoutTest, HeaderFieldsComeFirst) {
    // magic at offset 0
    EXPECT_EQ(offsetof(SharedState, magic), 0u);
    // version right after
    EXPECT_EQ(offsetof(SharedState, version), 4u);
}

// A short name is copied whole and terminated
TEST(CopyNameTest, ShortNameIsCopied) {
    // Buffer pre-filled with junk
    char dst[kNameSize];
    // Junk
    std::memset(dst, 'x', sizeof dst);
    // Copy
    CopyName(dst, "Liver");
    // Exact text
    EXPECT_STREQ(dst, "Liver");
}

// A name of exactly 63 characters fits with its terminator
TEST(CopyNameTest, MaximumLengthFits) {
    // 63 characters
    const std::string name(kNameSize - 1, 'a');
    // Buffer
    char dst[kNameSize];
    // Copy
    CopyName(dst, name.c_str());
    // Unchanged
    EXPECT_EQ(std::string(dst), name);
}

// A longer name is truncated to 63 characters and still terminated
TEST(CopyNameTest, LongNameIsTruncatedAndTerminated) {
    // 100 characters
    const std::string name(100, 'b');
    // Buffer
    char dst[kNameSize];
    // Copy
    CopyName(dst, name.c_str());
    // 63 characters
    EXPECT_EQ(std::strlen(dst), static_cast<size_t>(kNameSize - 1));
    // Terminator in the last byte
    EXPECT_EQ(dst[kNameSize - 1], '\0');
}

// An empty name gives an empty string
TEST(CopyNameTest, EmptyName) {
    // Buffer with junk
    char dst[kNameSize] = "junk";
    // Copy ""
    CopyName(dst, "");
    // Empty
    EXPECT_STREQ(dst, "");
}

// Fixture: a zero-filled SharedState, as the parent process creates it
class SeqlockTest : public ::testing::Test {
protected:
    // Heap-allocated (it is several KB) and value-initialised to zero
    std::unique_ptr<SharedState> state_ = std::make_unique<SharedState>();
};

// Nothing written yet: ReadStatus returns 0 and leaves the output alone
TEST_F(SeqlockTest, ReadBeforeFirstWriteReturnsZero) {
    // Output with a marker
    Status out{};
    // Marker
    out.frame_index = 99;
    // Nothing to read
    EXPECT_EQ(ReadStatus(*state_, out), 0u);
    // Untouched
    EXPECT_EQ(out.frame_index, 99);
}

// One write: the counter goes 0 → 2 and the reader gets exactly what was written
TEST_F(SeqlockTest, WriteThenReadRoundTrips) {
    // A status
    Status in{};
    // Fields
    in.timestamp = 1234;
    // Frame
    in.frame_index = 7;
    // Phase
    CopyName(in.phase, "ClippingCutting");
    // Two livers
    in.detected[4] = 2;
    // Progress
    in.progress = 0.31f;
    // Minutes
    in.remaining_minutes = 25;
    // Write
    WriteStatus(*state_, in);
    // Even counter after the write
    EXPECT_EQ(state_->status_seq.load(), 2u);
    // Read back
    Status out{};
    // Sequence of the copy
    EXPECT_EQ(ReadStatus(*state_, out), 2u);
    // Byte-for-byte equal
    EXPECT_EQ(std::memcmp(&in, &out, sizeof(Status)), 0);
}

// Every write advances the counter by 2 and the reader always sees the latest status
TEST_F(SeqlockTest, CounterAdvancesByTwoPerWrite) {
    // Status
    Status s{};
    // Ten writes
    for (int i = 1; i <= 10; ++i) {
        // New frame
        s.frame_index = i;
        // Write
        WriteStatus(*state_, s);
        // Read
        Status out{};
        // 2 * i
        EXPECT_EQ(ReadStatus(*state_, out), static_cast<uint64_t>(2 * i));
        // Latest
        EXPECT_EQ(out.frame_index, i);
    }
}

// A concurrent reader never sees a torn status (all fields of one write together)
TEST_F(SeqlockTest, ConcurrentReaderNeverSeesTornStatus) {
    // Writer finished
    std::atomic<bool> done{false};
    // Torn copies seen by the reader
    std::atomic<int> torn{0};
    // Successful reads
    std::atomic<int> reads{0};
    // Reader thread
    std::thread reader([&] {
        // Until the writer is done
        while (!done.load()) {
            // Copy
            Status out{};
            // Nothing yet
            if (ReadStatus(*state_, out) == 0) continue;
            // Each write sets every detected[] slot and frame_index to the same value
            for (int c = 0; c < kMaxClasses; ++c) {
                // Mismatch = torn
                if (out.detected[c] != static_cast<int32_t>(out.frame_index)) {
                    // Count it
                    ++torn;
                    // One per copy is enough
                    break;
                }
            }
            // Count the read
            ++reads;
        }
    });
    // Writer: 20000 statuses
    for (int i = 1; i <= 20000; ++i) {
        // Status
        Status s{};
        // Frame
        s.frame_index = i;
        // Same value everywhere
        for (int c = 0; c < kMaxClasses; ++c) s.detected[c] = i;
        // Publish
        WriteStatus(*state_, s);
    }
    // Stop the reader
    done = true;
    // Wait
    reader.join();
    // Never torn
    EXPECT_EQ(torn.load(), 0);
    // The reader did read
    EXPECT_GT(reads.load(), 0);
}

// The control flags start at zero and are plain atomics
TEST_F(SeqlockTest, FlagsStartAtZero) {
    // Not ready
    EXPECT_EQ(state_->tables_ready.load(), 0u);
    // No stop request
    EXPECT_EQ(state_->stop_requested.load(), 0u);
    // Parent asks to stop
    state_->stop_requested.store(1);
    // Visible
    EXPECT_EQ(state_->stop_requested.load(), 1u);
}

}  // namespace
