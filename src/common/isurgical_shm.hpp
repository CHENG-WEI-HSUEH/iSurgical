// Include guard: make sure this header is only processed once per translation unit.
#pragma once

// Shared-memory layout between isurgical (writer, the child process) and a controller such as
// iprovision's AppManager (reader, the parent). The parent creates the segment with shm_open,
// zero-fills it, sets magic / version and starts isurgical with --shm <name>.
//
//   tables : written once after the config is loaded, then tables_ready is set to 1
//   status : rewritten for every processed frame, guarded by the status_seq seqlock
//
// Only fixed-size, trivially copyable members, so the layout is identical in both processes.
// Both sides must be built from this same header; version is bumped on any layout change.

// std::atomic and std::atomic_thread_fence, used for cross-process synchronisation.
#include <atomic>
// Fixed-width integer types (uint32_t, int64_t, ...), so field sizes are the same everywhere.
#include <cstdint>
// std::strncpy and std::memcpy.
#include <cstring>
// std::is_trivially_copyable_v, used in the static_assert below.
#include <type_traits>

// All shared-memory definitions live in their own namespace to avoid name clashes.
namespace isurgical_shm {

// Magic number stored in SharedState::magic by the parent. Its bytes in memory (little-endian)
// spell "ISUG"; the child checks it to confirm the segment really is an isurgical segment.
inline constexpr uint32_t kMagic = 0x47555349;
// Layout version stored in SharedState::version; must match on both sides, bump on any layout change.
inline constexpr uint32_t kVersion = 1;
// Maximum number of surgical phases that fit in Tables::phase_names.
inline constexpr int kMaxPhases = 16;
// Maximum number of segmentation classes that fit in the per-class arrays.
inline constexpr int kMaxClasses = 32;
// Size of every fixed-length name buffer, including the terminating '\0' (so at most 63 characters).
inline constexpr int kNameSize = 64;

// Static information from the procedure config
struct Tables {
    // Number of valid entries in phase_names (<= kMaxPhases).
    int32_t num_phases;
    // Phase names, in the same order as the phase model's output.
    char phase_names[kMaxPhases][kNameSize];
    // Number of valid entries in class_names / class_colours (<= kMaxClasses).
    int32_t num_classes;
    // Class names from label_names; the array index is the global class id.
    char class_names[kMaxClasses][kNameSize];
    // Display colour of each class (from kclass_colours), RGB components in [0, 1].
    float class_colours[kMaxClasses][3];
// End of Tables.
};

// Latest result, refreshed for every processed frame
struct Status {
    // time_t (seconds since the epoch) when this status was written.
    int64_t timestamp;
    // Index of the source frame the segmentation result belongs to.
    int64_t frame_index;
    // Latest phase name; "" before the first phase result or when RSD is disabled.
    char phase[kNameSize];
    // Number of regions of each class found in this frame; 0 means not detected.
    int32_t detected[kMaxClasses];
    // Surgery progress in [0, 1], rounded to 2 decimals; -1 if unknown.
    float progress;
    // Remaining surgical duration in rounded minutes; -1 if unknown.
    int32_t remaining_minutes;
// End of Status.
};

// The complete object that is mapped into both processes' address spaces.
struct SharedState {
    // Must equal kMagic; set by the parent, checked by isurgical when it attaches.
    uint32_t magic;
    // Must equal kVersion; set by the parent, checked by isurgical when it attaches.
    uint32_t version;
    // Becomes 1 once isurgical has finished writing tables; the reader must not use tables before that.
    std::atomic<uint32_t> tables_ready;
    // Set by the parent to ask isurgical to stop; isurgical then finishes cleanly.
    std::atomic<uint32_t> stop_requested;
    // Seqlock counter for status: odd while status is being written, 0 before the first write.
    std::atomic<uint64_t> status_seq;
    // Static config tables (written once).
    Tables tables;
    // Per-frame result (rewritten for every frame).
    Status status;
// End of SharedState.
};

// Atomics in shared memory only work across processes if they are lock-free (no hidden mutex
// inside the object, which would be per-process); refuse to compile otherwise.
static_assert(std::atomic<uint32_t>::is_always_lock_free && std::atomic<uint64_t>::is_always_lock_free,
              // Error message shown when the assertion fails.
              "shared-memory atomics must be lock-free");
// Tables and Status are copied with memcpy, which is only valid for trivially copyable types.
static_assert(std::is_trivially_copyable_v<Tables> && std::is_trivially_copyable_v<Status>);

// Copies a string into a fixed-size name field, truncating if needed
inline void CopyName(char (&dst)[kNameSize], const char* src) {
    // Copy at most kNameSize - 1 characters, leaving room for the terminator.
    std::strncpy(dst, src, kNameSize - 1);
    // strncpy does not terminate when src is too long, so always write the '\0' explicitly.
    dst[kNameSize - 1] = '\0';
// End of CopyName.
}

// Writer side (single writer)
inline void WriteStatus(SharedState& state, const Status& status) {
    // Read the current (even) sequence number; relaxed is enough because only this thread writes it.
    const uint64_t seq = state.status_seq.load(std::memory_order_relaxed);
    // Make the counter odd to signal "write in progress" to readers.
    state.status_seq.store(seq + 1, std::memory_order_relaxed);
    // Ensure the odd counter becomes visible before any of the status bytes below are written.
    std::atomic_thread_fence(std::memory_order_release);
    // Copy the new status into shared memory.
    std::memcpy(&state.status, &status, sizeof(Status));
    // Make the counter even again; release publishes the status bytes together with the new value.
    state.status_seq.store(seq + 2, std::memory_order_release);
// End of WriteStatus.
}

// Reader side; returns the sequence number of the copy, 0 if nothing has been written yet
inline uint64_t ReadStatus(const SharedState& state, Status& status) {
    // Retry until a consistent (not torn) copy has been taken.
    for (;;) {
        // Read the counter before copying; acquire makes the writer's earlier status bytes visible.
        const uint64_t before = state.status_seq.load(std::memory_order_acquire);
        // 0 means the writer has never written a status, so there is nothing to read.
        if (before == 0) return 0;
        // An odd value means a write is in progress; try again.
        if (before & 1) continue;
        // Copy the status out of shared memory (it may still be overwritten while we copy).
        std::memcpy(&status, &state.status, sizeof(Status));
        // Keep the copy above from being reordered after the counter check below.
        std::atomic_thread_fence(std::memory_order_acquire);
        // If the counter is unchanged, no write happened during the copy, so the copy is valid.
        if (state.status_seq.load(std::memory_order_relaxed) == before) return before;
    // Otherwise the copy may be torn; loop and read again.
    }
// End of ReadStatus.
}

// Close the namespace.
}  // namespace isurgical_shm
