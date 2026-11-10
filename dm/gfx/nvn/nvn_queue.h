// Logical NVN queue completion is distinct from a physical Vulkan submission fence.
// Adapted from nxnative ea0c98df (MIT), including queue-only fixes 04419bc2,
// 9b69d27d and 25c2546e.
#pragma once
#include "nvn_obj.h"

void nvn_queue_submit(queue_t *q, int n, const uint64_t *handles);
uint64_t nvn_queue_present(queue_t *q, const tex_t *t, double period, const int crop[4]);
void nvn_queue_present_wait(uint64_t ticket);
void nvn_queue_present_release(uint64_t ticket);
void nvn_queue_fence(queue_t *q, sync_t *s);
void nvn_queue_wait_sync(queue_t *q, const sync_t *s);
void nvn_queue_finish(queue_t *q);
void nvn_queue_forget(queue_t *q);
void nvn_sync_release(sync_t *s);
int nvn_sync_wait(const sync_t *s, uint64_t timeout);
void nvn_event_publish(volatile uint32_t *word, unsigned mode, uint32_t value);
void nvn_event_publish_gpu(volatile uint32_t *word, unsigned mode, uint32_t value, uint64_t fence);
void nvn_queue_gpu_progress(void);

// Cumulative diagnostic counters. The frame summary takes deltas every 300
// frames; the device-free fixture can also assert which dependency path ran.
enum {
    NVN_QUEUE_STAT_COMMAND_NODES,
    NVN_QUEUE_STAT_PHYSICAL_SUBMITS,
    NVN_QUEUE_STAT_EVENT_NODES,
    NVN_QUEUE_STAT_SYNC_NODES,
    NVN_QUEUE_STAT_FENCE_NODES,
    NVN_QUEUE_STAT_PRESENT_NODES,
    NVN_QUEUE_STAT_EXACT_QUEUED_RELEASES,
    NVN_QUEUE_STAT_COMPLETION_RELEASES,
    NVN_QUEUE_STAT_CPU_RELEASES,
    NVN_QUEUE_STAT_CROSS_QUEUE_WAITS,
    NVN_QUEUE_STAT_LOCAL_WAITS,
    NVN_QUEUE_STAT_QUEUE_SWITCHES,
    NVN_QUEUE_STAT_PARKED_NS,
    NVN_QUEUE_STAT_COUNT,
};
void nvn_queue_stats(uint64_t out[NVN_QUEUE_STAT_COUNT]);
int nvn_queue_scheduler_enabled(void);
