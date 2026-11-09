#ifndef DM_VK_CS_GPU_LIFECYCLE_H
#define DM_VK_CS_GPU_LIFECYCLE_H

#include <stdint.h>

// The replay thread owns recording.  submitted/timeline are published under
// g_tmu only after vkQueueSubmit succeeds, so a presentation thread can never
// mistake an in-progress reuse of a ring slot for its previous submission.
typedef struct {
    uint16_t recording;
    uint16_t submitted;
    uint64_t timeline;
} vkr_cs_gpu_batch;

static inline int vkr_cs_gpu_record(vkr_cs_gpu_batch *b, uint16_t limit) {
    if (b->timeline || b->submitted || b->recording >= limit) return -1;
    return b->recording++;
}

static inline int vkr_cs_gpu_publish(vkr_cs_gpu_batch *b, uint64_t timeline) {
    if (b->timeline || b->submitted) return -1;
    if (!b->recording) return 0;
    if (!timeline) return -1;
    b->submitted = b->recording;
    b->recording = 0;
    b->timeline = timeline;
    return b->submitted;
}

static inline int vkr_cs_gpu_harvestable(const vkr_cs_gpu_batch *b, uint64_t done) {
    return b->timeline && b->timeline <= done;
}

static inline void vkr_cs_gpu_retire(vkr_cs_gpu_batch *b) {
    b->submitted = 0;
    b->timeline = 0;
}

#endif
