// Bounded live pass-log requests. The control-file thread only publishes a
// request; the replay thread owns remaining and advances it at present
// boundaries, so tracing never changes in the middle of a frame.
#pragma once
#include <stdatomic.h>

enum { VKP_PASSLOG_MAX_FRAMES = 64 };

typedef struct {
    _Atomic long request;
    long remaining;
} vkp_passlog;

static inline void vkp_passlog_request(vkp_passlog *p, long frames) {
    if (frames < 1) return;
    if (frames > VKP_PASSLOG_MAX_FRAMES) frames = VKP_PASSLOG_MAX_FRAMES;
    atomic_store_explicit(&p->request, frames, memory_order_release);
}

// Select whether the frame about to be recorded is traced. A newer request
// replaces an active one, matching DM_CTL's latest-command-wins behaviour.
static inline int vkp_passlog_next(vkp_passlog *p) {
    long request = atomic_exchange_explicit(&p->request, 0, memory_order_acq_rel);
    if (request > 0) p->remaining = request;
    if (p->remaining < 1) return 0;
    p->remaining--;
    return 1;
}
