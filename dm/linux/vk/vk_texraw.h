// One live host-texture readback request.  The control-file thread only
// publishes the serial and path; the replay thread consumes it at a present
// boundary, where its command pool and texture table are safe to use.
#pragma once

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

enum { VKT_TEXRAW_PATH_MAX = 1024 };

typedef struct {
    atomic_flag lock;
    uint64_t serial;
    char path[VKT_TEXRAW_PATH_MAX];
} vkt_texraw;

#define VKT_TEXRAW_INIT { .lock = ATOMIC_FLAG_INIT }

static inline void vkt_texraw_lock(vkt_texraw *r) {
    while (atomic_flag_test_and_set_explicit(&r->lock, memory_order_acquire)) {}
}

static inline void vkt_texraw_unlock(vkt_texraw *r) {
    atomic_flag_clear_explicit(&r->lock, memory_order_release);
}

// A newer request replaces one that has not reached a present yet.
static inline void vkt_texraw_request(vkt_texraw *r, uint64_t serial,
                                      const char *path) {
    if (!serial || !path || !*path) return;
    vkt_texraw_lock(r);
    r->serial = serial;
    size_t n = strnlen(path, sizeof r->path - 1);
    memcpy(r->path, path, n);
    r->path[n] = 0;
    vkt_texraw_unlock(r);
}

static inline uint64_t vkt_texraw_take(vkt_texraw *r, char *path,
                                       size_t path_size) {
    if (!path || !path_size) return 0;
    vkt_texraw_lock(r);
    uint64_t serial = r->serial;
    if (serial) {
        size_t n = strnlen(r->path, path_size - 1);
        memcpy(path, r->path, n);
        path[n] = 0;
        r->serial = 0;
        r->path[0] = 0;
    }
    vkt_texraw_unlock(r);
    return serial;
}
