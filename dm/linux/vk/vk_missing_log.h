// Bounded per-process key set for DM_VK_MISSING_LOG. The shader table already
// remembers failed translations, but this explicit set keeps the diagnostic
// sink independently bounded and makes its once-per-key contract testable.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum { VKX_MISSING_LOG_MAX = 4096 };

typedef struct {
    uint64_t key[VKX_MISSING_LOG_MAX];
    uint8_t used[VKX_MISSING_LOG_MAX];
    uint32_t count;
} vkx_missing_log_set;

// 1: first occurrence, 0: duplicate, -1: the fixed set is full.
static inline int vkx_missing_log_once(vkx_missing_log_set *s, uint64_t key) {
    if (!s) return -1;
    uint64_t h = key;
    h ^= h >> 33; h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33; h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    size_t p = (size_t)h & (VKX_MISSING_LOG_MAX - 1u);
    for (size_t n = 0; n < VKX_MISSING_LOG_MAX; n++, p = (p + 1u) & (VKX_MISSING_LOG_MAX - 1u)) {
        if (!s->used[p]) {
            s->used[p] = 1; s->key[p] = key; s->count++;
            return 1;
        }
        if (s->key[p] == key) return 0;
    }
    return -1;
}

static inline int vkx_missing_log_line(char *out, size_t cap,
                                       const char key[17], int index) {
    if (!out || !cap || !key) return -1;
    int n = snprintf(out, cap, "%s\t%d\n", key, index);
    return n >= 0 && (size_t)n < cap ? n : -1;
}
