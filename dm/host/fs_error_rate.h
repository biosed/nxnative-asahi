// A repeated host-file-system error can sit in a guest retry loop. Keep the
// first occurrence and power-of-two progress reports without turning one bad
// request into hundreds of thousands of log lines.
#pragma once
#include <pthread.h>
#include <stdint.h>

#define FS_ERROR_RATE_SLOTS 64
typedef struct {
    pthread_mutex_t lock;
    struct { uint64_t key, count; } slot[FS_ERROR_RATE_SLOTS];
} fs_error_rate;
#define FS_ERROR_RATE_INIT { PTHREAD_MUTEX_INITIALIZER, {{0, 0}} }

static inline uint64_t fs_error_hash_string(uint64_t h, const char *s) {
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ull; }
    return h;
}

// Returns true for occurrences 1, 2, 4, 8, ... and writes the full count.
// A direct-mapped collision starts a new series; it can add a diagnostic but
// can never hide the first occurrence of a different error.
static inline int fs_error_rate_hit(fs_error_rate *r, const char *op,
                                    const char *path, int error,
                                    uint64_t *count) {
    uint64_t key = fs_error_hash_string(1469598103934665603ull, op);
    key = fs_error_hash_string(key ^ 0xff, path);
    key ^= (uint32_t)error; key *= 1099511628211ull;
    if (!key) key = 1;
    size_t i = key % FS_ERROR_RATE_SLOTS;
    pthread_mutex_lock(&r->lock);
    if (r->slot[i].key != key) { r->slot[i].key = key; r->slot[i].count = 0; }
    uint64_t n = ++r->slot[i].count;
    pthread_mutex_unlock(&r->lock);
    *count = n;
    return !(n & (n - 1));
}
