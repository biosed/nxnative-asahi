// Bounded mesh-boundary telemetry shared by the replay and its device-free
// test.  One process emits at most one direct and one indirect line.
#pragma once
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *task_key, *mesh_key, *fragment_key;
    int indirect;
    uint32_t first, count;
    uint64_t address;
} vkx_mesh_trace_event;

static inline int vkx_mesh_trace_pick(const int *stage, const char (*key)[17],
                                      unsigned n, int wanted, char out[17]) {
    if (!stage || !key || !out) return 0;
    for (unsigned i = 0; i < n; i++) if (stage[i] == wanted) { memcpy(out, key[i], 17); return 1; }
    return 0;
}

static inline int vkx_mesh_trace_take(_Atomic unsigned *seen, int indirect) {
    unsigned bit = indirect ? 2u : 1u;
    return !(atomic_fetch_or(seen, bit) & bit);
}

static inline int vkx_mesh_trace_format(char *out, size_t n, const vkx_mesh_trace_event *e) {
    if (!out || !n || !e) return -1;
    int z = e->indirect ?
        snprintf(out, n, "mesh boundary indirect ts=%s ms=%s ps=%s address=%#llx record={count@+0,first@+4,stride=8}",
                 e->task_key, e->mesh_key, e->fragment_key, (unsigned long long)e->address) :
        snprintf(out, n, "mesh boundary direct ts=%s ms=%s ps=%s first=%u count=%u",
                 e->task_key, e->mesh_key, e->fragment_key, e->first, e->count);
    return z >= 0 && (size_t)z < n ? z : -1;
}

