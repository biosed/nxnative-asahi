// Shader compiles in flight, so a long driver compile is seen while it runs instead of as a silent stall: vk_draw.c
// sh_object enters each vkCreateShadersEXT of SPIR-V here, the compile watcher thread and the GPU heartbeat
// (vk_replay.c gp_thread) print "N compiling, oldest <key> ... for S s" from it. Ocarina's NDVE kernel
// d15370a382630d4e (7.4 MB of SPIR-V) took 57 s on Honeykrisp and looked like a GPU hang.
// Pure bookkeeping on a caller-owned, zero-initialized table, lock-free and allocation-free, checked without a
// device by tests/t_vk_compile_watch.c.
#pragma once
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { VKX_CW_SLOTS = 16, VKX_CW_KEY = 16 };
typedef struct {
    _Atomic int state;             // 0 free, 1 being filled, 2 in flight
    char key[VKX_CW_KEY + 1];      // the last byte is never written, so a reader racing a writer still sees a terminated string
    int stage, gl;
    uint64_t bytes, t0;            // SPIR-V bytes handed to the driver; the compile's start (CLOCK_MONOTONIC ns)
} vkx_cw_slot;
typedef struct { vkx_cw_slot s[VKX_CW_SLOTS]; _Atomic uint64_t dropped; } vkx_cw_table;

// A compile beginning at now: its slot, or -1 when every slot is taken (the compile runs unwatched and is counted in dropped)
static inline int vkx_cw_begin(vkx_cw_table *t, const char *key, int stage, int gl, uint64_t bytes, uint64_t now) {
    for (int i = 0; i < VKX_CW_SLOTS; i++) {
        int free_ = 0;
        if (!atomic_compare_exchange_strong(&t->s[i].state, &free_, 1)) continue;
        vkx_cw_slot *s = &t->s[i];
        snprintf(s->key, VKX_CW_KEY + 1, "%s", key ? key : "?");
        s->stage = stage; s->gl = gl; s->bytes = bytes; s->t0 = now;
        atomic_store(&s->state, 2);
        return i;
    }
    atomic_fetch_add(&t->dropped, 1);
    return -1;
}
static inline void vkx_cw_end(vkx_cw_table *t, int slot) {
    if (slot >= 0 && slot < VKX_CW_SLOTS) atomic_store(&t->s[slot].state, 0);
}
// The compiles in flight for at least min_ns at now, as one line for a log: their count and the oldest one,
// "2 compiling, oldest d15370a382630d4e stage 5 gl 0 7.42 MB SPIR-V for 31.0 s". Returns the count; out is "" when 0.
static inline int vkx_cw_text(const vkx_cw_table *t, uint64_t now, uint64_t min_ns, char *out, size_t n) {
    int count = 0, oldest = -1; uint64_t age = 0;
    for (int i = 0; i < VKX_CW_SLOTS; i++) {
        const vkx_cw_slot *s = &t->s[i];
        if (atomic_load(&s->state) != 2) continue;
        uint64_t a = now >= s->t0 ? now - s->t0 : 0;
        if (a < min_ns) continue;
        count++;
        if (oldest < 0 || a > age) { oldest = i; age = a; }
    }
    if (n) out[0] = 0;
    if (!count) return 0;
    const vkx_cw_slot *s = &t->s[oldest];
    snprintf(out, n, "%d compiling, oldest %s stage %d gl %d %.2f MB SPIR-V for %.1f s",
             count, s->key, s->stage, s->gl, (double)s->bytes / 1e6, (double)age / 1e9);
    return count;
}
