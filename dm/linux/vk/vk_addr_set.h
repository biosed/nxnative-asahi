// A set of guest addresses for the replay thread (vk_replay.c: the addresses ever given a non-timestamp counter report, whose value render_cond cannot know at
// replay). Every report adds to it and every conditional render looks two addresses up, so both must stay O(1): the earlier fixed 4096-slot table, hashed by
// a >> 4 and never cleared, filled with Kokiri's samples-passed report addresses and then probed all 4096 slots on every add and every miss (a fifth of the
// replay thread's time). A full table also dropped new addresses, which rc_value then read as known. Here the table doubles at half load; past cap entries it
// stops growing and reports every address as present, so callers fall back to their safe answer (render_cond: unknown, so the draw happens).
#ifndef VK_ADDR_SET_H
#define VK_ADDR_SET_H
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint64_t *slot; uint32_t cap, n, max_cap; int has_zero, saturated; } vkx_addr_set;

static inline uint32_t vkx_addr_set_hash(uint64_t a) {   // murmur3's 64-bit finalizer: report addresses differ only in a few middle bits
    a ^= a >> 33; a *= 0xff51afd7ed558ccdull; a ^= a >> 33; a *= 0xc4ceb9fe1a85ec53ull; a ^= a >> 33;
    return (uint32_t)a;
}
static inline void vkx_addr_set_put(uint64_t *slot, uint32_t cap, uint64_t a) {   // a is not 0 and not present; cap is a power of two with a free slot
    for (uint32_t i = vkx_addr_set_hash(a) & (cap - 1);; i = (i + 1) & (cap - 1)) if (!slot[i]) { slot[i] = a; return; }
}
// Add a. Returns 0 only when the set has saturated (it then answers 1 for every address)
static inline int vkx_addr_set_add(vkx_addr_set *s, uint64_t a) {
    if (s->saturated) return 0;
    if (!a) { s->has_zero = 1; return 1; }
    if (s->cap) for (uint32_t i = vkx_addr_set_hash(a) & (s->cap - 1); s->slot[i]; i = (i + 1) & (s->cap - 1)) if (s->slot[i] == a) return 1;
    if (2 * (s->n + 1) > s->cap) {
        uint32_t c = s->cap ? 2 * s->cap : 1024;
        uint64_t *t = c <= s->max_cap ? calloc(c, sizeof *t) : NULL;
        if (!t) { s->saturated = 1; return 0; }
        for (uint32_t i = 0; i < s->cap; i++) if (s->slot[i]) vkx_addr_set_put(t, c, s->slot[i]);
        free(s->slot); s->slot = t; s->cap = c;
    }
    vkx_addr_set_put(s->slot, s->cap, a); s->n++;
    return 1;
}
static inline int vkx_addr_set_has(const vkx_addr_set *s, uint64_t a) {
    if (s->saturated) return 1;
    if (!a) return s->has_zero;
    if (!s->cap) return 0;
    for (uint32_t i = vkx_addr_set_hash(a) & (s->cap - 1); s->slot[i]; i = (i + 1) & (s->cap - 1)) if (s->slot[i] == a) return 1;
    return 0;
}
#endif
