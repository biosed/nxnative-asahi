// The LINEAR texture registrations in the bindless pool (vk_replay.c vkr_cpu_written_refresh), one entry per ID. The pool holds hundreds of thousands of
// IDs, too many to scan per submission. A texture can be live under several IDs, so snapshotting deduplicates serials only after stale ID registrations are
// pruned; keeping only one representative ID loses the texture if that ID is reused while another alias remains live.
#ifndef VK_LINEAR_SET_H
#define VK_LINEAR_SET_H
#include <stdint.h>
#include <stdlib.h>

typedef struct { int id; uint64_t serial; } vkr_lin_entry;
typedef struct { vkr_lin_entry *e; size_t n, cap; } vkr_lin_set;

// Registration of a LINEAR texture: keep the current serial once per ID. 0 if out of memory
static inline int vkr_lin_note(vkr_lin_set *s, int id, uint64_t serial) {
    for (size_t i = 0; i < s->n; i++) if (s->e[i].id == id) { s->e[i].serial = serial; return 1; }
    if (s->n == s->cap) {
        size_t c = s->cap ? 2 * s->cap : 64;
        vkr_lin_entry *e = realloc(s->e, c * sizeof *e);
        if (!e) return 0;
        s->e = e; s->cap = c;
    }
    s->e[s->n++] = (vkr_lin_entry){id, serial};
    return 1;
}

// Drops the entries whose ID no longer holds that texture (live(id, serial) == 0), keeping the order of the rest; returns how many remain
static inline size_t vkr_lin_prune(vkr_lin_set *s, int (*live)(void *ctx, int id, uint64_t serial), void *ctx) {
    size_t w = 0;
    for (size_t r = 0; r < s->n; r++) if (live(ctx, s->e[r].id, s->e[r].serial)) s->e[w++] = s->e[r];
    return s->n = w;
}
// True only for the first live registration of this texture. Iterate in order
// and snapshot entries for which this returns true to refresh each serial once.
static inline int vkr_lin_first_serial(const vkr_lin_set *s, size_t at) {
    for (size_t i = 0; i < at; i++) if (s->e[i].serial == s->e[at].serial) return 0;
    return 1;
}
#endif
