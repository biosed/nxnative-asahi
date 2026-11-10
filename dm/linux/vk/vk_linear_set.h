// The LINEAR textures registered in the bindless pool, each once (vk_replay.c vkr_cpu_written_refresh). The font engine rasterizes Ocarina of Time's
// dialogue glyph atlas (2048x8192 R8, LINEAR) on the CPU after it is registered, and samples it only through the pool, so the replay re-uploads every
// registered LINEAR texture whose pages the CPU wrote. Kept as a set filled at registration time: the pool holds hundreds of thousands of IDs, too many to
// scan per submission, and the earlier scan of the first 64 IDs missed the atlas registered after them (blank message windows).
#ifndef VK_LINEAR_SET_H
#define VK_LINEAR_SET_H
#include <stdint.h>
#include <stdlib.h>

typedef struct { int id; uint64_t serial; } vkr_lin_entry;
typedef struct { vkr_lin_entry *e; size_t n, cap; } vkr_lin_set;

// Registration of a LINEAR texture: noted once per serial; another ID of the same texture moves the entry to that newest ID, so reusing an older ID
// for something else does not drop a texture still registered. 0 if out of memory
static inline int vkr_lin_note(vkr_lin_set *s, int id, uint64_t serial) {
    for (size_t i = 0; i < s->n; i++) if (s->e[i].serial == serial) { s->e[i].id = id; return 1; }
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
#endif
