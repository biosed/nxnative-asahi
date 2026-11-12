// Compute dispatches the replay leaves out on purpose (vk_draw.c vkx_dispatch), for GPUs that cannot run a known kernel in time: Ocarina of Time's NDVE
// movie-enhancement network has four compute shaders whose HMMA chains become scalar code on Honeykrisp and take 237-251 ms of GPU a frame.
//   DM_VK_SKIP_CS=<key>[,<key>...]   skip only these exact compute shader keys (16 hex digits each, as shader_map.tsv names them)
// There is deliberately no size/prefix rule: enabling the diagnostic cannot silently catch a newly generated or unrelated shader. Unset means no change to dispatches.
// A match drops the dispatch and never substitutes a blit or another operation. Pure (no Vulkan), so vk/tests/t_skipcs.c checks the parser, exact matching and disabled
// default; each selected key is logged once by the caller.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VKX_SKIPCS_MAX 32
typedef struct { char key[VKX_SKIPCS_MAX][17]; int n, on; uint64_t skipped; } vkx_skipcs;

static inline int vkx_skipcs_sep(char c) { return c == ',' || c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
// From DM_VK_SKIP_CS's value (NULL when unset): the keys taken are exactly 16 hex digits each, separated by commas/whitespace; anything else is dropped (returned count)
static inline int vkx_skipcs_parse(vkx_skipcs *c, const char *keys) {
    memset(c, 0, sizeof *c); int dropped = 0;
    for (const char *p = keys; p && *p;) {
        while (vkx_skipcs_sep(*p)) p++;
        if (!*p) break;
        const char *e = p; while (*e && !vkx_skipcs_sep(*e)) e++;
        int ok = e - p == 16;
        for (const char *q = p; ok && q < e; q++) ok = (*q >= '0' && *q <= '9') || (*q >= 'a' && *q <= 'f') || (*q >= 'A' && *q <= 'F');
        if (ok && c->n < VKX_SKIPCS_MAX) { memcpy(c->key[c->n], p, 16); c->key[c->n][16] = 0; for (char *q = c->key[c->n]; *q; q++) if (*q >= 'A' && *q <= 'F') *q += 'a' - 'A'; c->n++; }
        else dropped++;
        p = e;
    }
    c->on = c->n > 0;
    return dropped;
}
// Whether a dispatch of this exact compute shader key (16 lowercase hex digits) is skipped; counts every selected dispatch
static inline int vkx_skipcs_match(vkx_skipcs *c, const char *key) {
    if (!c->on) return 0;
    for (int i = 0; i < c->n; i++) if (!strcmp(c->key[i], key)) { c->skipped++; return 1; }
    return 0;
}

// DM_VK_RELAX_CS_BATCH: number of selected consecutive dispatches allowed
// before the diagnostic restores the ordinary full barrier.  One is a
// no-overlap control; 64 covers the bounded hot blocks seen so far.
static inline unsigned vkx_relaxcs_batch_parse(const char *s, int *bad) {
    *bad = 0;
    if (!s || !*s) return 2;
    char *end = NULL;
    unsigned long n = strtoul(s, &end, 10);
    if (end == s || *end || n < 1 || n > 64) { *bad = 1; return 2; }
    return (unsigned)n;
}

// Pure replay scheduling rule: a selected dispatch may follow the selected
// run until its bounded batch is full; state records do not issue GPU work;
// every other GPU operation is ordered before it.
static inline int vkx_relaxcs_barrier_before(unsigned run, int selected, int state_only, unsigned batch) {
    return run && ((selected && run >= batch) || (!selected && !state_only));
}
