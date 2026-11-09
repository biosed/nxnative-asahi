#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Demand-driven bindless residency + LRU eviction helpers under a strict
// speculative-image byte budget (e.g. DM_VK_BINDLESS_IMAGE_MB=512).
//
// Key invariant: GPU feedback bits in g_feedback_map[slot] must update
// id_last_used[id] for ALREADY-RESIDENT descriptors (real[id] != 0) as well as
// queuing unresolved descriptors (real[id] == 0). Otherwise resident bindless
// textures keep their initial registration timestamp and cannot be distinguished
// from stale title-screen textures when Kokiri Forest fills the 512 MiB budget.

typedef struct {
    uint64_t image_key;      // caller identifier (e.g. vtex serial or index)
    uint64_t bytes;          // allocation size in bytes
    uint64_t last_used_sub;  // max(t->last_sample, max id_last_used[id] for ids -> t)
    uint8_t  speculative;    // 1 if counted against the speculative bindless budget
    uint8_t  gpu_owned;      // 1 if render target / storage image (never evict)
    uint8_t  pinned;         // 1 if held for presentation or active in current submit
} vkx_bindless_lru_cand;

// Scans a completed command-buffer slot's feedback bitmap (nwords 32-bit words,
// bit b of word w = texture ID w*32 + b), updates id_last_used[id] = current_sub
// for every observed ID, appends unresolved IDs (real[id] == 0) to
// unresolved_ids[0..unresolved_cap-1] without duplicates, and zeroes bitmap.
// Returns the number of newly observed unresolved IDs added to unresolved_ids.
static inline unsigned vkx_bindless_feedback_harvest(
    uint32_t *bitmap, unsigned nwords, unsigned max_ids, uint64_t current_sub,
    const uint8_t *real, uint64_t *id_last_used,
    int *unresolved_ids, unsigned *unresolved_n, unsigned unresolved_cap) {
    if (!bitmap || !nwords) return 0;
    unsigned fresh = 0;
    for (unsigned w = 0; w < nwords; w++) {
        uint32_t bits = bitmap[w];
        if (!bits) continue;
        bitmap[w] = 0;
        while (bits) {
            unsigned b = (unsigned)__builtin_ctz(bits);
            unsigned id = w * 32u + b;
            bits &= bits - 1u;
            if (!id || id >= max_ids) continue;
            if (id_last_used) id_last_used[id] = current_sub;
            if (real && real[id]) continue;
            if (!unresolved_ids || !unresolved_n) continue;
            int seen = 0;
            for (unsigned i = 0; i < *unresolved_n; i++) {
                if (unresolved_ids[i] == (int)id) { seen = 1; break; }
            }
            if (!seen && *unresolved_n < unresolved_cap) {
                unresolved_ids[(*unresolved_n)++] = (int)id;
                fresh++;
            }
        }
    }
    return fresh;
}

// Computes the effective last-used submission of an image from its direct
// tex_sampled() stamp and all bindless texture IDs currently mapped to it.
static inline uint64_t vkx_bindless_image_last_used(
    uint64_t direct_last_sample, const int *bound_ids, int nbound_ids,
    const uint64_t *id_last_used, unsigned max_ids) {
    uint64_t last = direct_last_sample;
    if (!bound_ids || nbound_ids <= 0 || !id_last_used) return last;
    for (int i = 0; i < nbound_ids; i++) {
        int id = bound_ids[i];
        if (id > 0 && (unsigned)id < max_ids && id_last_used[id] > last)
            last = id_last_used[id];
    }
    return last;
}

// Returns the number of bytes that must be evicted from used_bytes so that
// request_bytes fits within budget_bytes (0 if it already fits or budget == 0).
static inline uint64_t vkx_bindless_lru_deficit(
    uint64_t used_bytes, uint64_t budget_bytes, uint64_t request_bytes) {
    if (!budget_bytes || request_bytes > budget_bytes) return 0;
    if (used_bytes <= budget_bytes && request_bytes <= budget_bytes - used_bytes)
        return 0;
    uint64_t target = budget_bytes - request_bytes;
    return used_bytes > target ? used_bytes - target : 0;
}

// Selects up to max_victims eligible speculative candidates in LRU order
// (oldest last_used_sub first; ties broken by larger byte size, then lower
// image_key) until at least deficit_bytes are reclaimed. Candidates used within
// the last min_age_subs submissions (last_used_sub + min_age_subs > current_sub)
// are protected from eviction to prevent thrashing active scene textures.
// Writes winning image_keys to out_keys[0..return-1] and total reclaimed bytes
// to *reclaimed_bytes.
static inline int vkx_bindless_lru_select(
    const vkx_bindless_lru_cand *cands, int ncands,
    uint64_t deficit_bytes, uint64_t current_sub, uint64_t min_age_subs,
    uint64_t *out_keys, int max_victims, uint64_t *reclaimed_bytes) {
    if (reclaimed_bytes) *reclaimed_bytes = 0;
    if (!cands || ncands <= 0 || !deficit_bytes || !out_keys || max_victims <= 0)
        return 0;
    uint64_t freed = 0;
    int nvict = 0;
    while (nvict < max_victims && freed < deficit_bytes) {
        int best = -1;
        for (int i = 0; i < ncands; i++) {
            const vkx_bindless_lru_cand *c = &cands[i];
            if (!c->speculative || c->gpu_owned || c->pinned || !c->bytes)
                continue;
            if (current_sub < c->last_used_sub ||
                current_sub - c->last_used_sub < min_age_subs)
                continue;
            int already = 0;
            for (int k = 0; k < nvict; k++) {
                if (out_keys[k] == c->image_key) { already = 1; break; }
            }
            if (already) continue;
            if (best < 0) {
                best = i;
                continue;
            }
            const vkx_bindless_lru_cand *b = &cands[best];
            if (c->last_used_sub < b->last_used_sub ||
                (c->last_used_sub == b->last_used_sub && c->bytes > b->bytes) ||
                (c->last_used_sub == b->last_used_sub && c->bytes == b->bytes &&
                 c->image_key < b->image_key)) {
                best = i;
            }
        }
        if (best < 0) break;
        out_keys[nvict++] = cands[best].image_key;
        freed += cands[best].bytes;
    }
    if (reclaimed_bytes) *reclaimed_bytes = freed;
    return nvict;
}
