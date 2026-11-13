#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "dmsh_abi.h"

enum {
    VKX_GMEM_MAP_LOG2 = 13,
    VKX_GMEM_MAP_MASK = DMSH_GMEM_MAP_ENTRIES - 1,
    VKX_GMEM_HINT_TAG_LIMIT = 1u << (32 - VKX_GMEM_MAP_LOG2),
};

// Flat table (DM_VK_GMEM_FLAT=1, vk_mem.c). Entry 0 is empty and the search never reads its delta (bytes 16..23): a flat table puts VKX_GMEM_FLAT_TAG in
// that word's low half and the entry count n (1..VKX_GMEM_FLAT_MAX) in its high half. Entries 1..n are then whole guest regions, sorted, every byte bound
// on the GPU, so a translator may resolve an address by testing those n entries' [start, end) in order (address + delta of the one holding it) instead of
// searching; an address in none of them is refused as before. Kept out of dmsh_abi.h for now so the host change does not alter the shader chain ID: a
// translator fast path that reads it must move these two values into dmsh_abi.h with its own chain change.
#define VKX_GMEM_FLAT_TAG 0x54414C46u   // 'FLAT'
#define VKX_GMEM_FLAT_MAX 4

static inline void vkx_gmem_map_mark_flat(void *dst, size_t size, int n) {
    if (!dst || size < (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY || n < 1 || n > VKX_GMEM_FLAT_MAX) return;
    uint32_t w[2] = {VKX_GMEM_FLAT_TAG, (uint32_t)n};
    memcpy((uint8_t *)dst + 16, w, sizeof w);
}

// What a translator fast path does with a flat table: 1 with the GPU address when addr lies in one of entries 1..n, 0 when outside all of them,
// -1 when the table is not flat (the caller searches).
static inline int vkx_gmem_map_eval_flat(const void *map, uint64_t addr, uint64_t *host_addr) {
    const uint32_t *w = (const uint32_t *)map;
    if (!map || w[4] != VKX_GMEM_FLAT_TAG || w[5] < 1 || w[5] > VKX_GMEM_FLAT_MAX) return -1;
    for (uint32_t k = 1; k <= w[5]; k++) {
        uint64_t e[3]; memcpy(e, (const uint8_t *)map + (size_t)k * DMSH_GMEM_MAP_ENTRY, sizeof e);
        if (addr >= e[0] && addr < e[1]) { if (host_addr) *host_addr = addr + e[2]; return 1; }
    }
    return 0;
}

// Packs sorted guest-memory segments [lo[i], hi[i]) with GPU base gpu[i] into
// the DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY raw-global lookup table,
// including 2-MiB chunk hints at byte +24 (word 6) as GlobalToStorage.cs and
// nvn_mtl.m (gmem_map) expect.
static inline int vkx_gmem_map_pack(void *dst, size_t size,
                                    const uint64_t *lo, const uint64_t *hi,
                                    const uint64_t *gpu, int n) {
    const size_t need = (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    if (!dst || size < need || n < 0 || (n > 0 && (!lo || !hi || !gpu))) return -1;
    uint8_t *d = (uint8_t *)dst;
    memset(d, 0, need);
    for (int k = 1; k < DMSH_GMEM_MAP_ENTRIES; k++)
        memset(d + (size_t)k * DMSH_GMEM_MAP_ENTRY, 0xff, 8);
    int out = 0;
    uint64_t end = 0;
    for (int i = 0; i < n && out < DMSH_GMEM_MAP_ENTRIES - 1; i++) {
        uint64_t l = lo[i] > end ? lo[i] : end, h = hi[i];
        if (l >= h) continue;
        uint64_t e[3] = {l, h, gpu[i] - lo[i]};
        memcpy(d + (size_t)(++out) * DMSH_GMEM_MAP_ENTRY, e, sizeof e);
        end = h;
    }
    const uint64_t chunk_size = 1ull << DMSH_GMEM_HINT_SHIFT;
    for (int k = 1; k <= out; k++) {
        uint64_t e[2];
        memcpy(e, d + (size_t)k * DMSH_GMEM_MAP_ENTRY, sizeof e);
        uint64_t first = (e[0] + chunk_size - 1) >> DMSH_GMEM_HINT_SHIFT;
        uint64_t last = e[1] >> DMSH_GMEM_HINT_SHIFT;
        for (uint64_t c = first; c < last && c < VKX_GMEM_HINT_TAG_LIMIT; c++) {
            uint32_t hint = (uint32_t)((c << VKX_GMEM_MAP_LOG2) | (unsigned)k);
            memcpy(d + (size_t)(c & VKX_GMEM_MAP_MASK) * DMSH_GMEM_MAP_ENTRY + 24,
                   &hint, sizeof hint);
        }
    }
    return out;
}

// Mirrors GlobalToStorage.TryGenerateRawStorageOp's address lookup for unit
// testing: returns 1 if addr resolves to a valid pool entry (writing the
// translated GPU address, winning entry index, and whether the 2-MiB hint
// bypassed the binary search), or 0 if addr is outside every pool.
static inline int vkx_gmem_map_eval(const void *map, uint64_t addr,
                                    uint64_t *host_addr, uint32_t *entry_out,
                                    int *used_hint) {
    if (!map) return 0;
    const uint32_t *w = (const uint32_t *)map;
    uint32_t addr_low = (uint32_t)addr, addr_high = (uint32_t)(addr >> 32);
    uint32_t chunk = (addr_high << (32 - DMSH_GMEM_HINT_SHIFT)) |
                     (addr_low >> DMSH_GMEM_HINT_SHIFT);
    uint32_t hint = w[(chunk & VKX_GMEM_MAP_MASK) * 8u + 6u];
    uint32_t hint_idx = hint & VKX_GMEM_MAP_MASK;
    int hint_fits = hint_idx != 0 &&
                    (hint >> VKX_GMEM_MAP_LOG2) == chunk &&
                    addr_high < (1u << (DMSH_GMEM_HINT_SHIFT - VKX_GMEM_MAP_LOG2));
    uint32_t selected = 0;
    if (hint_fits) {
        selected = hint_idx;
    } else {
        uint32_t idx = 0;
        for (uint32_t step = 1u << (VKX_GMEM_MAP_LOG2 - 1); step != 0; step >>= 1) {
            uint32_t cand = idx + step;
            uint32_t lo = w[cand * 8u + 0u], hi = w[cand * 8u + 1u];
            int at_or_below = hi < addr_high || (hi == addr_high && lo <= addr_low);
            if (at_or_below) idx = cand;
        }
        selected = idx;
    }
    if (used_hint) *used_hint = hint_fits;
    if (entry_out) *entry_out = selected;
    uint32_t end_lo = w[selected * 8u + 2u], end_hi = w[selected * 8u + 3u];
    int below_end = addr_high < end_hi || (addr_high == end_hi && addr_low < end_lo);
    if (!below_end) return 0;
    uint64_t delta = (uint64_t)w[selected * 8u + 4u] |
                     ((uint64_t)w[selected * 8u + 5u] << 32);
    if (host_addr) *host_addr = addr + delta;
    return 1;
}
