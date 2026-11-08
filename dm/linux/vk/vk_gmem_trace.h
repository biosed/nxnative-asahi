// Opt-in extension of dmsh_abi.h's raw-global store bitmap. The unmodified
// bitmap stays at the front; the first atomic winner for each map entry appends
// the full 64-bit guest page number. DM_VK_GMEM_TRACE enables this ABI.
#pragma once
#include "../gfx/nvn/dmsh_abi.h"
#include <stddef.h>
#include <stdint.h>

#define VKX_GMEM_PAGE_SHIFT 12u
#define VKX_GMEM_TRACE_BITMAP_WORDS ((DMSH_GMEM_MAP_ENTRIES + 31u) / 32u)
#define VKX_GMEM_TRACE_SLOTS 1024u
#define VKX_GMEM_TRACE_COUNT_WORD VKX_GMEM_TRACE_BITMAP_WORDS
#define VKX_GMEM_TRACE_PAGE_WORD (VKX_GMEM_TRACE_COUNT_WORD + 1u)
#define VKX_GMEM_TRACE_WORDS (VKX_GMEM_TRACE_PAGE_WORD + 2u * VKX_GMEM_TRACE_SLOTS)
#define VKX_GMEM_TRACE_BYTES (4u * VKX_GMEM_TRACE_WORDS)

// Decode the completed log. Each record is a 64-bit page number split low,
// high. When overflow is reported, wrapped pairs are not trustworthy and the
// caller must not invalidate from them.
static inline size_t vkx_gmem_trace_pages(const uint32_t *w, size_t words,
                                           uint64_t *out, size_t cap,
                                           int *overflow, uint32_t *stores) {
    if (overflow) *overflow = 0;
    if (stores) *stores = 0;
    if (!w || words < VKX_GMEM_TRACE_WORDS) return 0;
    uint32_t n = w[VKX_GMEM_TRACE_COUNT_WORD];
    if (stores) *stores = n;
    if (n > VKX_GMEM_TRACE_SLOTS) {
        if (overflow) *overflow = 1;
        return 0;
    }
    size_t take = n < cap ? n : cap;
    for (size_t i = 0; i < take; i++) {
        size_t k = VKX_GMEM_TRACE_PAGE_WORD + 2u * i;
        out[i] = (uint64_t)w[k] | (uint64_t)w[k + 1] << 32;
    }
    return take;
}

static inline int vkx_gmem_page_overlaps(uint64_t page, uintptr_t lo, uintptr_t hi) {
    uint64_t a = page << VKX_GMEM_PAGE_SHIFT;
    uint64_t b = a + (1u << VKX_GMEM_PAGE_SHIFT);
    return b > a && a < (uint64_t)hi && (uint64_t)lo < b;
}
