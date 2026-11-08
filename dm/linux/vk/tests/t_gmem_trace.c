// Device-free test of the opt-in raw-global page-log SPIR-V rewrite and the
// completed-log consumer. The synthetic module has one storage buffer and the
// same three-pointer raw-store helper shape emitted by dmsh_xlate.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spv_reflect.h"
#include "vk_gmem_trace.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
static void put(uint32_t *w, size_t *n, const uint32_t *x, size_t nx) { memcpy(w + *n, x, nx * 4); *n += nx; }
#define EMIT(...) do { const uint32_t x_[] = {__VA_ARGS__}; put(w, &n, x_, sizeof x_ / sizeof *x_); } while (0)

static size_t module(uint32_t *w) {
    size_t n = 0; uint32_t h[] = {0x07230203, 0x00010300, 0, 40, 0}; put(w, &n, h, 5);
    EMIT((2u<<16)|17, 1);                                      // Shader
    EMIT((3u<<16)|14, 0, 1);                                  // Logical GLSL450
    EMIT((5u<<16)|15, 5, 30, 0x6e69616d, 0);                  // main
    EMIT((6u<<16)|16, 30, 17, 1, 1, 1);                      // LocalSize
    EMIT((4u<<16)|71, 10, 6, 4);                              // ArrayStride 4
    EMIT((3u<<16)|71, 11, 2);                                 // Block
    EMIT((5u<<16)|72, 11, 0, 35, 0);                         // member Offset 0
    EMIT((4u<<16)|71, 14, 34, 0); EMIT((4u<<16)|71, 14, 33, 500);
    EMIT((2u<<16)|19, 1); EMIT((2u<<16)|20, 2);               // void, bool
    EMIT((4u<<16)|21, 3, 32, 0); EMIT((4u<<16)|21, 4, 32, 1);// u32, i32
    EMIT((3u<<16)|29, 10, 4); EMIT((3u<<16)|30, 11, 10);      // runtime i32[], block
    EMIT((4u<<16)|32, 12, 12, 11); EMIT((4u<<16)|32, 13, 12, 4);
    EMIT((4u<<16)|32, 15, 7, 3);                              // Function u32*
    EMIT((3u<<16)|33, 16, 1); EMIT((6u<<16)|33, 17, 1, 15, 15, 15);
    EMIT((4u<<16)|43, 3, 18, 0); EMIT((4u<<16)|43, 4, 19, 1);
    EMIT((4u<<16)|43, 3, 20, 1); EMIT((4u<<16)|43, 3, 21, 0);// scope, semantics/index
    EMIT((4u<<16)|59, 12, 14, 12);                            // log variable
    EMIT((5u<<16)|54, 1, 22, 0, 17);                         // helper
    EMIT((3u<<16)|55, 15, 23); EMIT((3u<<16)|55, 15, 24); EMIT((3u<<16)|55, 15, 25);
    EMIT((2u<<16)|248, 26);
    EMIT((6u<<16)|65, 13, 27, 14, 21, 21);
    EMIT((7u<<16)|241, 4, 28, 27, 20, 21, 19);               // bitmap AtomicOr
    EMIT((1u<<16)|253); EMIT((1u<<16)|56);
    EMIT((5u<<16)|54, 1, 30, 0, 16); EMIT((2u<<16)|248, 31);
    EMIT((1u<<16)|253); EMIT((1u<<16)|56);
    return n;
}
#undef EMIT

static int op_count(const uint32_t *w, size_t n, unsigned op) {
    int c = 0; for (size_t i = 5; i < n;) { unsigned z = w[i] >> 16; if (!z) break; c += (w[i] & 0xffff) == op; i += z; } return c;
}

int main(void) {
    uint32_t w[256]; size_t n = module(w), no = 0; unsigned patched = 0; char err[160]; spvr_info a, b;
    check(spvr_reflect(w, n, &a, err, sizeof err) == 0 && a.nbind == 1 && a.bind[0].binding == 500 && a.bind[0].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          "synthetic store log reflects as one storage buffer");
    uint32_t *o = spvr_gmem_trace(w, n, 500, &no, &patched);
    check(o && patched == 1 && no > n && o[3] > w[3], "one raw-store bitmap atomic gains the page append");
    check(o && op_count(o, no, 241) == 1 && op_count(o, no, 234) == 1 && op_count(o, no, 229) == 2,
          "original bitmap atomic remains beside count and two page-word atomics");
    check(o && spvr_reflect(o, no, &b, err, sizeof err) == 0 && b.nbind == a.nbind && b.bind[0].binding == a.bind[0].binding && b.bind[0].type == a.bind[0].type,
          "page instrumentation leaves reflection unchanged");
    size_t missn = 99; unsigned missp = 99; uint32_t *miss = spvr_gmem_trace(w, n, 499, &missn, &missp);
    check(!miss && missn == 0 && missp == 0, "wrong binding fails closed"); free(miss);

    uint32_t log[VKX_GMEM_TRACE_WORDS] = {0}; uint64_t pages[4] = {0}; int overflow = 0; uint32_t stores = 0;
    uint64_t p0 = 0x0000000fffd38000ull, p1 = 0x0000000fffc28123ull;
    log[VKX_GMEM_TRACE_COUNT_WORD] = 2;
    log[VKX_GMEM_TRACE_PAGE_WORD] = (uint32_t)p0; log[VKX_GMEM_TRACE_PAGE_WORD + 1] = (uint32_t)(p0 >> 32);
    log[VKX_GMEM_TRACE_PAGE_WORD + 2] = (uint32_t)p1; log[VKX_GMEM_TRACE_PAGE_WORD + 3] = (uint32_t)(p1 >> 32);
    check(vkx_gmem_trace_pages(log, VKX_GMEM_TRACE_WORDS, pages, 4, &overflow, &stores) == 2 && !overflow && stores == 2 && pages[0] == p0 && pages[1] == p1,
          "consumer reconstructs full canonical guest pages");
    uintptr_t a0 = (uintptr_t)(p0 << VKX_GMEM_PAGE_SHIFT);
    check(vkx_gmem_page_overlaps(p0, a0 + 128, a0 + 8192) && !vkx_gmem_page_overlaps(p1, a0 + 128, a0 + 8192),
          "only an image storage overlapping the written page is selected");
    log[VKX_GMEM_TRACE_COUNT_WORD] = VKX_GMEM_TRACE_SLOTS + 1;
    check(vkx_gmem_trace_pages(log, VKX_GMEM_TRACE_WORDS, pages, 4, &overflow, &stores) == 0 && overflow,
          "wrapped page pairs fail closed instead of invalidating a guessed range");
    free(o);
    printf("t_gmem_trace: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
