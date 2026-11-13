// Device-free unit test for the raw-global guest-memory lookup table and its
// 2-MiB chunk hints (vk/vk_gmem_map.h, GlobalToStorage.TryGenerateRawStorageOp).
// Build and run: make -C port/dm/linux GAME=0400486024DEC000 t_vk_gmem_map
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "vk_gmem_map.h"

static int bad;
static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

int main(void) {
    static uint8_t map[DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY];
    const uint64_t chunk_sz = 1ull << DMSH_GMEM_HINT_SHIFT;

    // 1. Two 32-MiB segments in separate regions + one partial 1-MiB tail + one
    //    segment that collides modulo 8192 (16 GiB apart = 8192 * 2 MiB).
    const uint64_t lo[4] = {
        0x1000000000ull,                         // entry 1: 32 MiB (16 full 2-MiB chunks)
        0x1004000000ull + 0x100000ull,           // entry 2: 1 MiB inside a single 2-MiB chunk (no full chunk)
        0x1080000000ull,                         // entry 3: 32 MiB in region 1 (2 GiB past entry 1, no collision)
        0x1400000000ull,                         // entry 4: collides with entry 1 chunk 0 (0x1000000000 + 16 GiB)
    };
    const uint64_t hi[4] = {
        0x1000000000ull + (32ull << 20),
        0x1004000000ull + 0x200000ull,
        0x1080000000ull + (32ull << 20),
        0x1400000000ull + chunk_sz,
    };
    const uint64_t gpu[4] = {
        0x8000000000ull,
        0x8004100000ull,
        0x8080000000ull,
        0x9400000000ull,
    };

    int n = vkx_gmem_map_pack(map, sizeof map, lo, hi, gpu, 4);
    check(n == 4, "four disjoint segments packed");

    // Non-colliding chunks of entry 1 (chunks 1..15) and entry 3 hit the 2-MiB hint.
    uint64_t host = 0;
    uint32_t entry = 0;
    int hinted = 0;
    check(vkx_gmem_map_eval(map, lo[0] + chunk_sz + 0x1234ull, &host, &entry, &hinted) == 1,
          "entry 1 second 2-MiB chunk resolves");
    check(entry == 1 && hinted == 1 && host == gpu[0] + chunk_sz + 0x1234ull,
          "entry 1 second 2-MiB chunk uses hint and translates to GPU address");

    check(vkx_gmem_map_eval(map, lo[2] + 7 * chunk_sz + 0x80ull, &host, &entry, &hinted) == 1,
          "entry 3 chunk resolves");
    check(entry == 3 && hinted == 1 && host == gpu[2] + 7 * chunk_sz + 0x80ull,
          "entry 3 chunk uses hint and translates to GPU address");

    // Entry 4 collides on the first 2-MiB chunk of entry 1: entry 4 takes the hint,
    // while entry 1's first chunk falls back to binary search and still resolves.
    check(vkx_gmem_map_eval(map, lo[3] + 0x40ull, &host, &entry, &hinted) == 1,
          "colliding entry 4 chunk resolves");
    check(entry == 4 && hinted == 1 && host == gpu[3] + 0x40ull,
          "later colliding chunk wins hint slot");

    check(vkx_gmem_map_eval(map, lo[0] + 0x40ull, &host, &entry, &hinted) == 1,
          "earlier colliding chunk in entry 1 still resolves");
    check(entry == 1 && hinted == 0 && host == gpu[0] + 0x40ull,
          "earlier colliding chunk falls back to binary search accurately");

    // Partial 2-MiB chunk (entry 2) has no hint and resolves via binary search.
    check(vkx_gmem_map_eval(map, lo[1] + 0x200ull, &host, &entry, &hinted) == 1,
          "partial 2-MiB segment resolves");
    check(entry == 2 && hinted == 0 && host == gpu[1] + 0x200ull,
          "partial 2-MiB segment uses binary search without false hint");
    check(vkx_gmem_map_eval(map, lo[1] - 0x100ull, &host, &entry, &hinted) == 0,
          "unmapped half of partial 2-MiB chunk is rejected");

    // Unused entries must have end == 0 and hint == 0: in particular chunk 0x7ffff
    // (0xffffe00000, slot 8191) must NOT match a 0xffffffff fill in word 6.
    check(vkx_gmem_map_eval(map, 0xffffe00100ull, &host, &entry, &hinted) == 0 && !hinted,
          "unmapped address in chunk 0x7ffff does not match unused entry 8191");

    // Overlapping segment is clipped to the tail past the previous segment's end
    // while keeping its original GPU delta.
    const uint64_t o_lo[2] = {0x1000000000ull, 0x1000000000ull + chunk_sz};
    const uint64_t o_hi[2] = {0x1000000000ull + 2 * chunk_sz, 0x1000000000ull + 4 * chunk_sz};
    const uint64_t o_gpu[2] = {0x5000000000ull, 0x6000000000ull};
    check(vkx_gmem_map_pack(map, sizeof map, o_lo, o_hi, o_gpu, 2) == 2,
          "overlapping segments pack into two disjoint entries");
    check(vkx_gmem_map_eval(map, o_lo[0] + 3 * chunk_sz + 0x10ull, &host, &entry, &hinted) == 1 &&
          entry == 2 && hinted == 1 && host == o_gpu[1] + 2 * chunk_sz + 0x10ull,
          "clipped overlapping segment preserves original GPU-minus-guest delta");

    // 2. Flat table (DM_VK_GMEM_FLAT=1): two whole regions, entry 0 tagged. The tag must not change what the search
    //    resolves (old shaders ignore it), and the fast path must agree with the search for every address.
    const uint64_t f_lo[2] = {0xfffc00000000ull, 0xfffe80000000ull};
    const uint64_t f_hi[2] = {0xfffc00000000ull + (8ull << 30), 0xfffe80000000ull + (4ull << 30)};
    const uint64_t f_gpu[2] = {0x2000000000ull, 0x4000000000ull};
    check(vkx_gmem_map_eval_flat(map, f_lo[0], &host) == -1, "untagged table is not flat");
    check(vkx_gmem_map_pack(map, sizeof map, f_lo, f_hi, f_gpu, 2) == 2, "two whole regions pack");
    vkx_gmem_map_mark_flat(map, sizeof map, 2);
    const uint64_t probes[] = {f_lo[0], f_lo[0] + 0x12345678ull, f_hi[0] - 4, f_hi[0], f_lo[1] - 1, f_lo[1], f_hi[1] - 1, f_hi[1], f_lo[0] - 1, 0x1000ull};
    for (size_t i = 0; i < sizeof probes / sizeof *probes; i++) {
        uint64_t hs = 0, hf = 0;
        int s = vkx_gmem_map_eval(map, probes[i], &hs, &entry, &hinted), f = vkx_gmem_map_eval_flat(map, probes[i], &hf);
        check(f == s && (!s || hf == hs), "flat fast path agrees with the search");
    }
    check(vkx_gmem_map_eval_flat(map, f_lo[1] + 0x40ull, &host) == 1 && host == f_gpu[1] + 0x40ull, "flat second region translates to its GPU address");
    check(vkx_gmem_map_eval_flat(map, f_hi[0] + 0x40ull, &host) == 0, "gap between regions is refused");
    vkx_gmem_map_mark_flat(map, sizeof map, VKX_GMEM_FLAT_MAX + 1);   // out of range: must leave the existing tag alone
    check(vkx_gmem_map_eval_flat(map, f_lo[0], &host) == 1, "out-of-range count is not written");
    check(vkx_gmem_map_pack(map, sizeof map, f_lo, f_hi, f_gpu, 2) == 2 && vkx_gmem_map_eval_flat(map, f_lo[0], &host) == -1,
          "repacking clears the flat tag");

    printf("t_vk_gmem_map: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
