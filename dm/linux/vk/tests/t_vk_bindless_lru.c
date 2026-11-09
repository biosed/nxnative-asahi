// Device-free unit test for demand-driven bindless feedback harvest and LRU
// speculative image eviction under the 512 MiB cap (vk/vk_bindless_lru.h).
// Build and run: make -C port/dm/linux GAME=0400486024DEC000 t_vk_bindless_lru
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "vk_bindless_lru.h"

static int bad;
static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

int main(void) {
    enum { MAX_IDS = 128, WORDS = MAX_IDS / 32 };
    uint32_t bitmap[WORDS] = {0};
    uint8_t real[MAX_IDS] = {0};
    uint64_t id_last_used[MAX_IDS] = {0};
    int retry[32] = {0};
    unsigned retry_n = 0;

    // ID 5 is already resident (real[5] = 1); ID 12 and ID 65 are unresolved (real = 0).
    // Bit 0 (ID 0) must be ignored.
    real[5] = 1;
    bitmap[0] = (1u << 0) | (1u << 5) | (1u << 12);
    bitmap[2] = (1u << 1); // ID 65

    unsigned fresh = vkx_bindless_feedback_harvest(
        bitmap, WORDS, MAX_IDS, 480ull, real, id_last_used, retry, &retry_n, 32);
    check(fresh == 2 && retry_n == 2, "two unresolved IDs harvested");
    check(retry[0] == 12 && retry[1] == 65, "unresolved IDs 12 and 65 queued in order");
    check(id_last_used[5] == 480ull && id_last_used[12] == 480ull && id_last_used[65] == 480ull,
          "both resident (ID 5) and unresolved (IDs 12, 65) timestamps updated to sub 480");
    check(id_last_used[0] == 0, "ID 0 ignored");
    check(bitmap[0] == 0 && bitmap[2] == 0, "feedback bitmap cleared after harvest");

    // Harvesting ID 12 again at sub 485 updates its timestamp without duplicating retry.
    bitmap[0] = (1u << 12);
    fresh = vkx_bindless_feedback_harvest(
        bitmap, WORDS, MAX_IDS, 485ull, real, id_last_used, retry, &retry_n, 32);
    check(fresh == 0 && retry_n == 2 && id_last_used[12] == 485ull,
          "repeated unresolved ID updates timestamp without duplicate retry entry");

    // Image last-used combines direct tex_sampled stamp and bound bindless IDs.
    const int bound_ids[2] = {3, 5};
    id_last_used[3] = 100ull;
    check(vkx_bindless_image_last_used(50ull, bound_ids, 2, id_last_used, MAX_IDS) == 480ull,
          "image last-used takes max across direct sample and resident bindless feedback");

    // Budget deficit calculation under a 512 MiB cap.
    const uint64_t mb = 1ull << 20;
    uint64_t budget = 512 * mb;
    check(vkx_bindless_lru_deficit(480 * mb, budget, 16 * mb) == 0,
          "no deficit when request fits under budget");
    uint64_t deficit = vkx_bindless_lru_deficit(504 * mb, budget, 24 * mb);
    check(deficit == 16 * mb, "deficit is exact bytes needed to fit 24 MiB into 512 MiB from 504 MiB");

    // Candidate selection:
    // - key 101: title-screen speculative, last used at sub 10, 8 MiB (eligible)
    // - key 102: title-screen speculative, last used at sub 15, 6 MiB (eligible)
    // - key 103: title-screen speculative, last used at sub 15, 12 MiB (eligible, larger tie-break)
    // - key 104: GPU-owned render target, last used at sub 5, 32 MiB (must NOT evict)
    // - key 105: presentation-pinned image, last used at sub 5, 32 MiB (must NOT evict)
    // - key 106: active Kokiri bindless texture refreshed at sub 480, 32 MiB (protected by min_age)
    const vkx_bindless_lru_cand cands[6] = {
        {.image_key = 101, .bytes = 8 * mb,  .last_used_sub = 10,  .speculative = 1},
        {.image_key = 102, .bytes = 6 * mb,  .last_used_sub = 15,  .speculative = 1},
        {.image_key = 103, .bytes = 12 * mb, .last_used_sub = 15,  .speculative = 1},
        {.image_key = 104, .bytes = 32 * mb, .last_used_sub = 5,   .speculative = 1, .gpu_owned = 1},
        {.image_key = 105, .bytes = 32 * mb, .last_used_sub = 5,   .speculative = 1, .pinned = 1},
        {.image_key = 106, .bytes = 32 * mb, .last_used_sub = 480, .speculative = 1},
    };
    uint64_t victims[8] = {0}, reclaimed = 0;
    int nv = vkx_bindless_lru_select(cands, 6, deficit, 500ull, 64ull, victims, 8, &reclaimed);
    check(nv == 2, "two old speculative victims selected to cover 16 MiB deficit");
    check(victims[0] == 101 && victims[1] == 103 && reclaimed == 20 * mb,
          "oldest (101) then larger tie-break at sub 15 (103) selected; gpu_owned/pinned/recent untouched");

    printf("t_vk_bindless_lru: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
