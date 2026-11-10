// The registered LINEAR textures the CPU-write refresh looks at (vk/vk_linear_set.h), without a device: every distinct texture is kept however many
//   IDs come before it (the old scan stopped at the first 64 IDs and never saw Ocarina of Time's glyph atlas), a texture registered under several IDs
//   is kept once, and an ID re-registered with another texture drops out.
// Build and run: make -C dm/linux GAME=<id> t_vk_linear
#include <stdio.h>
#include "vk_linear_set.h"
static int bad; static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
static uint64_t reg[200000];   // ID -> serial of the texture it holds (0: none)
static int live(void *ctx, int id, uint64_t serial) { (void)ctx; return id > 0 && id < 200000 && reg[id] == serial; }
int main(void) {
    vkr_lin_set s = {0};
    // 100 LINEAR textures at low IDs, then the atlas far beyond them
    for (int i = 1; i <= 100; i++) { reg[i] = 1000 + i; check(vkr_lin_note(&s, i, 1000 + i), "note"); }
    reg[150000] = 77; check(vkr_lin_note(&s, 150000, 77), "note the atlas");
    int seen = 0; for (size_t i = 0; i < s.n; i++) seen |= s.e[i].serial == 77;
    check(seen && s.n == 101, "a texture after more than 64 others is kept");

    // the atlas under a second ID: still one entry
    reg[150001] = 77; vkr_lin_note(&s, 150001, 77);
    check(s.n == 101, "a texture under two IDs is kept once");
    reg[150000] = 0;   // its first ID reused for nothing: the atlas is still registered under 150001
    check(vkr_lin_prune(&s, live, NULL) == 101, "reusing an older ID keeps a texture still registered");

    // ID 5 re-registered with a non-LINEAR texture: pruned; the others stay, in order
    reg[5] = 9999;
    check(vkr_lin_prune(&s, live, NULL) == 100, "a re-registered ID drops out");
    check(s.e[0].serial == 1001 && s.e[4].serial == 1006, "order of the rest is kept");
    int atlas = 0; for (size_t i = 0; i < s.n; i++) atlas |= s.e[i].serial == 77;
    check(atlas, "the atlas stays");
    free(s.e);
    printf("t_vk_linear: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
