// The registered LINEAR textures the CPU-write refresh looks at (vk/vk_linear_set.h), without a device: IDs past the old 64-entry cap remain tracked,
// aliases deduplicate only at snapshot time, and reusing either alias cannot drop a texture that is still live under another ID.
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

    // The atlas under a second ID: both registrations stay live, but a refresh
    // snapshot selects the serial only once.
    reg[150001] = 77; vkr_lin_note(&s, 150001, 77);
    size_t unique = 0; for (size_t i = 0; i < s.n; i++) unique += vkr_lin_first_serial(&s, i);
    check(s.n == 102 && unique == 101, "a texture under two IDs refreshes once");

    // Reusing the newer alias must fall back to the older live registration.
    reg[150001] = 9999;
    check(vkr_lin_prune(&s, live, NULL) == 101, "reusing a newer alias keeps the older live registration");
    int atlas = 0; for (size_t i = 0; i < s.n; i++) atlas |= s.e[i].serial == 77;
    check(atlas, "the older atlas alias stays");

    // And the inverse: restore the newer alias, then reuse the older one.
    reg[150001] = 77; check(vkr_lin_note(&s, 150001, 77), "restore atlas alias");
    reg[150000] = 0;
    check(vkr_lin_prune(&s, live, NULL) == 101, "reusing an older ID keeps a texture still registered");

    // ID 5 re-registered with a non-LINEAR texture: pruned; the others stay, in order
    reg[5] = 9999;
    check(vkr_lin_prune(&s, live, NULL) == 100, "a re-registered ID drops out");
    check(s.e[0].serial == 1001 && s.e[4].serial == 1006, "order of the rest is kept");
    atlas = 0; for (size_t i = 0; i < s.n; i++) atlas |= s.e[i].serial == 77;
    check(atlas, "the atlas stays");
    free(s.e);
    printf("t_vk_linear: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
