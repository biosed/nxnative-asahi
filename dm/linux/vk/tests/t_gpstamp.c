// Where a part's GPU timestamp goes on the guest clock (vk/vk_gpstamp.h), without a device: a stamp newer than the
//   reference end is refused (the unsigned difference would wrap and hand the game a timestamp of zero), a stamp at or
//   before it is placed on the guest clock, and the placement is monotone, never in the future and never negative.
// Build and run: make -C dm/linux GAME=<id> t_gpstamp
#include <stdio.h>
#include "vk_gpstamp.h"
static int bad; static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
int main(void) {
    const uint64_t now = 1000000000ull;   // 1 s on the guest clock
    const double per = 1.0;               // 1 ns per device tick
    uint64_t ns = 0;

    // no completed part end yet: nothing to place the stamp against, so "now"
    check(vkx_gp_stamp_ns(now, 0, 0, per, &ns) == VKX_GP_STAMP_NO_REF && ns == now, "no reference: the stamp is not used and now is reported");

    // a stamp at the reference end: the part ended now
    check(vkx_gp_stamp_ns(now, 500, 500, per, &ns) == VKX_GP_STAMP_USED && ns == now, "stamp at the reference end: now");

    // a stamp before the reference end: the part began (ref - at) ticks ago
    check(vkx_gp_stamp_ns(now, 500, 400, per, &ns) == VKX_GP_STAMP_USED && ns == now - 100, "stamp before the reference end: now minus the span");

    // a stamp newer than the reference end: the difference would wrap; now is reported, not zero
    check(vkx_gp_stamp_ns(now, 500, 501, per, &ns) == VKX_GP_STAMP_NEWER && ns == now, "a newer stamp: now, not a wrapped value");

    // the wrap itself, with a large reference: the unsigned difference would be enormous
    check(vkx_gp_stamp_ns(now, UINT64_MAX / 2, UINT64_MAX / 2 + 1, per, &ns) == VKX_GP_STAMP_NEWER && ns == now, "a large reference: the difference does not wrap");

    // a span longer than the guest clock: the part began before the clock's zero, so it stays zero
    check(vkx_gp_stamp_ns(now, 2000000000ull, 0, per, &ns) == VKX_GP_STAMP_USED && ns == 0, "a span longer than the guest clock: zero");

    // monotone in the stamp: a later stamp is never placed earlier
    int mono = 1; uint64_t prev = 0;
    for (uint64_t at = 0; at <= 500; at += 50) { vkx_gp_stamp_ns(now, 500, at, per, &ns); if (ns < prev) mono = 0; prev = ns; }
    check(mono, "the placement is monotone in the stamp");

    // never in the future, over the whole stamp range including the refused ones
    int range = 1;
    for (uint64_t at = 0; at <= 600; at += 50) { vkx_gp_stamp_ns(now, 500, at, per, &ns); if (ns > now) range = 0; }
    check(range, "the placement is never in the future");

    // the counter reasons are distinct, so a drop can be told apart in a heartbeat
    check(VKX_GP_STAMP_USED != VKX_GP_STAMP_NO_REF && VKX_GP_STAMP_NO_REF != VKX_GP_STAMP_NEWER && VKX_GP_STAMP_USED != VKX_GP_STAMP_NEWER, "the three reasons are distinct");

    printf("t_gpstamp: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
