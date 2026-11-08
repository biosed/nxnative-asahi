// Where a part's GPU timestamp goes on the guest clock (vk_replay.c gp_run), as pure arithmetic so it can be checked
//   without a device: the part's stamp (at, in device ticks) is placed against the newest completed part end (ref) and
//   reported as the guest time now - (ref - at) * timestampPeriod.
//   A stamp newer than ref belongs to a submission that reused the query slot after this part was recorded: the
//   difference would be negative and, computed unsigned, wrap to a huge value and hand the game a timestamp of zero.
//   So such a stamp is not used, and the caller reports "now" instead.
//   Pure (no Vulkan), so vk/tests/t_gpstamp.c checks the arithmetic; the caller (vk_replay.c) counts the reasons.
#pragma once
#include <stdint.h>

// Why a stamp was or was not placed on the guest clock.
enum vkx_gp_stamp {
    VKX_GP_STAMP_USED = 0,    // the stamp was placed; *ns holds the guest time
    VKX_GP_STAMP_NO_REF = 1,  // no completed part end to place it against yet; *ns is "now"
    VKX_GP_STAMP_NEWER = 2,   // the stamp is newer than the reference end (a reused query slot); *ns is "now"
};

// now: the guest clock now, in ns.  ref: the newest completed part end, in device ticks (0 = none yet).
// at: this part's stamp, in device ticks.  per: ns per device tick (VkPhysicalDeviceLimits::timestampPeriod).
// *ns is always written: the guest time to report.
static inline enum vkx_gp_stamp vkx_gp_stamp_ns(uint64_t now, uint64_t ref, uint64_t at, double per, uint64_t *ns) {
    if (!ref) { *ns = now; return VKX_GP_STAMP_NO_REF; }   // nothing to place the stamp against yet
    if (at > ref) { *ns = now; return VKX_GP_STAMP_NEWER; }   // a newer stamp: the unsigned difference would wrap
    uint64_t back = (uint64_t)((double)(ref - at) * per);
    *ns = back < now ? now - back : 0;   // a part that began before the guest clock's zero stays zero
    return VKX_GP_STAMP_USED;
}
