// The game's own upscaler network (agl's NDSR, DM_GAME_UPSCALER_KEYS from the shader map) or a plain scaler in its place: the player's choice,
// DM_UPSCALER=game (default) | linear | fsr1 | probe. Device-free decisions only; vk_draw.c does the Vulkan work (upscale_dispatch).
//   game    the network runs, as before.
//   linear  a network stage that reads one texture and writes one image, from its push-constant input rectangle to a strictly larger output
//           rectangle, in the encoding agl's NDSR uses (sqrt(linear) RGB9E5 in, linear B10G11R11 out), is replaced by a bilinear scale between those
//           rectangles (vk/shaders/upscale_sqrt.comp); anything else, or anything the device refuses, runs the network.
//   fsr1    as linear, but AMD FSR 1 (EASU edge-adaptive upscale, then RCAS sharpening; dm/third_party/fidelityfx-fsr1, MIT) in the
//           network's sqrt space; both rectangles must start at the texture origin.
//   probe   the network runs; the first dispatches of each upscaling stage log their textures, formats and push-constant rectangles.
// The look changes (a plain filter instead of the game's network), so it is never the default.
#ifndef VK_UPSCALER_H
#define VK_UPSCALER_H
#include <stdint.h>
#include <string.h>

enum { VKX_UPSCALER_GAME = 0, VKX_UPSCALER_LINEAR = 1, VKX_UPSCALER_PROBE = 2, VKX_UPSCALER_FSR1 = 3 };

// The mode of a DM_UPSCALER value; *bad is set for a value that is none of them (the network is kept)
static inline int vkx_upscaler_parse(const char *e, int *bad) {
    *bad = 0;
    if (!e || !*e || !strcmp(e, "game")) return VKX_UPSCALER_GAME;
    if (!strcmp(e, "linear")) return VKX_UPSCALER_LINEAR;
    if (!strcmp(e, "probe")) return VKX_UPSCALER_PROBE;
    if (!strcmp(e, "fsr1")) return VKX_UPSCALER_FSR1;
    *bad = 1; return VKX_UPSCALER_GAME;
}

// key (16 hex digits) is in a comma-separated key list
static inline int vkx_upscaler_named(const char *list, const char *key) {
    for (const char *p = list; p && *p;) {
        if (!strncmp(p, key, 16) && (p[16] == ',' || !p[16])) return 1;
        p = strchr(p, ','); if (p) p++;
    }
    return 0;
}

typedef struct { uint32_t x, y, w, h; } vkx_rect;

// A push-constant rectangle (origin x, y, then the last texel relative to the origin, inclusive) inside a w x h texture
static inline int vkx_upscaler_rect(const uint32_t v[4], uint32_t w, uint32_t h, vkx_rect *r) {
    uint64_t rw = (uint64_t)v[2] + 1, rh = (uint64_t)v[3] + 1;
    if (!w || !h || (uint64_t)v[0] + rw > w || (uint64_t)v[1] + rh > h) return 0;
    *r = (vkx_rect){v[0], v[1], (uint32_t)rw, (uint32_t)rh};
    return 1;
}

// A stage to replace: the output rectangle is at least the input in both directions and larger in one (an upscale, not a copy or a reduction)
static inline int vkx_upscaler_accept(vkx_rect in, vkx_rect out) {
    return out.w >= in.w && out.h >= in.h && (out.w > in.w || out.h > in.h);
}

// FSR 1's EASU takes a top-left aligned input viewport and writes a whole output image: both rectangles must start at the texture origin
static inline int vkx_upscaler_fsr_rects(vkx_rect in, vkx_rect out) {
    return !in.x && !in.y && !out.x && !out.y && vkx_upscaler_accept(in, out);
}

#endif
