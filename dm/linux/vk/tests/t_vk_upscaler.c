#include <stdio.h>
#include "vk_upscaler.h"

static int bad;

static void check(int condition, const char *what)
{
    if (!condition) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

int main(void)
{
    int b = 0;
    check(vkx_upscaler_parse(NULL, &b) == VKX_UPSCALER_GAME && !b, "unset keeps the game's network");
    check(vkx_upscaler_parse("", &b) == VKX_UPSCALER_GAME && !b, "empty keeps the game's network");
    check(vkx_upscaler_parse("game", &b) == VKX_UPSCALER_GAME && !b, "game is the network");
    check(vkx_upscaler_parse("linear", &b) == VKX_UPSCALER_LINEAR && !b, "linear selects the plain scaler");
    check(vkx_upscaler_parse("probe", &b) == VKX_UPSCALER_PROBE && !b, "probe logs and keeps the network");
    check(vkx_upscaler_parse("fsr1", &b) == VKX_UPSCALER_FSR1 && !b, "fsr1 selects the FSR 1 scaler");
    check(vkx_upscaler_parse("metalfx", &b) == VKX_UPSCALER_GAME && b, "an unknown value keeps the network and is reported");

    const char *list = "0253bd17deceee79,a335c1db6270c355,d99fbeeb53d38847";
    check(vkx_upscaler_named(list, "0253bd17deceee79"), "the first key is named");
    check(vkx_upscaler_named(list, "d99fbeeb53d38847"), "the last key is named");
    check(!vkx_upscaler_named(list, "d99fbeeb53d38846"), "a key differing in its last digit is not named");
    check(!vkx_upscaler_named(list, "a335c1db"), "a prefix is not a key");
    check(!vkx_upscaler_named("", "d99fbeeb53d38847"), "an empty list names nothing");

    vkx_rect in, out;
    const uint32_t r_in[4] = {0, 0, 682, 383};   /* 683x384 in a 1536x864 texture */
    check(vkx_upscaler_rect(r_in, 1536, 864, &in) && in.w == 683 && in.h == 384 && !in.x && !in.y, "an inclusive last texel gives the size");
    const uint32_t r_out[4] = {0, 0, 2047, 1151};
    check(vkx_upscaler_rect(r_out, 2048, 1152, &out) && out.w == 2048 && out.h == 1152, "a whole-texture rectangle fits exactly");
    const uint32_t r_over[4] = {1, 0, 2047, 1151};
    check(!vkx_upscaler_rect(r_over, 2048, 1152, &out), "a rectangle one texel past the edge is refused");
    const uint32_t r_huge[4] = {0xffffffffu, 0, 0xffffffffu, 0};
    check(!vkx_upscaler_rect(r_huge, 2048, 1152, &out), "an overflowing rectangle is refused");
    check(!vkx_upscaler_rect(r_in, 0, 0, &out), "an empty texture holds no rectangle");

    check(vkx_upscaler_accept((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 2048, 1152}), "a 3x upscale is replaced");
    check(vkx_upscaler_accept((vkx_rect){0, 0, 1024, 576}, (vkx_rect){0, 0, 2048, 576}), "a one-direction upscale is replaced");
    check(!vkx_upscaler_accept((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 683, 384}), "an equal-size stage keeps the network");
    check(!vkx_upscaler_accept((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 341, 192}), "a reduction keeps the network");
    check(!vkx_upscaler_accept((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 2048, 300}), "shrinking in one direction keeps the network");

    check(vkx_upscaler_fsr_rects((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 2048, 1152}), "FSR takes an origin-aligned upscale");
    check(!vkx_upscaler_fsr_rects((vkx_rect){8, 0, 683, 384}, (vkx_rect){0, 0, 2048, 1152}), "FSR refuses an offset input viewport");
    check(!vkx_upscaler_fsr_rects((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 4, 2048, 1148}), "FSR refuses an offset output rectangle");
    check(!vkx_upscaler_fsr_rects((vkx_rect){0, 0, 683, 384}, (vkx_rect){0, 0, 683, 384}), "FSR refuses a copy");

    printf("t_vk_upscaler: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
