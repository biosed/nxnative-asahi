#include <stdint.h>
#include <stdio.h>
#include "vk_omap.h"

static int bad;
static void check(int condition, const char *what)
{
    if (!condition) { printf("FAIL: %s\n", what); bad++; }
}

int main(void)
{
    const uint32_t fallback = 0xdeadbeefu;
    check(vkx_json_u32_or("{\"frag_out_map\":0}", "\"frag_out_map\"", fallback) == 0,
          "depth-only zero map is retained");
    check(vkx_json_u32_or("{\"frag_out_map\":15}", "\"frag_out_map\"", fallback) == 0xf,
          "single-target RGBA map is parsed");
    check(vkx_json_u32_or("{\"frag_out_map\":261}", "\"frag_out_map\"", fallback) == 0x105,
          "multiple target nibbles are parsed");
    check(vkx_json_u32_or("{\"frag_out_map\":-1}", "\"frag_out_map\"", fallback) == UINT32_MAX,
          "signed all-components map is reinterpreted as uint32");
    check(vkx_json_u32_or("{\"frag_out_map\":-2147483648}", "\"frag_out_map\"", fallback) == 0x80000000u,
          "signed high-bit map is reinterpreted as uint32");
    check(vkx_json_u32_or("{\"other\":15}", "\"frag_out_map\"", fallback) == fallback,
          "missing reflection field keeps compatibility fallback");
    check(vkx_json_u32_or("{\"frag_out_map\":null}", "\"frag_out_map\"", fallback) == fallback,
          "non-numeric reflection field keeps compatibility fallback");

    check(vkx_fragment_write_mask(0xffffffffu, 0x105u, 0) == 0x5,
          "target zero uses its shader-map nibble");
    check(vkx_fragment_write_mask(0xffffffffu, 0x105u, 2) == 0x1,
          "target two uses its shader-map nibble");
    check(vkx_fragment_write_mask(0xffffffddu, 0x105u, 0) == 0x5,
          "fixed-function and shader component masks combine");
    check(vkx_fragment_write_mask(0xffffffffu, 0, 0) == 0,
          "depth-only shader writes no colour components");
    check(vkx_fragment_write_mask(0xffffffffu, 0xffffffffu, 7) == 0xf,
          "eighth target retains all components");
    check(vkx_fragment_write_mask(0xffffffffu, 0xffffffffu, 8) == 0,
          "out-of-range target fails closed");

    printf("t_vk_omap: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
