#include <stdio.h>
#include "vk_draw.h"

static int bad;
static void check(int c, const char *what) {
    if (!c) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    uint16_t used = 0, index = UINT16_MAX;
    for (uint16_t i = 0; i < VKX_DESCRIPTOR_POOL_SETS; i++) {
        check(vkx_descriptor_set_take(&used, &index) && index == i,
              "each persistent descriptor set is taken once in order");
    }
    check(!vkx_descriptor_set_take(&used, &index) &&
          used == VKX_DESCRIPTOR_POOL_SETS && index == VKX_DESCRIPTOR_POOL_SETS - 1,
          "a full persistent pool refuses another set without wrapping");
    used = 0;
    check(vkx_descriptor_set_take(&used, &index) && index == 0 && used == 1,
          "a completed ring slot reuses its first persistent set");
    check(!vkx_descriptor_pressure_values(VKX_DESCRIPTOR_SET_HIGH_WATER - 1,
                                          VKX_DESCRIPTOR_POOL_HIGH_WATER - 1),
          "descriptor pressure stays clear below both submission high-waters");
    check(vkx_descriptor_pressure_values(VKX_DESCRIPTOR_SET_HIGH_WATER, 0),
          "descriptor pressure trips exactly at the set high-water");
    check(vkx_descriptor_pressure_values(0, VKX_DESCRIPTOR_POOL_HIGH_WATER),
          "descriptor pressure trips exactly at the physical-pool high-water");
    check(vkx_descriptor_pressure_values(UINT32_MAX, UINT32_MAX),
          "descriptor pressure remains tripped above both high-waters");

    vkx_descriptor_union u = {0};
    check(vkx_descriptor_union_add(&u, VKX_STAGE_VS, 64, 't', 1) == 1,
          "vertex binding enters the vertex layout domain");
    check(vkx_descriptor_union_add(&u, VKX_STAGE_PS, 64, 't', 8192) == 1,
          "same canonical binding can have a larger pixel-stage count");
    check(vkx_descriptor_union_add(&u, VKX_STAGE_CS, 64, 't', 4) == 1,
          "compute set zero has its own layout domain");
    check(vkx_descriptor_union_add(&u, VKX_STAGE_TS, 200, 'u', 2) == 1 &&
          vkx_descriptor_union_add(&u, VKX_STAGE_HS, 201, 's', 3) == 1,
          "task and hull shaders share graphics set three without sharing bindings");
    uint32_t total[VKX_DESCRIPTOR_TYPE_COUNT];
    vkx_descriptor_union_totals(&u, vkx_descriptor_layout_domain(VKX_STAGE_VS), total);
    check(total[2] == 1, "pixel mega-array capacity does not contaminate vertex sets");
    vkx_descriptor_union_totals(&u, vkx_descriptor_layout_domain(VKX_STAGE_PS), total);
    check(total[2] == 8192, "pixel layout retains its required dynamic array capacity");
    vkx_descriptor_union_totals(&u, vkx_descriptor_layout_domain(VKX_STAGE_CS), total);
    check(total[2] == 4, "graphics layouts do not contaminate compute set zero");
    vkx_descriptor_union_totals(&u, vkx_descriptor_layout_domain(VKX_STAGE_TS), total);
    check(total[0] == 2 && total[1] == 3,
          "task/hull set-domain union accounts for both descriptor classes");
    check(vkx_descriptor_union_add(&u, VKX_STAGE_PS, 64, 'u', 1) == 0,
          "a descriptor-type conflict remains loud within one stage domain");
    check(vkx_descriptor_union_add(&u, VKX_STAGE_PS,
                                   VKX_DESCRIPTOR_BINDING_LIMIT, 't', 1) == -1,
          "out-of-range canonical bindings are rejected");

    vkx_bindless_handle_ids ids = vkx_bindless_ids((7u << 20) | 19u,
                                                    1000, 200, 262144, 4096);
    check(ids.texture == 1019 && ids.sampler == 207,
          "combined handles use the same texture/sampler pool-base rule");
    ids = vkx_bindless_ids(0, 1000, 200, 262144, 4096);
    check(!ids.texture && !ids.sampler, "zero IDs remain null after rebasing");
    ids = vkx_bindless_ids((4095u << 20) | 0xfffffu,
                           262143, 4095, 262144, 4096);
    check(!ids.texture && !ids.sampler,
          "rebased IDs outside the persistent tables fail closed");
    printf("t_vk_descriptor_pool: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
