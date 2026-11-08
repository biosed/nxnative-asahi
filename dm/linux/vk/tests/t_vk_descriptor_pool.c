#include <stdio.h>
#include "vk_draw.h"

static int bad;
static void check(int c, const char *what) {
    if (!c) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    check(vkx_pool_alloc_result(VK_SUCCESS, 0) == VKX_POOL_ALLOC_OK,
          "success from a fresh pool is accepted");
    check(vkx_pool_alloc_result(VK_SUCCESS, 255) == VKX_POOL_ALLOC_OK,
          "success from a used pool is accepted");
    check(vkx_pool_alloc_result(VK_ERROR_OUT_OF_POOL_MEMORY, 1) == VKX_POOL_ALLOC_ADVANCE,
          "used out-of-pool advances");
    check(vkx_pool_alloc_result(VK_ERROR_FRAGMENTED_POOL, 17) == VKX_POOL_ALLOC_ADVANCE,
          "used fragmented pool advances");
    check(vkx_pool_alloc_result(VK_ERROR_OUT_OF_POOL_MEMORY, 0) == VKX_POOL_ALLOC_FAIL,
          "fresh out-of-pool fails closed");
    check(vkx_pool_alloc_result(VK_ERROR_FRAGMENTED_POOL, 0) == VKX_POOL_ALLOC_FAIL,
          "fresh fragmented pool fails closed");
    check(vkx_pool_alloc_result(VK_ERROR_DEVICE_LOST, 42) == VKX_POOL_ALLOC_FAIL,
          "device loss never advances");
    check(vkx_pool_alloc_result(VK_ERROR_INITIALIZATION_FAILED, 42) == VKX_POOL_ALLOC_FAIL,
          "non-capacity error never advances");
    check(!vkx_descriptor_pressure_values(VKX_DESCRIPTOR_SET_HIGH_WATER - 1,
                                          VKX_DESCRIPTOR_POOL_HIGH_WATER - 1),
          "descriptor pressure stays clear below both submission high-waters");
    check(vkx_descriptor_pressure_values(VKX_DESCRIPTOR_SET_HIGH_WATER, 0),
          "descriptor pressure trips exactly at the set high-water");
    check(vkx_descriptor_pressure_values(0, VKX_DESCRIPTOR_POOL_HIGH_WATER),
          "descriptor pressure trips exactly at the physical-pool high-water");
    check(vkx_descriptor_pressure_values(UINT32_MAX, UINT32_MAX),
          "descriptor pressure remains tripped above both high-waters");
    printf("t_vk_descriptor_pool: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
