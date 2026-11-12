#include <stdio.h>
#include "vk_draw.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

int main(void) {
    p_draw d = {.base_instance = 7};
    p_draw_idx x = {.base_instance = 3};
    check(vkx_draw_base_instance(&d, NULL) == 7 &&
          vkx_draw_base_instance(NULL, &x) == 3 &&
          vkx_draw_base_instance(NULL, NULL) == 0,
          "array and indexed draw records select their base instance");
    check(vkx_stage_of_name("ts set 3") == VKX_STAGE_TS && vkx_stage_of_name("ms set 4") == VKX_STAGE_MS,
          "TaskEXT and MeshEXT metadata names are recognized");
    check(vkx_stage_set(VKX_STAGE_TS) == 3 && vkx_stage_set(VKX_STAGE_MS) == 4,
          "task/mesh descriptor sets follow remapped host slots");
    check(vkx_program_host_mask(0xc0) == ((1u << 3) | (1u << 4)) &&
          vkx_program_host_mask((1u << 0) | (1u << 1)) == ((1u << 0) | (1u << 1)),
          "raw task/mesh BindProgram bits remap to host slots 3/4");
    check(vkx_stage_c0_ssbo_base(VKX_STAGE_TS) == 0x670 &&
          vkx_stage_c0_ssbo_base(VKX_STAGE_MS) == 0x770,
          "future task/mesh c0 storage-table bases are explicit");
    check(vkx_stage_vk_bit(VKX_STAGE_TS) == VK_SHADER_STAGE_TASK_BIT_EXT &&
          vkx_stage_vk_bit(VKX_STAGE_MS) == VK_SHADER_STAGE_MESH_BIT_EXT,
          "task/mesh map to extension stage bits");
    check(vkx_stage_next(VKX_STAGE_TS) == VK_SHADER_STAGE_MESH_BIT_EXT &&
          vkx_stage_next(VKX_STAGE_MS) == VK_SHADER_STAGE_FRAGMENT_BIT,
          "native task-to-mesh-to-fragment links are explicit");
    check(!vkx_stage_runtime_enabled(VKX_STAGE_TS) && !vkx_stage_runtime_enabled(VKX_STAGE_MS) &&
          vkx_stage_runtime_enabled(VKX_STAGE_CS),
          "runtime remains fail-closed for task/mesh");
    check(vkx_stage_of_name("xx") < 0 && vkx_stage_set(99) < 0 && !vkx_stage_vk_bit(99),
          "unknown metadata fails closed");
    printf("t_vk_stage: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
