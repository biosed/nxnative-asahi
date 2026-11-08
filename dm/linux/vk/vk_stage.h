// Shader metadata stages understood by the Linux Vulkan loader. Task/mesh
// metadata is accepted so offline output can be audited. The replay owns six
// state slots; native task/mesh keep their actual stage identity while using
// the remapped host slots below, and remain runtime-gated.
#pragma once
#include <string.h>
#include <vulkan/vulkan.h>

enum {
    VKX_STAGE_VS, VKX_STAGE_PS, VKX_STAGE_GS, VKX_STAGE_HS,
    VKX_STAGE_DS, VKX_STAGE_CS, VKX_STAGE_TS, VKX_STAGE_MS,
    VKX_META_STAGE_COUNT
};
enum { VKX_STATE_STAGE_COUNT = 6 };

static inline int vkx_stage_of_name(const char *s) {
    static const char *const names[VKX_META_STAGE_COUNT] = {"vs", "ps", "gs", "hs", "ds", "cs", "ts", "ms"};
    if (!s) return -1;
    for (int i = 0; i < VKX_META_STAGE_COUNT; i++) if (!strncmp(s, names[i], 2)) return i;
    return -1;
}

// hle_nvn remaps guest task/mesh state 6/7 into host slots 3/4.  Sets follow
// those host slots so stage_set(set, shader) reads the state actually bound.
static inline int vkx_stage_set(int st) {
    static const signed char sets[VKX_META_STAGE_COUNT] = {0, 1, 2, 3, 4, 0, 3, 4};
    return st >= 0 && st < VKX_META_STAGE_COUNT ? sets[st] : -1;
}

// BindProgram carries raw NVN stage bits even though resource binds were
// already remapped by hle_nvn.  Collapse task/mesh bits into the same slots.
static inline unsigned vkx_program_host_mask(unsigned stages) {
    unsigned host = stages & ((1u << VKX_STATE_STAGE_COUNT) - 1u);
    if (stages & (1u << VKX_STAGE_TS)) host |= 1u << 3;
    if (stages & (1u << VKX_STAGE_MS)) host |= 1u << 4;
    return host;
}

// Native shader driver-table bases, selected by actual shader stage even when
// resources come from remapped host slot 3/4.
static inline int vkx_stage_c0_ssbo_base(int st) {
    static const short base[VKX_META_STAGE_COUNT] = {
        0x110, 0x510, 0x410, -1, -1, 0x310, 0x670, 0x770
    };
    return st >= 0 && st < VKX_META_STAGE_COUNT ? base[st] : -1;
}

static inline VkShaderStageFlagBits vkx_stage_vk_bit(int st) {
    static const VkShaderStageFlagBits bits[VKX_META_STAGE_COUNT] = {
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT
    };
    return st >= 0 && st < VKX_META_STAGE_COUNT ? bits[st] : 0;
}

static inline VkShaderStageFlags vkx_stage_next(int st) {
    switch (st) {
    case VKX_STAGE_VS: return VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case VKX_STAGE_GS: case VKX_STAGE_DS: case VKX_STAGE_MS: return VK_SHADER_STAGE_FRAGMENT_BIT;
    case VKX_STAGE_HS: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case VKX_STAGE_TS: return VK_SHADER_STAGE_MESH_BIT_EXT;
    default: return 0;
    }
}

// Fail closed until vk_dev enables VK_EXT_mesh_shader and vk_replay records
// vkCmdDrawMeshTasksEXT/IndirectEXT. Recognizing metadata is not permission to
// create a shader object or issue a command.
static inline int vkx_stage_runtime_enabled(int st) { return st >= 0 && st < VKX_STATE_STAGE_COUNT; }
static inline int vkx_stage_runtime_enabled_on(int st, int mesh_runtime) {
    return vkx_stage_runtime_enabled(st) || (mesh_runtime && (st == VKX_STAGE_TS || st == VKX_STAGE_MS));
}
