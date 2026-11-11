// Compose NVN format, texture-builder and texture-view component mappings for
// a Vulkan sampled image view.  Kept device-free so the ordering is testable.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

static inline VkComponentMapping vkx_swizzle_explicit(VkComponentMapping m) {
    if (m.r == VK_COMPONENT_SWIZZLE_IDENTITY) m.r = VK_COMPONENT_SWIZZLE_R;
    if (m.g == VK_COMPONENT_SWIZZLE_IDENTITY) m.g = VK_COMPONENT_SWIZZLE_G;
    if (m.b == VK_COMPONENT_SWIZZLE_IDENTITY) m.b = VK_COMPONENT_SWIZZLE_B;
    if (m.a == VK_COMPONENT_SWIZZLE_IDENTITY) m.a = VK_COMPONENT_SWIZZLE_A;
    return m;
}

static inline VkComponentSwizzle vkx_swizzle_pick(unsigned selector,
                                                   const VkComponentSwizzle base[4]) {
    switch (selector) {
    case 0: return VK_COMPONENT_SWIZZLE_ZERO;
    case 1: return VK_COMPONENT_SWIZZLE_ONE;
    case 2: case 3: case 4: case 5: return base[selector - 2];
    default: return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
}

static inline VkComponentMapping vkx_swizzle_compose(VkComponentMapping format,
                                                      int texture_set, uint16_t texture,
                                                      int has_view, int view_set,
                                                      const uint8_t view[4]) {
    format = vkx_swizzle_explicit(format);
    VkComponentSwizzle c[4] = {format.r, format.g, format.b, format.a};
    if (texture_set) {
        VkComponentSwizzle n[4];
        for (int i = 0; i < 4; i++) n[i] = vkx_swizzle_pick((texture >> (3 * i)) & 7u, c);
        for (int i = 0; i < 4; i++) c[i] = n[i];
    }
    if (has_view && view_set) {
        VkComponentSwizzle n[4];
        for (int i = 0; i < 4; i++) n[i] = vkx_swizzle_pick(view[i], c);
        for (int i = 0; i < 4; i++) c[i] = n[i];
    }
    return (VkComponentMapping){c[0], c[1], c[2], c[3]};
}

static inline const char *vkx_swizzle_name(VkComponentSwizzle swizzle) {
    switch (swizzle) {
    case VK_COMPONENT_SWIZZLE_IDENTITY: return "I";
    case VK_COMPONENT_SWIZZLE_ZERO: return "0";
    case VK_COMPONENT_SWIZZLE_ONE: return "1";
    case VK_COMPONENT_SWIZZLE_R: return "R";
    case VK_COMPONENT_SWIZZLE_G: return "G";
    case VK_COMPONENT_SWIZZLE_B: return "B";
    case VK_COMPONENT_SWIZZLE_A: return "A";
    default: return "?";
    }
}

static inline const char *vkx_nvn_swizzle_name(unsigned swizzle) {
    static const char *const names[8] = {"0", "1", "R", "G", "B", "A", "?", "?"};
    return names[swizzle & 7u];
}
