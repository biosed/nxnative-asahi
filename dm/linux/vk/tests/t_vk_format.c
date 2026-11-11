#include "../vk_replay.h"
#include "../vk_swizzle.h"

#include <assert.h>
#include <stdio.h>

static view_t format_view(uint16_t format) {
    view_t v = {0};
    v.set_format = 1;
    v.format = format;
    return v;
}

int main(void) {
    view_t r32u = format_view(NVN_FORMAT_R32UI);
    view_t r32i = format_view(NVN_FORMAT_R32I);
    view_t rgb9e5 = format_view(NVN_FORMAT_RGB9E5F);

    assert(vkr_word_view_compatible(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT));
    assert(vkr_word_view_compatible(VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SINT));
    assert(!vkr_word_view_compatible(VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT));
    assert(!vkr_word_view_compatible(VK_FORMAT_D32_SFLOAT, VK_FORMAT_R32_UINT));

    assert(vkr_view_format(VK_FORMAT_R8G8B8A8_UNORM, &r32u, 1) == VK_FORMAT_R32_UINT);
    assert(vkr_view_format(VK_FORMAT_R8G8B8A8_UNORM, &r32i, 1) == VK_FORMAT_R32_SINT);
    assert(vkr_view_format(VK_FORMAT_B10G11R11_UFLOAT_PACK32, &rgb9e5, 1) == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
    assert(vkr_view_format(VK_FORMAT_R16G16B16A16_SFLOAT, &r32u, 1) == VK_FORMAT_R16G16B16A16_SFLOAT);
    assert(vkr_view_format(VK_FORMAT_R8G8B8A8_UNORM, &r32u, 0) == VK_FORMAT_R8G8B8A8_UNORM);

    VkComponentMapping r8;
    assert(vkf_from_nvn(NVN_FORMAT_R8, NULL, &r8) == VK_FORMAT_R8_UNORM);
    VkComponentMapping plain = vkx_swizzle_compose(r8, 0, 0, 0, 0, NULL);
    // These are view selectors.  The R8 format itself supplies zero for G/B
    // and one for A when its otherwise-identity RGBA view is sampled.
    assert(plain.r == VK_COMPONENT_SWIZZLE_R && plain.g == VK_COMPONENT_SWIZZLE_G &&
           plain.b == VK_COMPONENT_SWIZZLE_B && plain.a == VK_COMPONENT_SWIZZLE_A);

    // A common one-channel glyph view: RGB are one and alpha reads R.
    uint16_t glyph = 1u | 1u << 3 | 1u << 6 | 2u << 9;
    VkComponentMapping font = vkx_swizzle_compose(r8, 1, glyph, 0, 0, NULL);
    assert(font.r == VK_COMPONENT_SWIZZLE_ONE && font.g == VK_COMPONENT_SWIZZLE_ONE &&
           font.b == VK_COMPONENT_SWIZZLE_ONE && font.a == VK_COMPONENT_SWIZZLE_R);

    // A view composes over the builder result rather than replacing it.
    const uint8_t bgra[4] = {4, 3, 2, 5};
    VkComponentMapping viewed = vkx_swizzle_compose(r8, 1, glyph, 1, 1, bgra);
    assert(viewed.r == VK_COMPONENT_SWIZZLE_ONE && viewed.g == VK_COMPONENT_SWIZZLE_ONE &&
           viewed.b == VK_COMPONENT_SWIZZLE_ONE && viewed.a == VK_COMPONENT_SWIZZLE_R);

    puts("t_vk_format: passed");
    return 0;
}
