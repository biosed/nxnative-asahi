#include "../vk_replay.h"

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

    puts("t_vk_format: passed");
    return 0;
}
