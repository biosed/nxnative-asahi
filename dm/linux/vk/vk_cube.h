// Which images may be viewed as cubes. Device-free; vk_replay.c creates the images, vk_draw.c's view_get makes the views.
// NVN lets a texture view retarget storage: a 2D or 2D-array texture of square layers can be sampled as a cube or cube array (TextureView SetTarget), and
// a shader's cube binding decides the view type (view_get). Vulkan allows a cube view only of an image created VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
// which needs a 2D image with square layers, at least six of them, and one sample; only whole cubes are taken (a multiple of six layers). Every image that meets those rules is created cube-compatible, so
// such a view is valid; a cube view of any other image is refused, and the texture samples nothing rather than reaching the driver invalid.
#ifndef VK_CUBE_H
#define VK_CUBE_H
#include <stdint.h>

// Whether an image of these dimensions can be created VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT (three_d: a 3D image)
static inline int vkx_cube_compatible(int three_d, uint32_t w, uint32_t h, uint32_t layers, uint32_t samples) {
    return !three_d && w && w == h && layers >= 6 && layers % 6 == 0 && samples == 1;
}

// Whether a cube or cube-array view may be made of an image of layers layers created with cube_ok: whole cubes only
static inline int vkx_cube_view_ok(int cube_ok, uint32_t layers) {
    return cube_ok && layers >= 6 && layers % 6 == 0;
}

#endif
