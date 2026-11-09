// Linux build: what the replay (vk_replay.c) shares with the draw path (vk_draw.c): its texture images and their lookups
#pragma once
#include "../gfx/nvn/nvn_obj.h"
#include <pthread.h>
#include <vulkan/vulkan.h>
#include "vk_fmt.h"
// Command buffers in flight: a submission is flushed in parts (vkr_flush), so several per frame; the draw path keeps its descriptor pools, set cache and arena per slot (vk_draw.c)
#define CB_RING 16
// The translated NVN pool handles are 20-bit texture IDs and 12-bit sampler
// IDs.  The title only registers a small prefix; keeping a bounded Vulkan
// table avoids consuming the device's entire million-descriptor limit while
// preserving ample headroom for OoT's observed IDs.
#define VKR_BINDLESS_TEXTURES 262144
#define VKR_BINDLESS_SAMPLERS 4096
// The sRGB / linear twin of an 8-bit RGBA or BGRA format (UNDEFINED for any other). Images of these formats are made mutable with both in their format list, so a texture view
// of the other one (TextureViewSetFormat) is honoured: Breath of the Wild writes and reads its picture through sRGB views of RGBA8 textures, and the views in the image's own
// format skipped the sRGB encoding, so the whole picture came out too dark (the Mac's "Render-target views that change the format")
static inline VkFormat vkr_srgb_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// NVN permits a 32-bit colour texel to be addressed as one integer word.  In
// particular, image atomics in Ocarina's NDSR path use an R32UI view of an
// RGBA8 target.  These are members of Vulkan's 32-bit format compatibility
// class; compressed/depth formats and wider colour texels are deliberately
// excluded.
static inline int vkr_word_view_compatible(VkFormat image, VkFormat view) {
    if (view != VK_FORMAT_R32_UINT && view != VK_FORMAT_R32_SINT) return 0;
    switch (image) {
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return 1;
    default:
        return 0;
    }
}
// The format a guest TextureView requests when it is one of the compatible
// formats listed at image creation. Besides the sRGB pairs and compatible
// 32-bit integer word views, Ocarina aliases an R11G11B10 image as RGB9E5 when
// consuming raw words. Returning the base rejects unrelated reinterpretations.
static inline VkFormat vkr_view_format(VkFormat img_fmt, const view_t *v, int has_v) {
    if (!has_v || !v->set_format) return img_fmt;
    VkFormat vf = vkf_from_nvn(v->format, NULL, NULL);
    VkFormat twin = vkr_srgb_twin(img_fmt);
    if (twin != VK_FORMAT_UNDEFINED && vf == twin) return vf;
    if (vkr_word_view_compatible(img_fmt, vf)) return vf;
    if (img_fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && vf == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return vf;
    return img_fmt;
}
typedef struct vkview { uint64_t key; VkImageView view; struct vkview *next; } vkview;   // views of an image, keyed by shape/components and compatible VkFormat
typedef struct vtex {
    uint64_t serial; struct vtex *next;
    VkImage img; VkDeviceMemory mem; VkFormat fmt; VkImageAspectFlags aspect;
    uint64_t bytes;            // VkMemoryRequirements.size of the image allocation
    int speculative;           // created only to populate a bindless ID; charged to that separate strict budget
    uint32_t w, h, d, layers, levels, samples; int target, fresh;   // fresh: still VK_IMAGE_LAYOUT_UNDEFINED
    VkImageUsageFlags usage;
    int gpu_owned;            // written by the GPU (a render target, cleared, a copy destination): its content is the GPU's, never reloaded from guest memory
    int uploaded; uint64_t stamp;   // loaded from guest memory, at this dirty-tracking stamp (nvn_dirty.c)
    uint64_t clean_at, last_sample; const void *span;   // dirty clock / last replay submission sampled; tracked range (nvn_dirty_span)
    uint64_t release_after;   // released: destroyed once the timeline passes this (0 = live)
    vkview *views;
    const uint8_t *kmem, *kend; nvn_texdesc kd; uint32_t khash;
    int refs, serial_refs;   // all references, and the subset represented in vk_replay.c's serial table
    const void *pool;   // the NVN memory pool the storage is on: its Finalize retires the image (mtl_alias_drop_pool)
} vtex;
extern pthread_mutex_t g_tmu;   // the texture table
vtex *vkr_tex_sampled(const tex_t *o, VkCommandBuffer upload_unused);   // under g_tmu: the image, loaded from guest memory first when it must be
vtex *vkr_tex_target(const tex_t *o);                                   // under g_tmu: the image without loading guest content, for an overwrite
vtex *vkr_tex_sampled_budgeted(const tex_t *o, uint64_t total_budget,
                               uint64_t single_max);   // speculative set-5 creation only; existing images are always returned
void vkr_tex_gpu_write(vtex *t);                       // under g_tmu: mark an image as GPU-written
int vkr_treg(int id, tex_t *t, view_t *v, int *has_v);                  // a registered texture ID
int vkr_sreg(int id, samp_par *p);                                       // a registered sampler ID
int vkr_bindless_take_tex_dirty(int *ids, int cap);                    // registration IDs changed since the previous take
int vkr_bindless_take_samp_dirty(int *ids, int cap);
int vkr_bindless_evict_old_speculative(uint64_t registered_before, int *ids,
                                       int cap, uint64_t *bytes);       // retire one bounded batch; returns affected current IDs
void vkr_cpu_written_refresh(void);   // before a submission's first draw: textures the CPU wrote since they were loaded (vk_replay.c)
int vkr_bindless_recent_tex_ids(uint64_t registered_after, int *ids,
                                int cap);                              // current registrations from the new scene generation
uint64_t vkr_submission_count(void);
uint64_t vkr_mesh_issue_count(void);
int vkr_tex_gpu_owned(uint64_t serial);
uint32_t vkr_pick_mem(uint32_t bits, VkMemoryPropertyFlags want);
uint64_t vkr_fence_next(void);   // the fence value the submission in progress will signal
uint64_t vkr_image_bytes(void);  // all live + deferred image allocation bytes (diagnostics)
int vkr_slot(void);              // the command buffer ring slot of the submission in progress
VkCommandBuffer vkr_cmd(void);   // its command buffer
// DM_VK_CS_GPU=1: timestamp one compute dispatch without splitting or waiting
// for its command buffer.  begin returns a slot-local token, or -1 when the
// bounded diagnostic is off/full; end consumes that token after vkCmdDispatch.
int vkr_cs_gpu_begin(VkCommandBuffer c, const char key[17], uint32_t x,
                     uint32_t y, uint32_t z, uint64_t indirect);
void vkr_cs_gpu_end(VkCommandBuffer c, int token);
void vkr_gmem_invalidate_pages(const char *key, const uint64_t *pages, size_t n,
                               int overflow, uint32_t stores);   // completed opt-in raw-global page log
// The NVN2 push-constant block of the graphics stages (0) or compute (1), 256 bytes (DMSH_PUSH_SIZE), as the records so far left it; bound as c[24] (vk_draw.c)
const uint8_t *vkr_push(int compute);
