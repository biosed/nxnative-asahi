// Linux build: the draw path of the Vulkan replay (vk_draw.c), fed by the replay's walker (vk_replay.c)
#pragma once
#include "../gfx/nvn/nvn_obj.h"
#include "vk_mesh.h"
#include <vulkan/vulkan.h>

// Keep descriptor-pool retry policy device-free and independently testable.
// A capacity error is evidence that a pool is exhausted only after that pool
// has successfully served at least one set.  Treating the same error from a
// fresh pool as "full" would otherwise create and burn the whole pool chain.
typedef enum {
    VKX_POOL_ALLOC_FAIL = 0,
    VKX_POOL_ALLOC_OK,
    VKX_POOL_ALLOC_ADVANCE,
} vkx_pool_alloc_action;
static inline vkx_pool_alloc_action vkx_pool_alloc_result(VkResult r, uint32_t used) {
    if (r == VK_SUCCESS) return VKX_POOL_ALLOC_OK;
    if (used && (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL))
        return VKX_POOL_ALLOC_ADVANCE;
    return VKX_POOL_ALLOC_FAIL;
}
enum {
    VKX_DESCRIPTOR_SET_HIGH_WATER = 8192,
    VKX_DESCRIPTOR_POOL_HIGH_WATER = 192,
};
static inline int vkx_descriptor_pressure_values(uint32_t sets, uint32_t pool) {
    return sets >= VKX_DESCRIPTOR_SET_HIGH_WATER ||
           pool >= VKX_DESCRIPTOR_POOL_HIGH_WATER;
}

void vkx_sub_begin(int slot, VkCommandBuffer c, uint64_t done);   // begins in this ring slot; done is the completed timeline value
int vkx_descriptor_pressure(void);                // current submission reached its bounded descriptor-set high-water
int vkx_prepare_draw(VkCommandBuffer c);          // the program's shaders and descriptor sets for the draw now, bound in c; 0 = it cannot be drawn
typedef struct {
    vkx_mesh_command_plan command;
    VkBuffer indirect_buffer;
    VkDeviceSize indirect_offset;
    uint64_t guest_indirect;
    char task_key[17], mesh_key[17], frag_key[17];
} vkx_mesh_submit;
int vkx_mesh_bridge(VkCommandBuffer c, const p_draw *d, vkx_mesh_submit *out); // outside rendering: submit-owned first/indirect storage + copies/barriers
void vkx_mesh_abort(void); // clear a pending binding-512 owner lease after any post-bridge failure
int vkx_prepare_mesh_program(VkCommandBuffer c, vkx_mesh_program_plan *plan); // proof-gated task(optional)+mesh+fragment preparation
int vkx_draw_mesh(VkCommandBuffer c, const vkx_mesh_submit *s, int ncolor, uint32_t rt_w, uint32_t rt_h, int samples); // inside rendering; proof-gated command issue
void vkx_mesh_draw_telemetry(const p_draw *d); // DM_VK_MESH_LOG: one direct + one indirect bound-key boundary line
void vkx_bind_program(const prog_t *p, int stages);   // NVN stage bits
void vkx_prog_forget(const prog_t *p);                // finalized programs leave the donor's global bindings
void vkx_bind_buf(int ssbo, const p_bind_buf *b);
void vkx_bind_handles(int what, const p_bind_handles *b, const uint64_t *h);   // what: 0 textures, 1 samplers, 2 images; h = the handles as they are now
void vkx_set_pool(int tex, int base);
void vkx_trace(int on);
void vkx_dispatch(VkCommandBuffer c, const p_dispatch *p);
int vkx_draw(VkCommandBuffer c, uint32_t op, const void *a, uint32_t size, int ncolor, uint32_t rt_w, uint32_t rt_h, int samples); // 1 only when a Vulkan draw command was issued
void vkx_bind_state(const p_bind_state *s);
void vkx_viewports(const p_viewports *v); void vkx_scissor(const p_scissor *s); void vkx_depth_range(const p_depth_range *r);
void vkx_stencil(const p_stencil *s); void vkx_poly_offset(const p_poly_offset *p); void vkx_blend_color(const p_blend_color *b);
void vkx_depth_clamp(int on); void vkx_prim_restart(int on); void vkx_rast_discard(int on);
int vkr_ub_shadow(uint64_t addr, uint64_t size, VkBuffer *b, VkDeviceSize *off);   // vk_replay.c: a uniform binding over updates deferred in the pass: 1 and a shadow
int vkx_scissor_get(int r[4]);   // the scissor rectangle; 0 when none was set
uint32_t vkx_chmask(void);
void vkx_prof_take(uint64_t out[8]);   // DM_VK_PROF: the draws' time by part since the last take       // the channel write mask, 4 bits per colour slot
int vkx_origin_ll(void);   // the window origin is lower-left (the presented image is bottom-up)
void vkx_vattr(const p_bind_vattr *a); void vkx_vstream(const p_bind_vstream *s); void vkx_vbuf(const p_bind_vbuf *b);
// Per-queue bound state (vk_draw.c): select the queue about to submit (mtl_queue_activate), drop a finalized one (mtl_queue_forget)
void vkx_queue_select(uint64_t serial);
void vkx_queue_forget(uint64_t serial);
// Compute dispatches left out by the exact-key DM_VK_SKIP_CS diagnostic so far (vk_skipcs.h)
uint64_t vkx_skipcs_count(void);
// Compute shader keys dispatched since the last call, most first, " <key>:<count>"...; the DM_VK_OPS line (vk_replay.c)
int vkx_dispatch_keys_log(char *out, size_t n);
// Shader compiles in flight for 2 s or more, "N compiling, oldest <key> ... for S s" (vk_compile_watch.h); 0 = none. The GPU heartbeat appends it (vk_replay.c gp_thread)
int vkx_compile_text(char *out, size_t n);
