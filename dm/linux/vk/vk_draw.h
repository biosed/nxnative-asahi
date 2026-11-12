// Linux build: the draw path of the Vulkan replay (vk_draw.c), fed by the replay's walker (vk_replay.c)
#pragma once
#include "../gfx/nvn/nvn_obj.h"
#include "vk_mesh.h"
#include "vk_stage.h"
#include <vulkan/vulkan.h>

// Descriptor bindings are canonical within a shader stage, not across every
// stage. Keep one union for each graphics set plus a separate compute-set-0
// domain: a large pixel array must not reserve the same capacity in every
// vertex/task/mesh/compute set allocated by a submission.
enum {
    VKX_DESCRIPTOR_BINDING_LIMIT = 1024,
    VKX_DESCRIPTOR_TYPE_COUNT = 5,
};

typedef struct { uint32_t texture, sampler; } vkx_bindless_handle_ids;
// An NVN handle stores a pool-local 20-bit texture ID and 12-bit sampler ID.
// ID zero remains the null descriptor; nonzero IDs are rebased to the pools
// bound by nvnCommandBufferSetTexturePool/SamplerPool and fail closed if the
// persistent Vulkan tables cannot represent them.
static inline vkx_bindless_handle_ids vkx_bindless_ids(uint32_t handle,
                                                       int texture_base,
                                                       int sampler_base,
                                                       uint32_t texture_count,
                                                       uint32_t sampler_count) {
    uint32_t tl = handle & 0xfffffu, sl = handle >> 20;
    uint64_t tg = texture_base >= 0 ? (uint64_t)tl + (uint32_t)texture_base : UINT64_MAX;
    uint64_t sg = sampler_base >= 0 ? (uint64_t)sl + (uint32_t)sampler_base : UINT64_MAX;
    return (vkx_bindless_handle_ids){
        tl && tg < texture_count ? (uint32_t)tg : 0,
        sl && sg < sampler_count ? (uint32_t)sg : 0,
    };
}
typedef struct {
    char type[VKX_STATE_STAGE_COUNT][VKX_DESCRIPTOR_BINDING_LIMIT];
    uint32_t count[VKX_STATE_STAGE_COUNT][VKX_DESCRIPTOR_BINDING_LIMIT];
} vkx_descriptor_union;

static inline int vkx_descriptor_layout_domain(int stage) {
    if (stage == VKX_STAGE_CS) return VKX_STAGE_CS;
    int set = vkx_stage_set(stage);
    return set >= 0 && set < VKX_STATE_STAGE_COUNT ? set : -1;
}

// 1: added, 0: descriptor-type conflict in the same domain, -1: out of range.
static inline int vkx_descriptor_union_add(vkx_descriptor_union *u, int stage,
                                           uint32_t binding, char type,
                                           uint32_t count) {
    int domain = vkx_descriptor_layout_domain(stage);
    if (!u || domain < 0 || binding >= VKX_DESCRIPTOR_BINDING_LIMIT) return -1;
    if (u->type[domain][binding] && u->type[domain][binding] != type) return 0;
    u->type[domain][binding] = type;
    if (!count) count = 1;
    if (count > u->count[domain][binding]) u->count[domain][binding] = count;
    return 1;
}

static inline int vkx_descriptor_type_index(char type) {
    static const char types[VKX_DESCRIPTOR_TYPE_COUNT] = {'u', 's', 't', 'i', 'b'};
    for (int i = 0; i < VKX_DESCRIPTOR_TYPE_COUNT; i++) if (type == types[i]) return i;
    return -1;
}

static inline void vkx_descriptor_union_totals(const vkx_descriptor_union *u,
                                               int domain,
                                               uint32_t out[VKX_DESCRIPTOR_TYPE_COUNT]) {
    for (int i = 0; i < VKX_DESCRIPTOR_TYPE_COUNT; i++) out[i] = 0;
    if (!u || domain < 0 || domain >= VKX_STATE_STAGE_COUNT) return;
    for (uint32_t b = 0; b < VKX_DESCRIPTOR_BINDING_LIMIT; b++) {
        int type = vkx_descriptor_type_index(u->type[domain][b]);
        if (type >= 0) out[type] += u->count[domain][b];
    }
}

enum {
    VKX_DESCRIPTOR_POOL_SETS = 256,
    VKX_DESCRIPTOR_SET_HIGH_WATER = 8192,
    VKX_DESCRIPTOR_POOL_HIGH_WATER = 192,
};
// A pool allocates all of its same-layout sets together once.  Recording takes
// each handle at most once until the owning ring-slot fence passes and its used
// count is reset; the handles themselves stay valid and can then be rewritten.
static inline int vkx_descriptor_set_take(uint16_t *used, uint16_t *index) {
    if (*used >= VKX_DESCRIPTOR_POOL_SETS) return 0;
    *index = (*used)++;
    return 1;
}
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
int vkx_mesh_prebridge(VkCommandBuffer c, uint64_t ind); // outside rendering: copy an indirect mesh draw's {count, first} now, for that draw later in the pass
void vkx_mesh_prebridge_barrier(VkCommandBuffer c);       // after the prebridges of a pass, before it begins
void vkx_mesh_prebridge_stats(uint64_t *hits, uint64_t *misses, int *valid);   // diagnostic: indirect mesh draws that found their early copy, and not
int vkx_mesh_prebridged(uint64_t ind);                     // an unused prebridge of this record is ready (the pass need not end for the draw)
void vkx_mesh_prebridge_reset(void);                       // the pass ended: unused prebridges are dropped
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
int vkx_dispatch_relax(const p_dispatch *p);       // exact-key diagnostic: current compute program is selected for bounded barrier relaxation
unsigned vkx_dispatch_relax_batch(void);           // selected dispatches per independently ordered batch (1..64)
// Graphics c0 + DMSH_C0_BASE_INSTANCE must describe the draw before its
// descriptor set is built. Maxwell's instance ID omits this value, and the
// translated shader adds the driver word when it indexes per-instance data.
static inline uint32_t vkx_draw_base_instance(const p_draw *d,
                                              const p_draw_idx *x) {
    return (uint32_t)(d ? d->base_instance : x ? x->base_instance : 0);
}
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
