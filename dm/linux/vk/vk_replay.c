// Linux build: the NVN command replay on Vulkan, behind the mtl_* calls hle_nvn.c makes (nvn_obj.h). Step 3.2 of the backend (LINUX.md): the skeleton.
//   - one command buffer per submission (mtl_sub_begin … mtl_sub_commit), signalling a timeline semaphore with the submission's fence value, so mtl_wait and
//     mtl_finish wait for the GPU itself
//   - textures as VkImages, made the first time a texture is a render target or presented, found by the texture's serial (the game copies tex_t by value)
//   - the record stream walked as nvn_mtl.m walks it (END returns from a CALL or ends the handle, JUMP follows a segment, CALL descends); set render targets and the
//     colour / depth-stencil clears are carried out, every other record is counted (DM_VK_OPS=1 logs the counts)
//   - mtl_present hands the presented texture to the window (vk_window.c), which copies it once the GPU has finished the submissions before it
// Images stay in VK_IMAGE_LAYOUT_GENERAL and each clear is followed by a full memory barrier: correct and simple, fast enough for a skeleton.
// Without a device (DM_VK=0) every call behaves as the headless backend did: fences complete at once.
#include <stddef.h>
#include <stdatomic.h>
#include "dm.h"
#include "vk_dev.h"
#include "vk_push.h"
#include "vk_fmt.h"
#include "vk_window.h"
#include "vk_draw.h"
#include "vk_mem.h"
#include "vk_gmem_trace.h"
#include "vk_counter_report.h"
#include "vk_cs_gpu_lifecycle.h"
#include "vk_image_refs.h"
#include "vk_tex_lifetime.h"
int nvn_dpool_base(const void *pool, int tex);   // hle_nvn.c: a descriptor pool's ID base
#include "../gfx/nvn/nvn_obj.h"
#include "../gfx/nvn/nvn_hmap.h"
#include "../gfx/nvn/nvn_dirty.h"
#include "../gfx/nvn/nvn_timestamp.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include <stdatomic.h>

#include "vk_replay.h"
// Optional runtime-side bridge for Ocarina's decoded NV12 frame.  Keeping the
// reference weak preserves standalone renderer/test links.
extern int dm_codec_movie_nv12_present(void *dst, uint32_t w, uint32_t h) __attribute__((weak));
pthread_mutex_t g_tmu = PTHREAD_MUTEX_INITIALIZER;
// Images are shared by storage, as nvn_mtl.m's sharing table does: an NVN texture object is only a descriptor, and an object rebuilt over the same storage with the same description
// is the same texture. Mario Wonder rebuilds its transient render targets (and the cube maps its lighting samples) on new objects every frame: one image per object made 74,546
// images in two minutes over 447 storages, and each new one started empty, so what the GPU had rendered into the old one was lost (black lit models). Serials map to images
// (g_tab); images are found by storage (g_stab) and stay while no object refers to them, until a texture of another description takes over their storage
typedef struct sref { uint64_t serial; vtex *t; struct sref *next; } sref;
static sref *g_tab[4096]; static vtex *g_stab[4096]; static vtex *g_dead;
static nvn_hmap g_tex_live = {.klen = 8};   // serial -> the NVNtexture passed to Initialize, under g_tmu
static uint32_t stor_hash(const uint8_t *mem, const nvn_texdesc *d) { uint64_t h = (uint64_t)(uintptr_t)mem >> 6; h ^= (uint64_t)d->width * 31 + d->height * 131 + (uint64_t)d->format * 7919 + (uint64_t)d->target * 104729; return (uint32_t)(h ^ h >> 13) % 4096; }
static VkSemaphore g_tl; static _Atomic uint64_t g_fence;
static VkCommandPool g_pool; static VkCommandBuffer g_cb[CB_RING]; static _Atomic uint64_t g_cb_val[CB_RING]; static int g_cb_i = -1, g_cb_n = CB_RING, g_open;
// DM_VK_MESH_LOG submission breadcrumbs. A mesh command is tagged while it
// is recorded, then associated with the exact replay timeline value when its
// command buffer is submitted. Diagnostics only: no ordering change.
static _Atomic uint64_t g_mesh_issue_seq, g_mesh_last_signal;
static uint64_t g_cb_mesh_seq[CB_RING];
static int g_on = -1;
static const tex_t *g_rt[8], *g_rt_depth; static view_t g_rt_view[8], g_rt_dview; static uint8_t g_rt_has_view[8], g_rt_has_dview; static int g_rt_n;
static uint64_t g_ops[64], g_clears; static _Atomic uint64_t g_subs; static uint64_t g_live, g_dead_bytes;
static _Atomic uint64_t g_image_bytes, g_spec_image_bytes;
// Work the replay cannot do yet is counted and reported (DM_VK_OPS=1 adds it to the records line) rather than drawn wrongly: NVN2 mesh/task draws stay
// behind the exact root physical-proof token, alongside indirect-count multi-draws and records current nxnative
// added after this replay was written. Ordinary indirect draws are supported and counted as forwarded. Each unsupported kind logs [RED] once.
static _Atomic uint64_t g_skip_mesh, g_ind_seen, g_skip_mdi, g_push_applied, g_push_dropped, g_skip_multi, g_skip_dlss, g_skip_draw_tex, g_skip_alpha;
static _Atomic uint64_t g_mesh_fail_bridge, g_mesh_fail_rp, g_mesh_fail_draw;
#define SKIP_ONCE(ctr, ...) do { if (atomic_fetch_add(&(ctr), 1) == 0) LOG("[RED] vulkan replay: " __VA_ARGS__); } while (0)
static _Atomic uint64_t g_samples_placeholder;   // reports written that way (the DM_VK_OPS line)
static vkx_push_st g_push;   // the NVN2 push-constant blocks, graphics and compute, as the record stream left them (vk_push.h; replay thread only)
const uint8_t *vkr_push(int compute) { return g_push.block[compute ? 1 : 0]; }
void mtl_skip_stats(uint64_t out[6]) {
    out[0] = atomic_load(&g_skip_mesh); out[1] = atomic_load(&g_ind_seen); out[2] = atomic_load(&g_skip_mdi);
    out[3] = atomic_load(&g_push_dropped); out[4] = atomic_load(&g_skip_multi); out[5] = atomic_load(&g_skip_dlss);
}
static VkCommandBuffer g_up[CB_RING]; static int g_up_open;   // the uploads of the submission in progress (loading textures)
static char g_dump_dir[900]; static int g_dump_left = -1; static uint64_t g_dump_ser[64]; static int g_ndump;   // DM_VK_TEXDUMP

// GPU time per frame (mtl_gpu_ms, the long-frame log and DM_NVN_TSV's @gpu_ms): a timestamp at the start and end of each submission, summed into the frame the game was
// recording when it began. Read once the submission is done, by the next use of its ring slot or by mtl_gpu_ms
enum { SEG_Q = 128 };   // timestamp reports with a query of their own, per part (seg_ts)
static VkQueryPool g_sq; static int g_sq_n[CB_RING];
static VkQueryPool g_qp; static uint64_t g_cb_frame[CB_RING]; static int g_cb_timed[CB_RING];
typedef struct { uint64_t f, ns, t0, t1; } gpu_frame;
static gpu_frame g_gf[64]; static pthread_mutex_t g_gfmu = PTHREAD_MUTEX_INITIALIZER;
uint64_t vkh_frame(void); static uint64_t done_value(void);
static uint64_t mono_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }

// Per-dispatch GPU timing is deliberately separate from the always-on frame
// queries.  It is a diagnostic, enabled by DM_VK_CS_GPU=1, and must not split
// command buffers or introduce a wait: timestamps are written around the
// existing dispatch and read only after the ring slot's timeline value has
// completed.  Both GPU queries and their host metadata have fixed bounds.
#define CS_GPU_PER_SLOT 512
#define CS_GPU_DEFAULT_LIMIT 4096
#define CS_GPU_MAX_LIMIT 65536
typedef struct {
    char key[17];
    uint32_t x, y, z;
    uint64_t indirect, frame;
} cs_gpu_meta;
static VkQueryPool g_cs_qp;
static cs_gpu_meta g_cs_meta[CB_RING][CS_GPU_PER_SLOT];
static vkr_cs_gpu_batch g_cs_batch[CB_RING];
static uint64_t g_cs_recorded, g_cs_logged, g_cs_limit;
static _Atomic uint64_t g_cs_overflow;
static int g_cs_cap_logged;
static uint32_t cs_query(int slot, int sample) {
    return 2u * ((uint32_t)slot * CS_GPU_PER_SLOT + (uint32_t)sample);
}
static void cs_gpu_harvest(int slot, uint64_t done) {   // under g_tmu: only published batches are visible
    vkr_cs_gpu_batch *b = &g_cs_batch[slot];
    if (!g_cs_qp || !vkr_cs_gpu_harvestable(b, done)) return;
    uint32_t n = b->submitted;
    uint64_t q[2 * CS_GPU_PER_SLOT];
    VkResult r = vkGetQueryPoolResults(g_vkd.dev, g_cs_qp, cs_query(slot, 0),
        2 * n, sizeof q, q, sizeof *q, VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) {
        LOG("[RED] vulkan compute GPU profile: %u timestamp pairs from completed slot %d could not be read (%d)", n, slot, (int)r);
        vkr_cs_gpu_retire(b);
        return;
    }
    double per = g_vkd.props.limits.timestampPeriod;
    for (uint32_t i = 0; i < n; i++) {
        cs_gpu_meta *m = &g_cs_meta[slot][i];
        uint64_t ticks = q[2 * i + 1] > q[2 * i] ? q[2 * i + 1] - q[2 * i] : 0;
        double ms = ticks * per / 1e6;
        if (m->indirect)
            LOG("[vk-cs-gpu] frame %llu shader %s groups indirect@%#llx record %ux%ux%u gpu %.3f ms",
                (unsigned long long)m->frame, m->key, (unsigned long long)m->indirect,
                m->x, m->y, m->z, ms);
        else
            LOG("[vk-cs-gpu] frame %llu shader %s groups %ux%ux%u gpu %.3f ms",
                (unsigned long long)m->frame, m->key, m->x, m->y, m->z, ms);
        g_cs_logged++;
    }
    vkr_cs_gpu_retire(b);
    if (g_cs_logged == g_cs_limit)
        LOG("vulkan compute GPU profile: completed bounded sample of %llu dispatches%s",
            (unsigned long long)g_cs_logged, atomic_load(&g_cs_overflow) ? " (some command-buffer slots overflowed before the global cap)" : "");
}
int vkr_cs_gpu_begin(VkCommandBuffer c, const char key[17], uint32_t x,
                     uint32_t y, uint32_t z, uint64_t indirect) {
    if (!g_cs_qp || g_cb_i < 0 || c != g_cb[g_cb_i]) return -1;
    { static long after = -1; if (after < 0) { const char *e = getenv("DM_VK_CS_GPU_AFTER"); after = e ? atol(e) : 0; }   // DM_VK_CS_GPU_AFTER=<frame>: no samples before that frame, so the bounded total reaches gameplay
      if (after > 0 && vkh_frame() < (uint64_t)after) return -1; }
    if (g_cs_recorded >= g_cs_limit) {
        if (!g_cs_cap_logged++)
            LOG("vulkan compute GPU profile: %llu-dispatch recording cap reached; existing samples will still be harvested",
                (unsigned long long)g_cs_limit);
        return -1;
    }
    int i = vkr_cs_gpu_record(&g_cs_batch[g_cb_i], CS_GPU_PER_SLOT);
    if (i < 0) {
        uint64_t overflow = atomic_fetch_add(&g_cs_overflow, 1) + 1;
        if (!(overflow & (overflow - 1)))
            LOG("[RED] vulkan compute GPU profile: command-buffer slot %d exceeded %d dispatches (%llu untimed so far); execution is unchanged",
                g_cb_i, CS_GPU_PER_SLOT, (unsigned long long)overflow);
        return -1;
    }
    cs_gpu_meta *m = &g_cs_meta[g_cb_i][i];
    snprintf(m->key, sizeof m->key, "%s", key);
    m->x = x; m->y = y; m->z = z; m->indirect = indirect; m->frame = vkh_frame();
    g_cs_recorded++;
    vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_cs_qp, cs_query(g_cb_i, i));
    return i;
}
void vkr_cs_gpu_end(VkCommandBuffer c, int token) {
    if (!g_cs_qp || token < 0 || token >= CS_GPU_PER_SLOT || g_cb_i < 0 || c != g_cb[g_cb_i]) return;
    vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_cs_qp, cs_query(g_cb_i, token) + 1);
}
static void gpu_harvest(int i) {   // slot i's timestamps into its frame (its submission is done)
    if (!g_cb_timed[i]) return;
    uint64_t q[2]; if (vkGetQueryPoolResults(g_vkd.dev, g_qp, (uint32_t)(2 * i), 2, sizeof q, q, 8, VK_QUERY_RESULT_64_BIT)) return;
    g_cb_timed[i] = 0;
    double per = g_vkd.props.limits.timestampPeriod; uint64_t ns = q[1] > q[0] ? (uint64_t)((q[1] - q[0]) * per) : 0;
    pthread_mutex_lock(&g_gfmu);
    gpu_frame *g = &g_gf[g_cb_frame[i] % 64];
    if (g->f != g_cb_frame[i]) *g = (gpu_frame){g_cb_frame[i], 0, UINT64_MAX, 0};
    g->ns += ns; if ((uint64_t)(q[0] * per) < g->t0) g->t0 = (uint64_t)(q[0] * per); if ((uint64_t)(q[1] * per) > g->t1) g->t1 = (uint64_t)(q[1] * per);
    pthread_mutex_unlock(&g_gfmu);
}
static void gpu_harvest_done(void) {
    uint64_t d = done_value();
    for (int i = 0; i < CB_RING; i++) {
        uint64_t submitted = atomic_load(&g_cb_val[i]);
        if (submitted && submitted <= d && g_cb_timed[i]) gpu_harvest(i);
        cs_gpu_harvest(i, d);
    }
}
double mtl_gpu_ms(uint64_t f) {
    if (g_on <= 0 || !g_qp) return -1;
    pthread_mutex_lock(&g_tmu); gpu_harvest_done(); pthread_mutex_unlock(&g_tmu);
    pthread_mutex_lock(&g_gfmu); double r = g_gf[f % 64].f == f ? g_gf[f % 64].ns / 1e6 : -1; pthread_mutex_unlock(&g_gfmu);
    return r;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }   // (GPU clock, not the host's: not given)
static int vkr_on(void) {
    if (g_on >= 0) return g_on;
    if (!vkd_ok()) return g_on = 0;
    // Diagnostic memory bound: a per-mesh flush normally rotates through all
    // CB_RING slots, warming each slot's replay arena/descriptor pools and
    // keeping several driver command buffers alive at once.  A smaller active
    // ring makes sub_open wait, reset and reuse those slots sooner without
    // changing the normal build's 16-slot allocation or default behaviour.
    // DM_VK_CB_RING=1 is the strict first-gameplay probe: every flushed mesh
    // command completes before the same slot is recorded again.
    const char *e = getenv("DM_VK_CB_RING");
    if (e) {
        char *end = NULL; long n = strtol(e, &end, 10);
        if (end != e && !*end && n >= 1 && n <= CB_RING) g_cb_n = (int)n;
        else LOG("[RED] vulkan replay: DM_VK_CB_RING=%s is not 1..%d; using %d", e, CB_RING, CB_RING);
    }
    VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_vkd.qf};
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = CB_RING};
    if (vkCreateSemaphore(g_vkd.dev, &si, NULL, &g_tl) || vkCreateCommandPool(g_vkd.dev, &pci, NULL, &g_pool)) { LOG("[RED] vulkan replay: no timeline semaphore or command pool"); return g_on = 0; }
    ai.commandPool = g_pool;
    if (vkAllocateCommandBuffers(g_vkd.dev, &ai, g_cb) || vkAllocateCommandBuffers(g_vkd.dev, &ai, g_up)) { LOG("[RED] vulkan replay: no command buffers"); return g_on = 0; }
    VkQueryPoolCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * CB_RING};
    if (!g_vkd.props.limits.timestampComputeAndGraphics || vkCreateQueryPool(g_vkd.dev, &qci, NULL, &g_qp)) { g_qp = VK_NULL_HANDLE; LOG("vulkan replay: no GPU timestamps (GPU time is not measured)"); }
    if (g_qp) { VkQueryPoolCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SEG_Q * CB_RING};
        if (vkCreateQueryPool(g_vkd.dev, &sci, NULL, &g_sq)) g_sq = VK_NULL_HANDLE; }
    const char *cs_gpu = getenv("DM_VK_CS_GPU");
    if (cs_gpu && strcmp(cs_gpu, "0")) {
        g_cs_limit = CS_GPU_DEFAULT_LIMIT;
        const char *lim = getenv("DM_VK_CS_GPU_LIMIT");
        if (lim) {
            char *end = NULL; unsigned long long n = strtoull(lim, &end, 10);
            if (end != lim && !*end && n >= 1 && n <= CS_GPU_MAX_LIMIT) g_cs_limit = n;
            else LOG("[RED] vulkan replay: DM_VK_CS_GPU_LIMIT=%s is not 1..%d; using %d", lim, CS_GPU_MAX_LIMIT, CS_GPU_DEFAULT_LIMIT);
        }
        qci.queryCount = 2 * CB_RING * CS_GPU_PER_SLOT;
        if (!g_vkd.props.limits.timestampComputeAndGraphics || vkCreateQueryPool(g_vkd.dev, &qci, NULL, &g_cs_qp))
            LOG("[RED] vulkan replay: DM_VK_CS_GPU requested, but compute timestamps are unavailable");
        else
            LOG("vulkan compute GPU profile: timing at most %d dispatches per command buffer and %llu total; no submissions or waits are added",
                CS_GPU_PER_SLOT, (unsigned long long)g_cs_limit);
    }
    // DM_VK_TEXDUMP=<dir>[:n]: the first n textures loaded from guest memory (16 by default, at most 64) as PNG files in dir, level 0 and layer 0
    const char *td = getenv("DM_VK_TEXDUMP");
    if (td && *td) { snprintf(g_dump_dir, sizeof g_dump_dir, "%s", td); char *cn = strrchr(g_dump_dir, ':'); g_dump_left = 16; if (cn) { g_dump_left = atoi(cn + 1); *cn = 0; } if (g_dump_left > 64) g_dump_left = 64; }
    LOG("vulkan replay: on (skeleton: render targets and clears; draws are counted, not drawn); command ring %d/%d", g_cb_n, CB_RING);
    return g_on = 1;
}
VkSemaphore vkr_timeline(void) { return g_tl; }
uint64_t vkr_last_mesh_fence(void) { return atomic_load(&g_mesh_last_signal); }
static uint64_t done_value(void) { uint64_t v = 0; vkGetSemaphoreCounterValue(g_vkd.dev, g_tl, &v); return v; }
static int mesh_log_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_VK_MESH_LOG"); on = e && atoi(e) != 0; }
    return on;
}
static int wait_value_named(uint64_t v, uint64_t timeout_ns, const char *where, int slot) {
    if (!v) return 1;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_tl, .pValues = &v};
    VkResult r;
    unsigned polls = 0;
    do {
        uint64_t once = timeout_ns == UINT64_MAX && mesh_log_on() ? 1000000000ull : timeout_ns;
        r = vkWaitSemaphores(g_vkd.dev, &wi, once);
        if (r == VK_TIMEOUT && timeout_ns == UINT64_MAX && mesh_log_on()) {
            polls++;
            uint64_t seq = atomic_load(&g_mesh_issue_seq), mf = atomic_load(&g_mesh_last_signal);
            if (seq && (polls == 1 || !(polls % 5)))
                LOG("[RED] vulkan mesh wait: %s slot=%d wants timeline=%llu done=%llu; last mesh issue=%llu signal=%llu; waiting %us",
                    where ? where : "unknown", slot, (unsigned long long)v, (unsigned long long)done_value(),
                    (unsigned long long)seq, (unsigned long long)mf, polls);
            continue;
        }
        break;
    } while (1);
    if (r == VK_TIMEOUT) return 0;
    if (r) { LOG("[RED] vulkan replay: vkWaitSemaphores failed (%d)", (int)r); return 1; }
    if (polls && atomic_load(&g_mesh_issue_seq))
        LOG("vulkan mesh wait: %s slot=%d timeline=%llu completed after %us", where ? where : "unknown", slot,
            (unsigned long long)v, polls);
    return 1;
}
static int wait_value(uint64_t v, uint64_t timeout_ns) { return wait_value_named(v, timeout_ns, "generic", -1); }

// ---- values the GPU writes ----
// Event signals (CommandBufferSignalEvent, ReportValue) and counter reports (ReportCounter) write guest memory when the GPU reaches them; the game reads that memory itself, with no
// host call. They are queued with the timeline value of the submission part that recorded them and written once the GPU is done with that part, in order, as nvn_mtl.m's post_run
// does in its completion handler: by a thread that waits on the timeline, and by any host wait that sees the value first, so a sync wait returns with them written. Mario Kart 8
// Deluxe's "Select a Mii" waited for such values to build its list: with them dropped, it stayed empty and took no input
typedef struct { uint64_t at; int kind, slot, q; volatile uint32_t *val; uint32_t mode, v; uint64_t addr, type; } gpost;   // kind 1: event (mode 1 adds), 2: counter report; slot: the command buffer recorded in; q: the report's own timestamp query in that slot (seg_ts), -1 none
static gpost *g_gp; static size_t g_gpn, g_gpcap; static pthread_mutex_t g_gpmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_gpcv = PTHREAD_COND_INITIALIZER;
// A counter report's time is the GPU time of the submission part it was recorded in, as nvn_mtl.m's segment clock gives it: TIMESTAMP_TOP (0x10) the part's start, any other
// type its end, from the part's timestamp queries (g_qp), placed on the guest clock against the newest part written now. Breath of the Wild measures its GPU time with a top
// report at the start of a frame and another at its end, and scales its resolution down when that passes 175% of a 60 Hz frame (main+0x1267070): with one time for all of
// them it read about 0 and stayed at its largest resolution however long the GPU took. Without the queries (or before a part's are ready) the report takes the time now
// Timestamps by segment (profile gpu_timestamps segment; DM_VK_GPU_TS=segment / part overrides it), as nvn_mtl.m's seg_ts: on hardware a timestamp report is
// written when the GPU reaches it, so the reports of one submission bracket the GPU work between them. Ocarina of Time makes about 26 TIMESTAMP_TOP reports
// a frame, one before each of its passes, and two plain ones, and its dynamic resolution decides from their differences: with a part's start for every TOP
// report they were all equal, the game measured 7.5 ms of a 35 to 60 ms frame and never left its largest resolution. Each timestamp report now takes a query
// of its own at its place in the command buffer: TOP_OF_PIPE for TIMESTAMP_TOP (the start of the work that follows), BOTTOM_OF_PIPE for the others (the end
// of the work before). SEG_Q reports a part; more, or a query not ready, take the part's span as before
static int seg_ts(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_GPU_TS"); v = e && *e ? !strcmp(e, "segment") : DM_GAME_GPU_TS_SEGMENT;
        LOG("vulkan replay: counter timestamps: %s (profile key gpu_timestamps; DM_VK_GPU_TS=segment / part)", v ? "each report's own GPU time" : "the GPU start / end of the part holding the report"); }
    return v;
}
static int seg_query(VkCommandBuffer c, uint64_t type) {   // on the replay thread: the query a timestamp report is written to in the part being recorded, -1 none
    if (!g_sq || !seg_ts() || g_cb_i < 0 || g_sq_n[g_cb_i] >= SEG_Q) return -1;
    int q = g_sq_n[g_cb_i]++;
    vkCmdWriteTimestamp(c, type == 0x10 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_sq, (uint32_t)(g_cb_i * SEG_Q + q));
    return q;
}
static int seg_tick(int slot, int q, uint64_t *t) {   // a report's own GPU time in device ticks
    if (!g_sq || slot < 0 || q < 0) return 0;
    uint64_t v; if (vkGetQueryPoolResults(g_vkd.dev, g_sq, (uint32_t)(slot * SEG_Q + q), 1, sizeof v, &v, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS || !v) return 0;
    *t = v; return 1;
}
static int gp_span(int slot, uint64_t *t0, uint64_t *t1) {   // the part's GPU start / end in device ticks
    if (!g_qp || slot < 0) return 0;
    uint64_t q[2]; if (vkGetQueryPoolResults(g_vkd.dev, g_qp, (uint32_t)(2 * slot), 2, sizeof q, q, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS || q[1] < q[0]) return 0;
    *t0 = q[0]; *t1 = q[1]; return 1;
}
// DM_VK_TSLOG=1 (diagnostic): what the game's timestamp reports are given, every 600 of them: how many were TIMESTAMP_TOP (0x10) and plain TIMESTAMP (0), how
// many took a part's GPU span, the mean span, and the mean and largest time between one TOP report and the next plain one, which is the GPU time a game that
// brackets its frame that way measures (Ocarina of Time's dynamic resolution decides from it). Under g_gpmu
static void ts_log(const gpost *o, uint64_t ns, int spanned, double span_ms) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_TSLOG"); on = e && atoi(e) != 0; }
    if (!on) return;
    static uint64_t n, ntop, nspan, last_top, npair; static double span_sum, pair_sum, pair_max; static int slots[2] = {-1, -1}; static uint64_t same_slot;
    n++; if (spanned) { nspan++; span_sum += span_ms; }
    if (o->type == 0x10) { ntop++; last_top = ns; slots[0] = o->slot; }
    else if (last_top) { double d = ns >= last_top ? (double)(ns - last_top) / 1e6 : 0; npair++; pair_sum += d; if (d > pair_max) pair_max = d; if (o->slot == slots[0]) same_slot++; last_top = 0; }
    if (n % 600) return;
    LOG("vulkan timestamps: %llu reports (%llu top), %llu with a part span (mean %.2f ms); top-to-next pairs %llu, mean %.2f ms, max %.2f ms, %llu in the same part",
        (unsigned long long)n, (unsigned long long)ntop, (unsigned long long)nspan, nspan ? span_sum / nspan : 0, (unsigned long long)npair, npair ? pair_sum / npair : 0, pair_max, (unsigned long long)same_slot);
    ntop = nspan = npair = same_slot = 0; span_sum = pair_sum = pair_max = 0;   // each line is one window of 600 reports
}
static void gp_run(uint64_t done) {   // writes everything the GPU is done with
    pthread_mutex_lock(&g_gpmu);
    size_t k = 0; uint64_t now = guest_ns(), ref = 0; double per = g_vkd.props.limits.timestampPeriod;   // ref: the latest part end among these, read as now
    for (size_t i = 0; i < g_gpn && g_gp[i].at <= done; i++) { uint64_t a, b; if (g_gp[i].kind == 2 && gp_span(g_gp[i].slot, &a, &b) && b > ref) ref = b; }
    for (; k < g_gpn && g_gp[k].at <= done; k++) {
        const gpost *o = &g_gp[k];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; continue; }
        uint64_t ns = now, a, b;
        if (ref && gp_span(o->slot, &a, &b)) { uint64_t at = o->type == 0x10 ? a : b, own;
            if (seg_tick(o->slot, o->q, &own) && own <= ref) at = own;   // its own place in the part
            uint64_t back = (uint64_t)((double)(ref - at) * per); ns = back < now ? now - back : 0; }
        if (o->type == 0 || o->type == 0x10) ts_log(o, ns, ref != 0 && gp_span(o->slot, &a, &b), ref ? (double)(b - a) * per / 1e6 : 0);
        uint64_t t = nvn_timestamp_from_ns(ns), rec[2] = {vkr_counter_value(o->type, t, DM_GAME_RENDER_COND, NVN_COUNTER_TYPE_SAMPLES_PASSED), t}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);   // render_cond: a timestamp's counter field is 0, as on the GPU (rc_value)
    }
    if (k) { memmove(g_gp, g_gp + k, (g_gpn - k) * sizeof *g_gp); g_gpn -= k; }
    pthread_mutex_unlock(&g_gpmu);
}
static void *gp_thread(void *arg) {
    (void)arg;
    static int heartbeat = -1;
    static uint64_t heartbeat_at;
    for (;;) {
        pthread_mutex_lock(&g_gpmu);
        while (!g_gpn) pthread_cond_wait(&g_gpcv, &g_gpmu);
        uint64_t at = g_gp[0].at;
        pthread_mutex_unlock(&g_gpmu);
        if (wait_value(at, 100000000)) gp_run(done_value());   // a value not submitted yet is waited for again
        if (heartbeat < 0) {
            const char *e = getenv("DM_VK_GPU_HEARTBEAT");
            heartbeat = e && atoi(e) != 0;
        }
        uint64_t now = mono_ns();
        if (heartbeat && now - heartbeat_at >= 1000000000ull) {
            heartbeat_at = now;
            char cw[160]; int ncw = vkx_compile_text(cw, sizeof cw);   // a replay thread inside the driver's compile looks like a GPU that never finishes: name the compile
            LOG("vulkan GPU heartbeat: submitted=%llu done=%llu report-wait=%llu%s%s",
                (unsigned long long)atomic_load(&g_fence),
                (unsigned long long)done_value(), (unsigned long long)at, ncw ? " " : "", cw);
        }
    }
    return NULL;
}
static void gp_add(gpost o) {   // on the replay thread, in recording order
    static int started; if (!started) { started = 1; pthread_t t; if (!pthread_create(&t, NULL, gp_thread, NULL)) pthread_detach(t); else LOG("[RED] vulkan replay: no thread for GPU-written values"); }
    o.at = atomic_load(&g_fence) + 1; o.slot = g_cb_i;   // the part being recorded signals this value (sub_submit), from this ring slot
    pthread_mutex_lock(&g_gpmu);
    if (g_gpn == g_gpcap) { g_gpcap = g_gpcap ? g_gpcap * 2 : 256; g_gp = realloc(g_gp, g_gpcap * sizeof *g_gp); }
    g_gp[g_gpn++] = o; pthread_cond_signal(&g_gpcv);
    pthread_mutex_unlock(&g_gpmu);
}
// ---- conditional rendering (profile render_cond) ----
// SetRenderEnableConditional(mode, rec) draws only while the 64-bit counters at rec and rec + 16 compare equal (mode 0) or not equal (mode 1), until SetRenderEnable(true), as
// nvn_mtl.m's OP_RENDER_COND evaluates it: on the CPU at replay, from the reports in memory. Animal Crossing: New Horizons and Tears of the Kingdom wrap their clears this way,
// with a TIMESTAMP_TOP report into rec after each clear (its counter field 0) so a target is not cleared again until the CPU sets rec: with the condition ignored (as this
// replay did) every clear ran, four clear-only passes wiped Animal Crossing's composite before the window copy, and every frame was black. A report still queued (g_gp) counts
// with the value it will write: 0 for a timestamp, unknown for another counter. A condition on an address a samples-passed report ever targeted (never written here: no
// occlusion queries) is unknown too, and an unknown condition draws, as before;
// the completion-time visible placeholder is still written for guest waits
static int g_rc_skip;   // a render-enable block evaluated false is active (draws and clears dropped)
static uint64_t g_rc_occ[4096];   // addresses ever given a non-timestamp report (open addressing)
static void rc_occ_add(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return; if (!g_rc_occ[i]) { g_rc_occ[i] = a; return; } } }
static int rc_occ_has(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return 1; if (!g_rc_occ[i]) return 0; } return 0; }
static int rc_value(uint64_t a, uint64_t *v) {   // 0 = unknown now
    if (rc_occ_has(a)) return 0;
    int r = -1;
    pthread_mutex_lock(&g_gpmu);
    for (size_t i = g_gpn; i-- > 0;) if (g_gp[i].kind == 2 && g_gp[i].addr == a) { r = g_gp[i].type == 0 || g_gp[i].type == 0x10; *v = 0; break; }
    if (r < 0) { *v = *(const volatile uint64_t *)(uintptr_t)a; r = 1; }   // read under the lock: gp_run writes the queued ones under it
    pthread_mutex_unlock(&g_gpmu);
    return r;
}
static void rc_eval(const p_render_cond *rc) {
    if (rc->mode >= 2) { g_rc_skip = rc->mode == 3; return; }
    uint64_t x = 0, y = 0; int known = rc_value(rc->addr, &x) & rc_value(rc->addr + 16, &y);
    g_rc_skip = known && (rc->mode ? x == y : x != y);
}

// ---- textures ----
static void tex_ready(VkCommandBuffer c, vtex *t);
static VkCommandBuffer up_cb(void);
static void rp_end(VkCommandBuffer c);
static void pc_flush(VkCommandBuffer c);
static vtex *tex_find(uint64_t serial) { for (sref *r = g_tab[serial % 4096]; r; r = r->next) if (r->serial == serial) return r->t; return NULL; }
static void sref_add(uint64_t serial, vtex *t) {
    if (!vkx_image_serial_ref_add(&t->refs, &t->serial_refs,
                                  nvn_hm_get(&g_tex_live, &serial) != NULL)) return;
    sref *r = calloc(1, sizeof *r);
    if (!r) { vkx_image_serial_ref_drop(&t->refs, &t->serial_refs); return; }
    r->serial = serial; r->t = t; r->next = g_tab[serial % 4096]; g_tab[serial % 4096] = r;
}
static void sref_drop_all(vtex *t) {
    for (int h = 0; h < 4096; h++) for (sref **p = &g_tab[h]; *p;) {
        if ((*p)->t == t) { sref *r = *p; *p = r->next; free(r); }
        else p = &(*p)->next;
    }
    vkx_image_serial_refs_drop_all(&t->refs, &t->serial_refs);
}
static void stor_unlink(vtex *t) { for (vtex **p = &g_stab[t->khash]; *p; p = &(*p)->next) if (*p == t) { *p = t->next; return; } }
static void pc_drop(const vtex *t);
static void tex_retire(vtex *t) { pc_drop(t); t->release_after = atomic_load(&g_fence) + 1; t->next = g_dead; g_dead = t; g_dead_bytes += t->bytes; }   // the submission in progress may use it too
void vkr_tex_gpu_write(vtex *t) { if (t) t->gpu_owned = 1; }
static uint64_t g_shared_hits, g_taken;
static void image_stats(uint64_t *cpu, uint64_t *gpu, uint64_t *empty, uint64_t *stale) {
    *cpu = *gpu = *empty = *stale = 0;
    for (int h = 0; h < 4096; h++) for (vtex *t = g_stab[h]; t; t = t->next) {
        if (t->gpu_owned) *gpu += t->bytes;
        else if (t->uploaded) *cpu += t->bytes;
        else *empty += t->bytes;
        if (!t->gpu_owned && t->uploaded && g_subs > t->last_sample + 256) *stale += t->bytes;
    }
}
static void reap(void) {   // images released once the GPU is past their last use (under g_tmu)
    uint64_t done = done_value();
    for (vtex **p = &g_dead; *p;) {
        vtex *t = *p;
        if (t->release_after <= done) { *p = t->next; for (vkview *v = t->views, *nv; v; v = nv) { nv = v->next; vkDestroyImageView(g_vkd.dev, v->view, NULL); free(v); } vkDestroyImage(g_vkd.dev, t->img, NULL); vkFreeMemory(g_vkd.dev, t->mem, NULL); g_dead_bytes -= t->bytes; atomic_fetch_sub(&g_image_bytes, t->bytes); if (t->speculative) atomic_fetch_sub(&g_spec_image_bytes, t->bytes); free(t); g_live--; }
        else p = &t->next;
    }
}
static void retire_overlaps(const uint8_t *kmem, const uint8_t *kend) {
    if (!kmem) return;
    for (int i = 0; i < 4096; i++) for (vtex **p = &g_stab[i]; *p;) { vtex *s = *p;
        if (!s->refs && s->kmem && s->kmem < kend && kmem < s->kend) { *p = s->next; tex_retire(s); g_taken++; }
        else p = &s->next;
    }
}
static void retire_unused_speculative(void) {
    for (int i = 0; i < 4096; i++) for (vtex **p = &g_stab[i]; *p;) { vtex *s = *p;
        if (!s->refs && s->speculative && !s->gpu_owned) { *p = s->next; tex_retire(s); g_taken++; }
        else p = &s->next;
    }
}
static _Thread_local uint64_t t_spec_budget, t_spec_max;
static void add_view_format(VkFormat *views, uint32_t *n, VkFormat f) {
    if (f == VK_FORMAT_UNDEFINED) return;
    for (uint32_t i = 0; i < *n; i++) if (views[i] == f) return;
    views[(*n)++] = f;
}
// The image for a texture, made the first time it is needed (under g_tmu). NULL for what is not made (logged once per kind). Layers as nvn_mtl.m counts them: an array's and a
// cube array's depth (a cube array's counts layer-faces), 6 for a cube, a 1D array's height; 1D textures are one-row 2D images
static vtex *tex_get(const tex_t *o) {
    if (!o || !o->serial) return NULL;
    vtex *t = tex_find(o->serial); if (t) return t;
    const nvn_texdesc *d = &o->d;
    if (d->target == NVN_TEXTURE_TARGET_BUFFER || d->target < 0 || d->target > NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) {
        static int warned[16]; if (d->target >= 0 && d->target < 16 && !warned[d->target]++) LOG("[RED] vulkan replay: texture target %d is not an image (serial %llu)", d->target, (unsigned long long)o->serial);
        return NULL;
    }
    VkFormat f = vkf_from_nvn(d->format, NULL, NULL);
    if (!f) { static int warned[256]; if (!warned[d->format & 255]++) LOG("[RED] vulkan replay: NVN format %#x has no Vulkan format here", d->format); return NULL; }
    VkImageAspectFlags asp = vkf_aspect(f);
    uint32_t kh = stor_hash(o->mem, d);
    const uint8_t *kmem = o->mem, *kend = kmem ? kmem + (o->L.size ? o->L.size : 1) : NULL;
    if (kmem) for (vtex *s = g_stab[kh]; s; s = s->next) if (s->kmem == kmem && !memcmp(&s->kd, d, sizeof *d)) {
        sref_add(o->serial, s); g_shared_hits++; return s;
    }
    /* Match the mature Metal bindless pool's pool_unwritten_target rule.
     * COMPRESSIBLE textures are GPU targets on this title; before a pass has
     * written one there is no image content for a shader to sample.  A direct
     * attachment/storage path creates it without t_spec_budget, after which
     * the existing-serial/storage lookups above return that real image.  Do
     * not spend speculative residency on empty/transient targets. */
    if ((t_spec_budget || t_spec_max) && (!kmem || (d->flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE))) {
        static _Atomic uint64_t deferred; uint64_t n = atomic_fetch_add(&deferred, 1) + 1;
        if (!(n & (n - 1))) LOG("vulkan replay: deferred %llu unwritten/empty bindless targets from speculative residency",
                               (unsigned long long)n);
        return NULL;
    }
    /* Released texture objects retain their storage-backed image so the next
     * object over the same bytes can take it over.  Reclaim those dead images
     * before testing the speculative budget: doing the budget check first
     * leaves a full title-screen pool permanently unable to admit the next
     * scene's textures.  Retired images remain alive until their GPU fence. */
    retire_overlaps(kmem, kend);
    reap();
    /* The bindless population budget applies to images this path creates
     * speculatively for their whole lifetime.  A later direct use or GPU write
     * does not free the allocation, so it must not free budget either.  This
     * keeps the title-to-scene transition bounded while overlap retirement
     * above lets the scene replace released title images.  Images first made
     * by a direct path are non-speculative and never enter this accounting.
     * Avoid create/destroy churn once the speculative pool itself is full; the
     * exact requirement is checked again below after memory requirements. */
    if (t_spec_budget && atomic_load(&g_spec_image_bytes) >= t_spec_budget) {
        retire_unused_speculative();
        reap();
        if (atomic_load(&g_spec_image_bytes) >= t_spec_budget) return NULL;
    }
    int tg = d->target;
    uint32_t layers = tg == NVN_TEXTURE_TARGET_2D_ARRAY || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY || tg == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY ? (d->depth ? d->depth : 1)
                    : tg == NVN_TEXTURE_TARGET_CUBEMAP ? 6 : tg == NVN_TEXTURE_TARGET_1D_ARRAY ? (d->height ? d->height : 1) : 1;
    uint32_t h = tg == NVN_TEXTURE_TARGET_1D || tg == NVN_TEXTURE_TARGET_1D_ARRAY ? 1 : (d->height ? d->height : 1);
    uint32_t depth = tg == NVN_TEXTURE_TARGET_3D ? (d->depth ? d->depth : 1) : 1;
    uint32_t samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vkd.pd, f, &fp);
    VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
    VkImageUsageFlags use = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) use |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) use |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) use |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1) use |= VK_IMAGE_USAGE_STORAGE_BIT;
    // NVN image operations may address any compatible 32-bit colour texel as
    // an R32 integer word. Extended usage lets such a view supply storage and
    // atomic support even when the base colour format does not.
    VkFormat word_views[2] = {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    if (samples == 1) for (int i = 0; i < 2; i++) {
        VkFormat wf = i ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        if (!vkr_word_view_compatible(f, wf)) continue;
        VkFormatProperties ip; vkGetPhysicalDeviceFormatProperties(g_vkd.pd, wf, &ip);
        if (ip.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
            word_views[i] = wf;
            use |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
    }
    // a 3D image that can be a render target is rendered slice by slice through 2D-array views of it (2D_ARRAY_COMPATIBLE)
    VkImageCreateFlags cf = (tg == NVN_TEXTURE_TARGET_CUBEMAP || tg == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) && layers % 6 == 0 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    if (tg == NVN_TEXTURE_TARGET_3D && (use & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) cf |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkFormat views[6] = {f}; uint32_t nviews = 1;
    add_view_format(views, &nviews, vkr_srgb_twin(f));
    if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        add_view_format(views, &nviews, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
    }
    add_view_format(views, &nviews, word_views[0]);
    add_view_format(views, &nviews, word_views[1]);
    VkImageFormatListCreateInfo fl = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = nviews, .pViewFormats = views};
    if (nviews > 1) cf |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;   // compatible view formats can supply their own usage features
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = nviews > 1 ? &fl : NULL, .flags = cf,
        .imageType = tg == NVN_TEXTURE_TARGET_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, .format = f, .extent = {d->width ? d->width : 1, h, depth},
        .mipLevels = d->levels ? d->levels : 1, .arrayLayers = layers, .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = use, .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    t = calloc(1, sizeof *t);
    VkResult r = vkCreateImage(g_vkd.dev, &ci, NULL, &t->img);
    if (r) { LOG("[RED] vulkan replay: image %ux%ux%u format %d (NVN %#x, target %d) not made (%d)", ci.extent.width, h, depth, (int)f, d->format, tg, (int)r); free(t); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vkd.dev, t->img, &mr);
    uint64_t used = atomic_load(&g_spec_image_bytes);
    if ((t_spec_max && mr.size > t_spec_max) ||
        (t_spec_budget && (used >= t_spec_budget || mr.size > t_spec_budget - used))) {
        static _Atomic uint64_t rejected; uint64_t n = atomic_fetch_add(&rejected, 1) + 1;
        if (!(n & (n - 1)))
            LOG("vulkan replay: speculative bindless image rejected by strict budget: %ux%ux%u, %u layers, %u levels, %.0f MiB (%.0f MiB speculative; reject %llu)",
                ci.extent.width, h, depth, layers, ci.mipLevels, mr.size / 1048576.0,
                used / 1048576.0, (unsigned long long)n);
        vkDestroyImage(g_vkd.dev, t->img, NULL); free(t); return NULL;
    }
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < g_vkd.mem.memoryTypeCount && mt == UINT32_MAX; i++) if ((mr.memoryTypeBits >> i & 1) && (g_vkd.mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) mt = i;
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt};
    if (mt == UINT32_MAX || vkAllocateMemory(g_vkd.dev, &ma, NULL, &t->mem) || vkBindImageMemory(g_vkd.dev, t->img, t->mem, 0)) {
        LOG("[RED] vulkan replay: no memory for image %ux%u", ci.extent.width, h); vkDestroyImage(g_vkd.dev, t->img, NULL); free(t); return NULL; }
    t->serial = o->serial; t->fmt = f; t->aspect = asp; t->bytes = mr.size; t->speculative = t_spec_budget || t_spec_max; t->w = ci.extent.width; t->h = h; t->d = depth; t->layers = layers; t->levels = ci.mipLevels; t->samples = samples;
    t->target = tg; t->usage = use; t->fresh = 1;
    g_live++; atomic_fetch_add(&g_image_bytes, t->bytes); if (t->speculative) atomic_fetch_add(&g_spec_image_bytes, t->bytes);
    t->kmem = kmem; t->kend = kend; t->kd = *d; t->khash = kh; t->pool = o->pool;
    t->next = g_stab[kh]; g_stab[kh] = t;
    sref_add(o->serial, t);
    if (g_open) tex_ready(up_cb(), t);   // into GENERAL before the submission runs: never inside a render pass
    { static int tl = -1; if (tl < 0) tl = getenv("DM_VK_TEXLOG") != NULL;   // diagnostic: each image made, with its guest storage
      if (tl) LOG("[tex] serial %llu target %d %ux%ux%u layers %u levels %u NVN %#x storage %p..%p (layer %llu bytes) pool %p off %lld", (unsigned long long)o->serial, tg, t->w, t->h, depth, layers, t->levels, d->format,
                  (void *)o->mem, (void *)(o->mem + o->L.size), (unsigned long long)o->L.layer_size, (void *)o->pool, (long long)o->off); }
    static uint64_t n; n++; if (!(n & (n - 1))) { uint64_t cpu, gpu, empty, stale; image_stats(&cpu, &gpu, &empty, &stale);
        LOG("vulkan replay: %llu images made, %llu live, %llu texture objects given a shared one, %llu taken over; image memory CPU-source %.0f MiB, GPU-owned %.0f MiB, empty %.0f MiB, retired-pending %.0f MiB, CPU-source idle >256 submits %.0f MiB (latest %ux%ux%u format %d, NVN %#x, target %d, %u layers, %u levels)",
            (unsigned long long)n, (unsigned long long)g_live, (unsigned long long)g_shared_hits, (unsigned long long)g_taken,
            cpu / 1048576.0, gpu / 1048576.0, empty / 1048576.0, g_dead_bytes / 1048576.0, stale / 1048576.0,
            t->w, t->h, depth, (int)f, d->format, tg, layers, t->levels); }
    return t;
}
// Into VK_IMAGE_LAYOUT_GENERAL the first time it is used (under g_tmu)
static void tex_ready(VkCommandBuffer c, vtex *t) {
    if (!t->fresh) return;
    VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = t->img, .subresourceRange = {t->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    t->fresh = 0;
}

// ---- loading textures from guest memory ----
// A texture the GPU has not written is loaded from its storage the first time it is sampled, and again when the game has written the storage since (nvn_dirty.c write-protects
// it after each load). The block-linear storage is deswizzled on the CPU, level by level and layer by layer, into a staging buffer, and copied to the image by the submission's
// upload command buffer, which goes to the queue right before the submission's own (that one starts with a full barrier). Staging buffers are freed once the GPU is past them
typedef struct stage_buf { VkBuffer buf; VkDeviceMemory mem; uint64_t after; struct stage_buf *next; } stage_buf;
static stage_buf *g_stage;   // waiting to be freed (under g_tmu)
static uint64_t g_uploads, g_reuploads, g_up_bytes;
static uint32_t pick_mem(uint32_t bits, VkMemoryPropertyFlags want);
static VkCommandBuffer up_cb(void) {   // the upload command buffer of the submission in progress, begun on first use
    if (!g_up_open) {
        vkResetCommandBuffer(g_up[g_cb_i], 0);
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(g_up[g_cb_i], &bi); g_up_open = 1;
    }
    return g_up[g_cb_i];
}
static _Atomic uint64_t g_up_ns;   // time spent loading textures (the long-frame log)
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_uploads; *reuploads = g_reuploads; }
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_up_ns, 0); }
uint64_t mtl_upload_bytes(void) { return g_up_bytes; }
static int upload_(vtex *t, const tex_t *o);
static int upload(vtex *t, const tex_t *o) { uint64_t t0 = mono_ns(); int r = upload_(t, o); atomic_fetch_add(&g_up_ns, mono_ns() - t0); return r; }
static int upload_(vtex *t, const tex_t *o) {   // under g_tmu, inside a submission
    const nvn_texdesc *d = &o->d;
    if (!(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || t->samples > 1 || !o->mem) return 0;   // depth / stencil and multisampled content is the GPU's
    if ((d->flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && !o->has_layout && o->pool &&
        (o->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) return 0;   // an unwritten GPU target, not CPU texture data
    if (o->pool && o->pool->vp) { static int w; if (!w++) LOG("[RED] vulkan replay: a texture in a sparse pool is not loaded yet"); return 0; }
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb) { static int w[256]; if (!w[d->format & 255]++) LOG("[RED] vulkan replay: NVN format %#x has no block size: not loaded", d->format); return 0; }
    int linear = (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;   // rows d->stride apart, as nvn_deswizzle_geom reads the flags
    size_t lstride = d->stride ? d->stride : 0;
    // A linear texture loaded before takes only the rows on pages the game wrote since (its rows map straight onto pages): a glyph cache that grows by a few rows a
    // frame copied whole was 16 MB a time (Mario Wonder's 2048x8192 R8 at about 4 a second)
    int partial = linear && t->uploaded && t->levels == 1 && t->layers == 1 && t->d == 1;
    uint64_t prev = t->stamp;
    size_t total = 0;
    for (uint32_t l = 0; l < t->levels; l++) {
        uint32_t w = t->w >> l ? t->w >> l : 1, h = t->h >> l ? t->h >> l : 1, z = t->d >> l ? t->d >> l : 1;
        total += (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * z * t->layers;
    }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    stage_buf *s = calloc(1, sizeof *s);
    if (vkCreateBuffer(g_vkd.dev, &bi, NULL, &s->buf)) { free(s); return 0; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, s->buf, &mr);
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
        .memoryTypeIndex = pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    void *map = NULL;
    if (ma.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_vkd.dev, &ma, NULL, &s->mem) || vkBindBufferMemory(g_vkd.dev, s->buf, s->mem, 0) || vkMapMemory(g_vkd.dev, s->mem, 0, VK_WHOLE_SIZE, 0, &map)) {
        LOG("[RED] vulkan replay: no staging memory for a %zu byte texture", total); vkDestroyBuffer(g_vkd.dev, s->buf, NULL); if (s->mem) vkFreeMemory(g_vkd.dev, s->mem, NULL); free(s); return 0; }
    t->stamp = nvn_dirty_now(); nvn_dirty_protect(o->mem, o->L.size);   // stamped before reading: a write during the copy makes it dirty again
    size_t pg = (size_t)getpagesize(), part_bytes = 0;
    VkCommandBuffer c = up_cb(); tex_ready(c, t);
    VkBufferImageCopy rg[256]; int nr = 0; size_t off = 0; int bad = 0;
    for (uint32_t l = 0; l < t->levels && !bad; l++) {
        uint32_t w = t->w >> l ? t->w >> l : 1, h = t->h >> l ? t->h >> l : 1, z = t->d >> l ? t->d >> l : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
        for (uint32_t y = 0; y < t->layers && !bad; y++) {
            uint8_t *dst = (uint8_t *)map + off;
            if (linear) {   // linear storage: rows d->stride apart (one level, one layer)
                if (l || y) { bad = 1; break; }
                size_t st = lstride ? lstride : pitch, grp = pg / st ? pg / st : 1;   // rows per page
                for (size_t r0 = 0; r0 < rows * z; r0 += grp) {
                    size_t n = rows * z - r0 < grp ? rows * z - r0 : grp;
                    if (partial && !nvn_dirty_since(o->mem + r0 * st, (n - 1) * st + pitch, prev)) continue;
                    for (size_t r = r0; r < r0 + n; r++) memcpy(dst + r * pitch, o->mem + r * st, pitch);
                    if (partial) {
                        if (nr == 256) {
                            vkCmdCopyBufferToImage(c, s->buf, t->img, VK_IMAGE_LAYOUT_GENERAL,
                                (uint32_t)nr, rg);
                            nr = 0;
                        }
                        rg[nr++] = (VkBufferImageCopy){.bufferOffset = off + r0 * pitch, .bufferRowLength = (uint32_t)(pitch / bpb * bw), .bufferImageHeight = (uint32_t)(n * bh),
                            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageOffset = {0, (int32_t)(r0 * bh), 0}, .imageExtent = {w, (uint32_t)(n * bh) < h - r0 * bh ? (uint32_t)(n * bh) : h - (uint32_t)(r0 * bh), 1}};
                        part_bytes += n * pitch;
                    }
                }
                if (partial) { off += pitch * rows * z; continue; }
            } else if (nvn_deswizzle(d, l, y, o->mem, dst, pitch)) {
                static int w2[256]; if (!w2[d->format & 255]++) LOG("[RED] vulkan replay: deswizzle failed (NVN format %#x, target %d)", d->format, d->target);
                bad = 1; break;
            }
            if (nr == 256) {
                vkCmdCopyBufferToImage(c, s->buf, t->img, VK_IMAGE_LAYOUT_GENERAL,
                    (uint32_t)nr, rg);
                nr = 0;
            }
            rg[nr++] = (VkBufferImageCopy){.bufferOffset = off, .bufferRowLength = (uint32_t)(pitch / bpb * bw), .bufferImageHeight = (uint32_t)(rows * bh),
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, t->d > 1 ? 0 : y, 1}, .imageExtent = {w, h, z}};
            off += pitch * rows * z;
        }
    }
    if (nr) vkCmdCopyBufferToImage(c, s->buf, t->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nr, rg);
    s->after = atomic_load(&g_fence) + 1; s->next = g_stage; g_stage = s;
    if (partial) off = part_bytes;
    if (t->uploaded) g_reuploads++; else g_uploads++;
    t->uploaded = 1; g_up_bytes += off;
    if (g_dump_left > 0 && g_ndump < 64 && !bad) { g_dump_ser[g_ndump++] = t->serial; g_dump_left--; }
    if (!((g_uploads + g_reuploads) & (g_uploads + g_reuploads - 1)))
        LOG("vulkan replay: textures loaded from guest memory: %llu, again after a write %llu, %llu MiB (latest %ux%u NVN %#x target %d, %u levels %u layers%s)", (unsigned long long)g_uploads,
            (unsigned long long)g_reuploads, (unsigned long long)(g_up_bytes >> 20), t->w, t->h, d->format, d->target, t->levels, t->layers, linear ? ", linear" : "");
    return !bad;
}
// A texture about to be sampled: its image, loaded from guest memory first if the GPU does not own its content and it was not loaded or was written since (under g_tmu)
static vtex *tex_sampled(const tex_t *o) {
    vtex *t = tex_get(o); if (!t) return NULL;
    t->last_sample = g_subs;
    if (t->gpu_owned) return t;
    // Every write to tracked memory moves the dirty clock: a texture found clean at clock N is still clean while the clock reads N. Checked page by page on every bind
    // (with a scan of the tracked ranges to find its own), this was 28% of the host's CPU time in Mario Wonder's levels
    uint64_t now = nvn_dirty_now();
    if (t->uploaded && t->clean_at == now) return t;
    if (!t->span) t->span = nvn_dirty_span(o->mem, o->L.size);
    if (!t->uploaded || nvn_dirty_since_span(t->span, o->mem, o->L.size, t->stamp)) upload(t, o); else t->clean_at = now;
    return t;
}

void vkr_gmem_invalidate_pages(const char *key, const uint64_t *pages, size_t n, int overflow, uint32_t stores) {
    static uint64_t calls, overflows, matched_pages, invalidated, pair_logs;
    if (overflow) {
        overflows++; if (overflows <= 8 || !(overflows & (overflows - 1)))
            LOG("[RED] vulkan raw-global trace: shader %s logged %u map-entry winners, over the %u-slot bound; no image invalidated (%llu overflows)",
                key, stores, VKX_GMEM_TRACE_SLOTS, (unsigned long long)overflows);
        return;
    }
    uint64_t overlaps = 0, changed = 0;
    pthread_mutex_lock(&g_tmu);
    for (size_t p = 0; p < n; p++) {
        int page_hit = 0;
        for (int h = 0; h < 4096; h++) for (vtex *t = g_stab[h]; t; t = t->next) {
            if (!t->kmem || !vkx_gmem_page_overlaps(pages[p], (uintptr_t)t->kmem, (uintptr_t)t->kend)) continue;
            overlaps++; page_hit = 1;
            if (!t->uploaded && !t->gpu_owned) continue;
            t->uploaded = 0; t->gpu_owned = 0; t->clean_at = 0; changed++; invalidated++;
            if (pair_logs++ < 32) {
                uint64_t a = pages[p] << VKX_GMEM_PAGE_SHIFT;
                LOG("vulkan raw-global trace: shader %s page %#llx..%#llx overlaps image serial %llu storage %p..%p; only that image is invalidated",
                    key, (unsigned long long)a, (unsigned long long)(a + (1u << VKX_GMEM_PAGE_SHIFT)), (unsigned long long)t->serial, t->kmem, t->kend);
            }
        }
        matched_pages += page_hit;
    }
    pthread_mutex_unlock(&g_tmu);
    calls++;
    if ((stores && calls <= 8) || (stores && !(calls & (calls - 1))))
        LOG("vulkan raw-global trace: shader %s, %u map-entry winners, %zu unique guest pages, %llu image overlaps, %llu newly invalidated (%llu pages matched / %llu images invalidated total)",
            key, stores, n, (unsigned long long)overlaps, (unsigned long long)changed, (unsigned long long)matched_pages, (unsigned long long)invalidated);
}
static void stage_reap(void) {   // under g_tmu
    uint64_t done = done_value();
    for (stage_buf **p = &g_stage; *p;) { stage_buf *s = *p; if (s->after <= done) { *p = s->next; vkDestroyBuffer(g_vkd.dev, s->buf, NULL); vkFreeMemory(g_vkd.dev, s->mem, NULL); free(s); } else p = &s->next; }
}

// ---- descriptor pools: texture and sampler IDs (mtl_reg_tex / mtl_reg_samp) ----
typedef struct { int ok, has_v; uint64_t registered_at; tex_t t; view_t v; } treg_t;
static treg_t *g_treg; static int g_ntreg; static samp_par *g_sreg; static uint8_t *g_sreg_ok; static int g_nsreg;
static int *g_tdirty, g_ntdirty, g_ctdirty, *g_sdirty, g_nsdirty, g_csdirty;
static uint8_t g_tdflag[VKR_BINDLESS_TEXTURES], g_sdflag[VKR_BINDLESS_SAMPLERS];
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
static void reg_dirty(int id, int limit, int **list, int *n, int *cap, uint8_t *flag, const char *kind) {
    if (id <= 0) return;
    if (id >= limit) {
        static uint8_t tex_warned, samp_warned;
        uint8_t *warned = kind[0] == 't' ? &tex_warned : &samp_warned;
        if (!*warned) {
            *warned = 1;
            LOG("[RED] vulkan draws: bindless %s registration ID %d exceeds the %d-entry pool; element dropped", kind, id, limit);
        }
        return;
    }
    if (flag[id]) return;
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 256; *list = realloc(*list, (size_t)*cap * sizeof **list); }
    flag[id] = 1; (*list)[(*n)++] = id;
}
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (id < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id >= g_ntreg) { int n = id + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    treg_t *r = &g_treg[id]; r->ok = t != NULL; r->registered_at = atomic_load(&g_subs); if (t) r->t = *t; r->has_v = v != NULL; if (v) r->v = *v;
    reg_dirty(id, VKR_BINDLESS_TEXTURES, &g_tdirty, &g_ntdirty, &g_ctdirty, g_tdflag, "texture");
    pthread_mutex_unlock(&g_rmu);
}
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id >= g_nsreg) { int n = id + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); g_nsreg = n; }
    g_sreg[id] = *p; g_sreg_ok[id] = 1;
    reg_dirty(id, VKR_BINDLESS_SAMPLERS, &g_sdirty, &g_nsdirty, &g_csdirty, g_sdflag, "sampler");
    pthread_mutex_unlock(&g_rmu);
}
static int treg_get(int id, treg_t *out) { pthread_mutex_lock(&g_rmu); int ok = id > 0 && id < g_ntreg && g_treg[id].ok; if (ok) *out = g_treg[id]; pthread_mutex_unlock(&g_rmu); return ok; }

/* The bindless descriptor table is persistent, while this title registers a
 * very large title-screen generation and leaves those texture objects alive
 * when its first mesh scene registers the next generation.  Overlap-only
 * reclamation therefore cannot make room.  Evict a bounded batch of old,
 * CPU-backed speculative images and report every still-current registration
 * that points at them.  vk_draw replaces those descriptors with dummies
 * before recording more work and queues the IDs for its normal retry path.
 * The images themselves remain alive behind tex_retire's timeline fence. */
int vkr_bindless_evict_old_speculative(uint64_t registered_before, int *ids,
                                       int cap, uint64_t *bytes) {
    int n = 0, victims = 0;
    if (bytes) *bytes = 0;
    if (!ids || cap <= 0 || !registered_before) return 0;
    pthread_mutex_lock(&g_tmu);
    pthread_mutex_lock(&g_rmu);
    for (int h = 0; h < 4096 && victims < 32; h++) for (vtex **p = &g_stab[h]; *p && victims < 32;) {
        vtex *t = *p;
        if (!t->speculative || t->gpu_owned) { p = &t->next; continue; }
        int first = n, fits = 1, recent = 0;
        for (int id = 1; id < g_ntreg; id++) if (g_treg[id].ok && tex_find(g_treg[id].t.serial) == t) {
            if (g_treg[id].registered_at >= registered_before) recent = 1;
            if (n == cap) { fits = 0; break; }
            ids[n++] = id;
        }
        if (!fits || recent) { n = first; p = &t->next; continue; }
        *p = t->next;
        /* This image is no longer a storage-table candidate.  In particular,
         * a presentation hold must retire it when released instead of leaving
         * an unlinked allocation pinned by its old guest address forever. */
        t->kmem = t->kend = NULL; t->pool = NULL;
        sref_drop_all(t);
        if (!t->refs) tex_retire(t);   // a presentation hold retires it in vkr_present_release
        if (bytes) *bytes += t->bytes;
        victims++;
    }
    pthread_mutex_unlock(&g_rmu);
    pthread_mutex_unlock(&g_tmu);
    return n;
}
// ---- textures the CPU writes after the replay first loaded them ----
// A texture a draw names by a bind (OP_BIND_TEX) or a per-stage binding is checked against the dirty clock each time (tex_sampled). One sampled through the
// bindless pool (set 1) is loaded once, when its ID is registered, and was never looked at again: Ocarina of Time registers its dialogue glyph cache
// (2048x8192 R8, LINEAR, rasterized by the font engine on the CPU) while it is still blank and samples it only through the pool, so every message window was
// empty while the HUD's directly bound text drew. Two sources, both applied on the replay thread before a submission's first draw (vkr_cpu_written_refresh):
//  - mtl_tex_cpu_written (hle_nvn.c: WriteTexels, a flushed or polled range over a texture's storage), which this host answered with an empty stub;
//  - LINEAR textures registered in the pool, which hle_nvn.c leaves out of that tracking because Metal maps them over the pool's bytes: here they are images
//    of their own, so each is checked against the dirty clock when it has moved (upload_ re-uploads only the rows of the pages written).
static pthread_mutex_t g_cwmu = PTHREAD_MUTEX_INITIALIZER; static tex_t *g_cw; static size_t g_cwn, g_cwcap; static uint64_t g_cw_refreshed, g_cw_linear;
void mtl_tex_cpu_written(const tex_t *t) {   // on a guest thread
    if (g_on <= 0 || !t || !t->serial) return;
    pthread_mutex_lock(&g_cwmu);
    size_t i = 0; while (i < g_cwn && g_cw[i].serial != t->serial) i++;
    if (i == g_cwn) { if (g_cwn == g_cwcap) { size_t n = g_cwcap ? g_cwcap * 2 : 64; tex_t *q = realloc(g_cw, n * sizeof *q); if (q) { g_cw = q; g_cwcap = n; } }
        if (g_cwn < g_cwcap) g_cwn++; else i = g_cwn; }
    if (i < g_cwn) g_cw[i] = *t;
    pthread_mutex_unlock(&g_cwmu);
}
void vkr_cpu_written_refresh(void) {   // on the replay thread, the submission's command buffer open
    enum { LIN = 64 };
    tex_t *w = NULL; size_t nw = 0; tex_t lin[LIN]; int nl = 0;
    pthread_mutex_lock(&g_cwmu); if (g_cwn) { w = malloc(g_cwn * sizeof *w); if (w) { memcpy(w, g_cw, g_cwn * sizeof *w); nw = g_cwn; g_cwn = 0; } } pthread_mutex_unlock(&g_cwmu);
    static uint64_t clock_seen; uint64_t now = nvn_dirty_now();
    if (now != clock_seen) { clock_seen = now;
        pthread_mutex_lock(&g_rmu);
        for (int id = 1; id < g_ntreg && nl < LIN; id++) if (g_treg[id].ok && g_treg[id].t.mem && (g_treg[id].t.d.flags & NVN_TEXTURE_FLAGS_LINEAR)) lin[nl++] = g_treg[id].t;
        pthread_mutex_unlock(&g_rmu); }
    if (!nw && !nl) return;
    uint64_t before = g_cw_refreshed + g_cw_linear;
    pthread_mutex_lock(&g_tmu);
    for (size_t i = 0; i < nw; i++) { vtex *t = tex_find(w[i].serial); if (t && !t->gpu_owned && t->uploaded) { t->clean_at = 0; upload(t, &w[i]); g_cw_refreshed++; } }
    for (int i = 0; i < nl; i++) { vtex *t = tex_find(lin[i].serial);
        if (t && !t->gpu_owned && t->uploaded) { uint64_t st = t->stamp; tex_sampled(&lin[i]); if (t->stamp != st) g_cw_linear++; } }
    pthread_mutex_unlock(&g_tmu);
    free(w);
    uint64_t n = g_cw_refreshed + g_cw_linear;
    if (n != before && (!(n & (n - 1)) || (before ^ n) >> 10)) LOG("vulkan replay: %llu textures re-uploaded after a CPU write (%llu reported by the game's writes and flushes, %llu LINEAR ones sampled through the pool)",
        (unsigned long long)n, (unsigned long long)g_cw_refreshed, (unsigned long long)g_cw_linear);
}
int vkr_bindless_recent_tex_ids(uint64_t registered_after, int *ids, int cap) {
    int n = 0;
    if (!ids || cap <= 0) return 0;
    pthread_mutex_lock(&g_rmu);
    for (int id = 1; id < g_ntreg && n < cap; id++)
        if (g_treg[id].ok && g_treg[id].registered_at >= registered_after) ids[n++] = id;
    pthread_mutex_unlock(&g_rmu);
    return n;
}
static uint64_t g_tex_unreg;
// Texture binds: each bound texture is made ready to sample now (its draws are not translated yet)
static void op_bind_tex(uint32_t op, const void *a) {
    p_bind_handles t1; const p_bind_handles *b = bind_rec(op, a, &t1);
    if (b->n < 0 || b->n > 32) return;
    uint64_t h[32]; memcpy(h, b->h, 8 * (size_t)b->n);
    if (b->defer && b->src) memcpy(h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);   // BindTexturesDeferred: the handles as they are at execution
    for (int k = 0; k < b->n; k++) {
        int tid = (int)(h[k] & 0xfffff); if (!tid) continue;
        treg_t r; if (!treg_get(tid, &r)) { g_tex_unreg++; continue; }
        pthread_mutex_lock(&g_tmu); tex_sampled(&r.t); pthread_mutex_unlock(&g_tmu);
    }
}

// For the window's present thread: the image of a presented texture, made ready in c if new
void *vkr_present_acquire(const tex_t *o) {
    pthread_mutex_lock(&g_tmu);
    vtex *t = o ? tex_find(o->serial) : NULL; if (t) vkx_image_hold_add(&t->refs);
    pthread_mutex_unlock(&g_tmu);
    return t;
}
void vkr_present_release(void *held) {
    if (!held) return;
    pthread_mutex_lock(&g_tmu);
    vtex *t = held;
    if (vkx_image_hold_drop_retires(&t->refs, t->kmem)) { stor_unlink(t); tex_retire(t); }
    pthread_mutex_unlock(&g_tmu);
}
int vkr_present_image(const tex_t *o, void *held, VkCommandBuffer c, VkImage *img, uint32_t *w, uint32_t *h, VkFormat *fmt) {
    pthread_mutex_lock(&g_tmu);
    vtex *t = held ? held : tex_get(o); if (t) { tex_ready(c, t); *img = t->img; *w = t->w; *h = t->h; *fmt = t->fmt; }
    pthread_mutex_unlock(&g_tmu);
    return t != NULL;
}
void mtl_tex_init(const tex_t *t) {
    if (!t || !t->serial) return;
    pthread_mutex_lock(&g_tmu);
    nvn_hm_put(&g_tex_live, &t->serial, (void *)t);
    pthread_mutex_unlock(&g_tmu);
}
int vkr_tex_gpu_owned(uint64_t serial) {
    pthread_mutex_lock(&g_tmu);
    vtex *t = tex_find(serial); int owned = t && t->gpu_owned;
    pthread_mutex_unlock(&g_tmu);
    return owned;
}
// nvnMemoryPoolFinalize (hle_nvn.c): the images built on the pool's storage. A released texture object leaves its storage-backed image in the table for
// the next object over the same bytes (mtl_tex_release), and only a new image overlapping those bytes retires it (retire_overlaps). Once the pool is
// finalized nothing can be built over its bytes again, so those images stayed for good: Ocarina of Time makes and finalizes pools continuously (16,384 in
// six minutes) and its render targets on dead pools grew the GPU-owned image memory from 1.2 to 3.0 GiB in five minutes of standing still. As nvn_mtl.m
// mtl_alias_drop_pool does: every image on the pool leaves the storage table and its serial references are dropped. An image still held by the
// presentation thread retires when that hold ends; every other image retires behind its replay fence now. A later pool at the same address holds other bytes.
static uint64_t g_pool_dropped, g_pool_dropped_bytes;
void mtl_alias_drop_pool(const void *pool) {
    if (!g_on || !pool) return;
    pthread_mutex_lock(&g_tmu);
    uint64_t n = 0, held = 0;
    for (int h = 0; h < 4096; h++) for (vtex **p = &g_stab[h]; *p;) {
        vtex *t = *p;
        if (t->pool != pool) { p = &t->next; continue; }
        *p = t->next;
        t->kmem = t->kend = NULL; t->pool = NULL; n++; g_pool_dropped_bytes += t->bytes;
        // A texture object released before the replay thread made the image leaves a stale serial reference (mtl_tex_init records nothing), so pool
        // finalization must drop every serial. A present in flight is a separate hold: it keeps the allocation until vkr_present_release has submitted
        // its copy, avoiding both early destruction and a second tex_retire of the same image.
        sref_drop_all(t);
        if (!t->refs) tex_retire(t); else held++;
    }
    if (n) { g_pool_dropped += n; uint64_t k = g_pool_dropped; if (!(k & (k - 1)))   // at 1, 2, 4, ... images
        LOG("vulkan replay: %llu images detached from finalized memory pools so far (%.0f MiB; %llu with this pool, %llu still held for presentation)", (unsigned long long)k, g_pool_dropped_bytes / 1048576.0, (unsigned long long)n, (unsigned long long)held); }
    pthread_mutex_unlock(&g_tmu);
}
void mtl_tex_release(tex_t *o) {
    if (!g_on || !o->serial) return;
    pthread_mutex_lock(&g_tmu);
    const tex_t *owner = nvn_hm_get(&g_tex_live, &o->serial);
    if (vkx_tex_release_is_copy(owner, o)) { pthread_mutex_unlock(&g_tmu); return; }
    nvn_hm_del(&g_tex_live, &o->serial);
    sref **p = &g_tab[o->serial % 4096];
    while (*p && (*p)->serial != o->serial) p = &(*p)->next;
    if (*p) { sref *r = *p; vtex *t = r->t; *p = r->next; free(r); vkx_image_serial_ref_drop(&t->refs, &t->serial_refs);
        if (!t->refs && !t->kmem) { stor_unlink(t); tex_retire(t); } }   // with storage it stays for the next object over it
    pthread_mutex_unlock(&g_tmu);
}

// ---- submissions ----
static void full_barrier(VkCommandBuffer c);
static void dump_uploads(uint64_t fence);
static void sub_open(void) {   // the next ring slot's command buffer, begun (its last use waited for)
    g_cb_i = (g_cb_i + 1) % g_cb_n;
    uint64_t submitted = atomic_load(&g_cb_val[g_cb_i]);
    wait_value_named(submitted, UINT64_MAX, "sub_open", g_cb_i);   // its last use is done
    uint64_t done = done_value();
    gp_run(done);   // its reports written while its timestamp queries still hold that use's times (they are reset below)
    if (g_qp || g_cs_qp) {
        pthread_mutex_lock(&g_tmu);
        gpu_harvest(g_cb_i); cs_gpu_harvest(g_cb_i, done);
        atomic_store(&g_cb_val[g_cb_i], 0);   // no old submission may publish this slot while it is recorded again
        if (g_cs_batch[g_cb_i].timeline || g_cs_batch[g_cb_i].submitted)
            LOG("[RED] vulkan compute GPU profile: slot %d was not retired after timeline %llu completed; disabling new samples in this reuse",
                g_cb_i, (unsigned long long)submitted);
        else
            g_cs_batch[g_cb_i].recording = 0;
        pthread_mutex_unlock(&g_tmu);
    } else atomic_store(&g_cb_val[g_cb_i], 0);
    vkResetCommandBuffer(g_cb[g_cb_i], 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(g_cb[g_cb_i], &bi); g_open = 1;
    if (g_sq) { vkCmdResetQueryPool(g_cb[g_cb_i], g_sq, (uint32_t)(g_cb_i * SEG_Q), SEG_Q); g_sq_n[g_cb_i] = 0; }
    if (g_qp) { vkCmdResetQueryPool(g_cb[g_cb_i], g_qp, (uint32_t)(2 * g_cb_i), 2); vkCmdWriteTimestamp(g_cb[g_cb_i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qp, (uint32_t)(2 * g_cb_i)); g_cb_frame[g_cb_i] = vkh_frame(); }
    if (g_cs_qp) vkCmdResetQueryPool(g_cb[g_cb_i], g_cs_qp, cs_query(g_cb_i, 0), 2 * CS_GPU_PER_SLOT);
    g_up_open = 0;
    full_barrier(g_cb[g_cb_i]);   // after everything before it on the queue: the upload command buffer, earlier submissions, presents
    vkx_sub_begin(g_cb_i, g_cb[g_cb_i], done);
    g_cb_mesh_seq[g_cb_i] = 0;
}
static uint64_t g_flush_draws, g_flushes, g_mesh_batch_draws; static int g_noflush = -1;   // draws recorded since the command buffer began; mid-submission flushes made
// DM_VK_MESH_FLUSH=N submits after N successful mesh draws.  The historical
// presence-only spelling (including an empty/invalid value) remains N=1 for
// the strict proof run; 0 or an unset variable leaves mesh draws to the normal
// draw/descriptor-pressure split.
static unsigned mesh_flush_every(void) {
    static int init; static unsigned every;
    if (init) return every;
    init = 1;
    const char *e = getenv("DM_VK_MESH_FLUSH");
    if (!e) return 0;
    char *end = NULL; unsigned long n = strtoul(e, &end, 10);
    if (!*e || end == e || *end || n > 1024) {
        every = 1;
        LOG("[RED] vulkan replay: DM_VK_MESH_FLUSH=%s is not 0..1024; using the legacy one-draw flush", e);
    } else {
        every = (unsigned)n;
    }
    if (every) LOG("vulkan replay: mesh batches flush after %u successful draw%s", every, every == 1 ? "" : "s");
    return every;
}
static uint64_t sub_submit(void) {   // the open command buffer (after its uploads) to the queue, signalling the next timeline value
    VkCommandBuffer c = g_cb[g_cb_i];
    pc_flush(c);   // a pending clear does not outlive its command buffer (every caller has ended the pass)
    uint64_t mesh_seq = g_cb_mesh_seq[g_cb_i];
    if (g_qp) { vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, (uint32_t)(2 * g_cb_i + 1)); g_cb_timed[g_cb_i] = 1; }
    if (mesh_seq && mesh_log_on())
        LOG("vulkan mesh submit: issue=%llu slot=%d vkEndCommandBuffer begin", (unsigned long long)mesh_seq, g_cb_i);
    VkResult er = vkEndCommandBuffer(c); g_open = 0;
    if (mesh_seq && mesh_log_on())
        LOG("vulkan mesh submit: issue=%llu slot=%d vkEndCommandBuffer=%d", (unsigned long long)mesh_seq, g_cb_i, (int)er);
    if (er) die("vulkan replay: vkEndCommandBuffer failed (%d)", (int)er);
    VkCommandBuffer cbs[2]; uint32_t ncb = 0;   // the uploads first, then the submission (which starts with a full barrier)
    if (g_up_open) { er = vkEndCommandBuffer(g_up[g_cb_i]); g_up_open = 0; if (er) die("vulkan replay: upload vkEndCommandBuffer failed (%d)", (int)er); cbs[ncb++] = g_up[g_cb_i]; }
    cbs[ncb++] = c;
    uint64_t v = atomic_load(&g_fence) + 1;
    VkTimelineSemaphoreSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &v};
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ts, .commandBufferCount = ncb, .pCommandBuffers = cbs, .signalSemaphoreCount = 1, .pSignalSemaphores = &g_tl};
    if (mesh_seq && mesh_log_on())
        LOG("vulkan mesh submit: issue=%llu slot=%d signal=%llu command_buffers=%u begin",
            (unsigned long long)mesh_seq, g_cb_i, (unsigned long long)v, ncb);
    pthread_mutex_lock(&g_vkd_qlock); VkResult r = vkQueueSubmit(g_vkd.q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_vkd_qlock);
    if (mesh_seq && mesh_log_on())
        LOG("vulkan mesh submit: issue=%llu slot=%d signal=%llu vkQueueSubmit=%d done=%llu",
            (unsigned long long)mesh_seq, g_cb_i, (unsigned long long)v, (int)r, (unsigned long long)done_value());
    if (r) die("vulkan replay: submit failed (%d)", (int)r);
    if (mesh_seq) atomic_store(&g_mesh_last_signal, v);
    pthread_mutex_lock(&g_tmu);
    if (g_cs_qp && vkr_cs_gpu_publish(&g_cs_batch[g_cb_i], v) < 0)
        LOG("[RED] vulkan compute GPU profile: slot %d could not publish its batch at timeline %llu; samples disabled for this slot",
            g_cb_i, (unsigned long long)v);
    atomic_store(&g_cb_val[g_cb_i], v);
    pthread_mutex_unlock(&g_tmu);
    atomic_store(&g_fence, v); g_flush_draws = 0; g_mesh_batch_draws = 0;
    return v;
}
void mtl_sub_begin(void) {
    if (!vkr_on()) return;
    if (g_noflush < 0) g_noflush = getenv("DM_VK_NOFLUSH") != NULL;   // diagnostic: one command buffer per submission, as before
    sub_open(); g_rc_skip = 0;
}
// A long submission goes to the GPU in parts: when enough draws are recorded and no render pass is open, what there is is submitted and recording goes on in the next
// command buffer, so the GPU works on a frame's first passes while the rest is replayed (one command buffer per submission left the GPU idle for the whole replay, about
// 10 ms of Mario Wonder's 24 ms frame, and the CPU idle while the GPU ran). Fences stay in order: each part signals the next timeline value
#define FLUSH_DRAWS 128
// DM_VK_PROF=1 (diagnostic): the replay's time by record kind (nvn_obj.h's OP_ numbers), mid-submission flushes (64) and submission ends (65), logged as milliseconds a frame
// every 300 frames: where the submit thread's time goes
static int g_prof = -1; static uint64_t g_opns[66], g_prof_f0;
static void prof_log(void) {
    uint64_t f = vkh_frame(); if (!g_prof_f0) g_prof_f0 = f; if (f < g_prof_f0 + 300) return;
    double nf = (double)(f - g_prof_f0), tot = 0; char b[700]; int o = 0; int idx[66]; for (int i = 0; i < 66; i++) { idx[i] = i; tot += g_opns[i]; }
    for (int i = 0; i < 66; i++) for (int j = i + 1; j < 66; j++) if (g_opns[idx[j]] > g_opns[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    for (int i = 0; i < 12 && g_opns[idx[i]] && o < (int)sizeof b - 32; i++) o += snprintf(b + o, sizeof b - o, " %d:%.2f", idx[i], g_opns[idx[i]] / 1e6 / nf);
    uint64_t dp[8]; vkx_prof_take(dp);
    LOG("vulkan replay profile, %.0f frames: %.2f ms a frame; by record (ms a frame):%s; draws: shaders + sets %.2f (set writes listed %.2f, sets made %.2f; %.0f cache hits, %.0f misses a frame), state %.2f, vertex %.2f, call %.2f",
        nf, tot / 1e6 / nf, b, dp[0] / 1e6 / nf, dp[4] / 1e6 / nf, dp[5] / 1e6 / nf, dp[6] / nf, dp[7] / nf, dp[1] / 1e6 / nf, dp[2] / 1e6 / nf, dp[3] / 1e6 / nf);
    memset(g_opns, 0, sizeof g_opns); g_prof_f0 = f;
}
static void vkr_flush(void) { uint64_t t0 = g_prof > 0 ? mono_ns() : 0; sub_submit(); uint64_t t1 = g_prof > 0 ? mono_ns() : 0; g_flushes++; sub_open(); if (g_prof > 0) { g_opns[60] += t1 - t0; g_opns[61] += mono_ns() - t1; } }   // DM_VK_PROF: 60 the submit, 61 the next part's start
uint64_t mtl_sub_commit(void) {
    if (!g_on) return atomic_fetch_add(&g_fence, 1) + 1;
    uint64_t ct = g_prof > 0 ? mono_ns() : 0;
    rp_end(g_cb[g_cb_i]);
    uint64_t v = sub_submit(); g_subs++;
    if (g_prof > 0) { g_opns[65] += mono_ns() - ct; prof_log(); }
    static int opslog = -1; if (opslog < 0) opslog = getenv("DM_VK_OPS") != NULL;
    if (opslog && !(g_subs & (g_subs - 1))) {
        char b[600]; int q = 0;
        for (int i = 0; i < 64 && q < (int)sizeof b - 24; i++) if (g_ops[i]) q += snprintf(b + q, sizeof b - q, " %d:%llu", i, (unsigned long long)g_ops[i]);
        { char dk[400]; if (vkx_dispatch_keys_log(dk, sizeof dk)) LOG("vulkan replay: compute dispatched since the last line, by key:%s", dk); }
        LOG("vulkan replay: %llu submissions, %llu clears; records by op:%s; skipped mesh %llu mdi %llu push-dropped %llu multi %llu dlss %llu drawtex %llu alpha %llu; NVN2 push applied %llu; ordinary indirect forwarded %llu; compute skipped by DM_VK_SKIP_CS %llu; samples-passed placeholders %llu",
            (unsigned long long)g_subs, (unsigned long long)g_clears, b,
            (unsigned long long)atomic_load(&g_skip_mesh), (unsigned long long)atomic_load(&g_skip_mdi), (unsigned long long)atomic_load(&g_push_dropped), (unsigned long long)atomic_load(&g_skip_multi), (unsigned long long)atomic_load(&g_skip_dlss),
            (unsigned long long)atomic_load(&g_skip_draw_tex), (unsigned long long)atomic_load(&g_skip_alpha), (unsigned long long)atomic_load(&g_push_applied), (unsigned long long)atomic_load(&g_ind_seen),
            (unsigned long long)vkx_skipcs_count(), (unsigned long long)atomic_load(&g_samples_placeholder));
    }
    pthread_mutex_lock(&g_tmu); reap(); stage_reap(); pthread_mutex_unlock(&g_tmu);
    if (g_ndump) dump_uploads(v);
    return v;
}
uint64_t mtl_last_fence(void) { return atomic_load(&g_fence); }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { if (g_on <= 0) return 1; if (!wait_value(fence, timeout_ns)) return 0; gp_run(done_value()); return 1; }
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
void mtl_finish(void) { if (g_on > 0) { wait_value(atomic_load(&g_fence), UINT64_MAX); if (g_cs_qp) { pthread_mutex_lock(&g_tmu); gpu_harvest_done(); pthread_mutex_unlock(&g_tmu); } gp_run(done_value()); } }
void mtl_drain(void) { mtl_finish(); }

// ---- the record stream ----
static void full_barrier(VkCommandBuffer c) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static VkImageSubresourceRange view_range(const vtex *t, const view_t *v, int has, VkImageAspectFlags asp) {
    VkImageSubresourceRange r = {asp, 0, 1, 0, t->layers};
    if (has && v->set_levels) { r.baseMipLevel = v->base_level; r.levelCount = v->nlevels ? v->nlevels : 1; }
    if (has && v->set_layers) { r.baseArrayLayer = v->min_layer; r.layerCount = v->nlayers ? v->nlayers : 1; }
    if (r.baseMipLevel >= t->levels) r.baseMipLevel = t->levels - 1;
    if (r.baseMipLevel + r.levelCount > t->levels) r.levelCount = t->levels - r.baseMipLevel;
    if (r.baseArrayLayer >= t->layers) r.baseArrayLayer = t->layers - 1;
    if (r.baseArrayLayer + r.layerCount > t->layers) r.layerCount = t->layers - r.baseArrayLayer;
    return r;
}

// ---- render passes over the bound render targets (step 3.4c) ----
// Dynamic rendering begins at the first draw after the targets were set, and ends before anything that cannot be inside it (a clear, a target change, a barrier, the
// submission's end), each end followed by a full barrier so the next pass or a sampling sees what was drawn
static int g_rp_on, g_rp_ncolor, g_rp_samples, g_rp_drew; static uint32_t g_rp_w, g_rp_h, g_rp_layers; static uint64_t g_passes; static uint64_t g_ub_updates; static int g_passlog; static uint64_t g_pass0; static long g_passstop;
typedef struct {
    uint8_t bound, resolved, dropped, view;
    uint64_t serial;
    int32_t nvn_format, vk_format;
    uint32_t usage;
} rp_att_diag;
typedef struct {
    int ncolor;
    rp_att_diag color[8], depth;
} rp_diag;
static rp_diag g_rp_diag;
// Temporary visibility escape hatch for titles whose output-resolution
// compute compositor is unavailable.  Keep copies of textures from render
// passes which actually issued a draw; mtl_present may hand a recent
// lower-resolution, same-aspect source to the window's existing Vulkan blit.
// No guest serial or title key is baked into the selection.
typedef struct { tex_t tex; uint32_t w, h; uint8_t depth; } final_candidate;
#define FINAL_HISTORY 512
static final_candidate g_final_hist[FINAL_HISTORY];
static uint32_t g_final_n;
static pthread_mutex_t g_final_mu = PTHREAD_MUTEX_INITIALIZER;
static int final_mode(void) {
    static int mode = -1;
    if (mode >= 0) return mode;
    const char *e = getenv("DM_VK_FINAL_BLIT");
    mode = !e || !*e || !strcmp(e, "0") ? 0 : !strcmp(e, "scene") ? 2 : 1;
    if (mode) LOG("vulkan replay: output bypass active (%s): presenting an executed low-resolution %s pass through the Vulkan window blit",
                  mode == 2 ? "scene" : "post", mode == 2 ? "depth-backed colour" : "colour");
    return mode;
}
static void final_remember(void) {
    if (!final_mode() || !g_rp_drew || !g_rt_n || !g_rt[0] || !g_rp_w || !g_rp_h ||
        (g_rt_has_view[0] && ((g_rt_view[0].set_levels && g_rt_view[0].base_level) ||
                              (g_rt_view[0].set_layers && g_rt_view[0].min_layer)))) return;
    final_candidate v = {.tex = *g_rt[0], .w = g_rp_w, .h = g_rp_h,
                         .depth = g_rp_diag.depth.view != 0};
    pthread_mutex_lock(&g_final_mu);
    g_final_hist[g_final_n++ % FINAL_HISTORY] = v;
    pthread_mutex_unlock(&g_final_mu);
}
static void mesh_rp_failure_log(int indirect) {
    uint64_t seen = atomic_fetch_add(&g_mesh_fail_rp, 1);
    atomic_fetch_add(&g_skip_mesh, 1);
    if (seen) return;
    LOG("[RED] vulkan replay: mesh/task draw (DrawMeshTasks%s) bridge passed but no usable render attachment", indirect ? "Indirect" : "");
    if (!mesh_log_on()) return;
    for (int i = 0; i < g_rp_diag.ncolor; i++) {
        const rp_att_diag *d = &g_rp_diag.color[i];
        LOG("[RED] vulkan mesh render target: color[%d] bound=%u serial=%llu NVN=%#x resolved=%u VkFormat=%d usage=%#x dropped=%u view=%u",
            i, d->bound, (unsigned long long)d->serial, (unsigned)d->nvn_format,
            d->resolved, d->vk_format, d->usage, d->dropped, d->view);
    }
    { const rp_att_diag *d = &g_rp_diag.depth;
      LOG("[RED] vulkan mesh render target: depth bound=%u serial=%llu NVN=%#x resolved=%u VkFormat=%d usage=%#x view=%u",
          d->bound, (unsigned long long)d->serial, (unsigned)d->nvn_format,
          d->resolved, d->vk_format, d->usage, d->view); }
}
static VkImageView att_view_n(vtex *t, uint32_t level, uint32_t layer, uint32_t n, VkImageAspectFlags asp, VkFormat vf) {   // under g_tmu; n layers (a 3D image's depth slices); vf is a compatible listed view format
    if (vf == VK_FORMAT_UNDEFINED) vf = t->fmt;
    uint64_t key = 0x80000000u | (level & 0xff) | (layer & 0xfff) << 8 | (n & 0x7ff) << 20 | (uint64_t)(uint32_t)vf << 32;
    for (vkview *v = t->views; v; v = v->next) if (v->key == key) return v->view;
    if (t->d > 1 && !(t->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) return VK_NULL_HANDLE;
    VkImageViewUsageCreateInfo vu = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = vf != t->fmt ? &vu : NULL, .image = t->img, .viewType = n > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
        .format = vf, .subresourceRange = {asp, level, 1, layer, n}};   // of a 3D image (2D_ARRAY_COMPATIBLE), the layers are depth slices of the level
    vkview *v = calloc(1, sizeof *v); v->key = key;
    if (vkCreateImageView(g_vkd.dev, &vi, NULL, &v->view)) { free(v); return VK_NULL_HANDLE; }
    v->next = t->views; t->views = v; return v->view;
}
static VkImageView att_view(vtex *t, uint32_t level, uint32_t layer, VkImageAspectFlags asp) { return att_view_n(t, level, layer, 1, asp, t->fmt); }
// ---- uniform updates inside a render pass (UpdateUniformBuffer) ----
// vkCmdUpdateBuffer cannot run inside a render pass, so each update used to end the pass, and Unity updates a uniform buffer before nearly every draw: Pokemon Shining
// Pearl's town drew 531 draws in 506 passes, each loading and storing its 4x 1280x720 targets on this tiled GPU (55 ms of GPU a frame). As the Mac replay (nvn_mtl.m, ubd):
// inside a pass an update is deferred; a draw whose uniform binding overlaps a deferred update reads a shadow (vkr_ub_shadow); the deferred updates are made in order when
// the pass ends, so later passes, copies and the CPU see the memory as before. A shadow is the binding as the GPU would read it at that draw: guest memory with every update
// the GPU has not executed yet laid over it in order (g_ulog keeps a copy of each until its submission's fence is passed; memory alone missed the ones still queued).
// Open: storage, vertex or index reads of deferred bytes in the same pass read memory without them (the Mac ends the pass for those). DM_VK_UB_DEFER=0 turns it off
typedef struct { uint64_t dst, n, after; uint8_t *data; } ulog_t;
static ulog_t *g_ulog; static size_t g_nulog, g_culog;   // the updates the GPU may not have executed yet, in order (under the replay's thread)
typedef struct { uint64_t dst, n; const uint8_t *data; } ubd_t;
static ubd_t *g_ubd; static size_t g_nubd, g_cubd;      // deferred in the pass in progress (data: in the command memory being replayed)
static uint64_t g_ub_deferred, g_ub_shadows;
static stage_buf *g_shr, *g_shr_all; static uint8_t *g_shr_map; static size_t g_shr_off, g_shr_cap;   // append-only shadow ring; each piece retires after its latest submission
static int ub_defer_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_UB_DEFER"); on = !e || atoi(e) != 0; } return on; }
static void ulog_add(uint64_t dst, uint64_t n, const uint8_t *data) {
    uint64_t done = done_value(); size_t k = 0;
    for (size_t i = 0; i < g_nulog; i++) { if (g_ulog[i].after <= done) free(g_ulog[i].data); else g_ulog[k++] = g_ulog[i]; }   // executed: in memory now
    g_nulog = k;
    if (g_nulog == g_culog) { size_t nc = g_culog ? 2 * g_culog : 256; ulog_t *nl = realloc(g_ulog, nc * sizeof *nl); if (!nl) return; g_ulog = nl; g_culog = nc; }
    uint8_t *cp = malloc(n); if (!cp) return; memcpy(cp, data, n);
    g_ulog[g_nulog++] = (ulog_t){dst, n, atomic_load(&g_fence) + 1, cp};
}
static void ubd_land(VkCommandBuffer c) {   // the deferred updates, in order, outside the pass
    for (size_t i = 0; i < g_nubd; i++) {
        VkBuffer b; VkDeviceSize off; VkDeviceAddress g;
        if (vkm_lookup((const void *)(uintptr_t)g_ubd[i].dst, g_ubd[i].n, &b, &off, &g)) vkCmdUpdateBuffer(c, b, off, g_ubd[i].n, g_ubd[i].data);
    }
    g_nubd = 0;
}
static void rp_end_keep(VkCommandBuffer c) { if (!g_rp_on) return; vkCmdEndRendering(c); final_remember(); g_rp_on = 0; g_rp_drew = 0; ubd_land(c); full_barrier(c); }   // the pass ends; pending clears stay pending
// ---- whole-attachment clears held for the next pass's load operation ----
// NVN's ClearColor / ClearDepthStencil outside a scissor clear the whole target, and the game draws into it next. Recorded as vkCmdClear*Image the clear is
// a pass of its own on a tiled GPU (a compute fill on Honeykrisp G13, behind a barrier), and the draw pass that follows then loads the image it just wrote
// back into the tile buffer. Ocarina of Time clears 49 attachments this way in a Kokiri Forest frame (biosed-agent's audit). The clear is kept pending
// instead: a pass that attaches exactly that image level and layer over its whole size takes it as VK_ATTACHMENT_LOAD_OP_CLEAR; anything else that could
// read the image first (another pass, a copy, a dispatch, a barrier, the submission's end: every rp_end) records the clear as before. DM_VK_PEND_CLEAR=0
// records every clear at once.
typedef struct { vtex *t; VkImageSubresourceRange r; VkClearValue v; VkImageAspectFlags asp; } pclear;
enum { PC_MAX = 12 };
static pclear g_pc[PC_MAX]; static int g_pcn; static uint64_t g_pc_taken, g_pc_recorded;
static int pc_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PEND_CLEAR"); v = !(e && !strcmp(e, "0")); } return v; }
static void pc_emit(VkCommandBuffer c, const pclear *q) {
    if (q->asp & VK_IMAGE_ASPECT_COLOR_BIT) vkCmdClearColorImage(c, q->t->img, VK_IMAGE_LAYOUT_GENERAL, &q->v.color, 1, &q->r);
    else { VkImageSubresourceRange r = q->r; r.aspectMask = q->asp; vkCmdClearDepthStencilImage(c, q->t->img, VK_IMAGE_LAYOUT_GENERAL, &q->v.depthStencil, 1, &r); }
    full_barrier(c); g_clears++; g_pc_recorded++;
}
static void pc_flush(VkCommandBuffer c) { for (int i = 0; i < g_pcn; i++) pc_emit(c, &g_pc[i]); g_pcn = 0; }
static void pc_drop(const vtex *t) { for (int i = 0; i < g_pcn;) if (g_pc[i].t == t) g_pc[i] = g_pc[--g_pcn]; else i++; }   // an image that is retired: nothing may name it again
static void pc_add(VkCommandBuffer c, vtex *t, VkImageSubresourceRange r, VkClearValue v, VkImageAspectFlags asp) {
    for (int i = 0; i < g_pcn; i++) { pclear *q = &g_pc[i];
        if (q->t != t || q->r.baseMipLevel != r.baseMipLevel || q->r.levelCount != r.levelCount || q->r.baseArrayLayer != r.baseArrayLayer || q->r.layerCount != r.layerCount) continue;
        if (asp & VK_IMAGE_ASPECT_COLOR_BIT) { q->v = v; return; }   // a later clear of the same range replaces the earlier one
        if (asp & VK_IMAGE_ASPECT_DEPTH_BIT) q->v.depthStencil.depth = v.depthStencil.depth;
        if (asp & VK_IMAGE_ASPECT_STENCIL_BIT) q->v.depthStencil.stencil = v.depthStencil.stencil;
        q->asp |= asp; return; }
    if (g_pcn == PC_MAX) { pc_emit(c, &g_pc[0]); memmove(g_pc, g_pc + 1, (size_t)(--g_pcn) * sizeof *g_pc); }
    g_pc[g_pcn++] = (pclear){t, r, v, asp};
}
// The pending clear of exactly this attachment (level, first layer, layer count, whole level = the render area), removed: its aspects, 0 when there is none
static VkImageAspectFlags pc_take(const vtex *t, uint32_t mip, uint32_t layer0, uint32_t nlayers, uint32_t w, uint32_t h, VkClearValue *v) {
    for (int i = 0; i < g_pcn; i++) { const pclear *q = &g_pc[i];
        uint32_t levels = q->r.levelCount == VK_REMAINING_MIP_LEVELS ? t->levels - q->r.baseMipLevel : q->r.levelCount;
        uint32_t tl = t->d > 1 ? (t->d >> mip ? t->d >> mip : 1) : t->layers;
        uint32_t layers = q->r.layerCount == VK_REMAINING_ARRAY_LAYERS ? tl - q->r.baseArrayLayer : q->r.layerCount;
        uint32_t lw = t->w >> mip ? t->w >> mip : 1, lh = t->h >> mip ? t->h >> mip : 1;
        if (q->t != t || q->r.baseMipLevel != mip || levels != 1 || q->r.baseArrayLayer != layer0 || layers != nlayers || lw != w || lh != h || t->d > 1) continue;
        VkImageAspectFlags asp = q->asp; *v = q->v; g_pc[i] = g_pc[--g_pcn]; g_pc_taken++;
        if (!(g_pc_taken & (g_pc_taken - 1))) LOG("vulkan replay: %llu whole-attachment clears taken as a pass's load operation, %llu recorded as clears", (unsigned long long)g_pc_taken, (unsigned long long)g_pc_recorded);
        return asp; }
    return 0;
}
static void rp_end(VkCommandBuffer c) { rp_end_keep(c); pc_flush(c); }
// A uniform binding [addr, addr + size) for the draw being prepared (vk_draw.c stage_set): 1 and a shadow in *b / *off when a deferred update overlaps it, else 0 (memory as bound)
int vkr_ub_shadow(uint64_t addr, uint64_t size, VkBuffer *b, VkDeviceSize *off) {
    int hit = 0;
    for (size_t i = 0; i < g_nubd && !hit; i++) hit = g_ubd[i].dst < addr + size && addr < g_ubd[i].dst + g_ubd[i].n;
    if (!hit || !size || size > 65536) return 0;
    size_t need = (size + 255) & ~(size_t)255;
    uint64_t after = atomic_load(&g_fence) + 1;
    if (!g_shr || g_shr_off + need > g_shr_cap) {   // a new ring piece, freed once the GPU is past its latest user
        uint64_t done = done_value();
        for (stage_buf **q = &g_shr_all; *q;) { stage_buf *o = *q; if (o->after <= done && o != g_shr) { *q = o->next; vkDestroyBuffer(g_vkd.dev, o->buf, NULL); vkFreeMemory(g_vkd.dev, o->mem, NULL); free(o); } else q = &o->next; }
        size_t cap = need > (1u << 20) ? need : (1u << 20);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
        stage_buf *sb = calloc(1, sizeof *sb); if (!sb) return 0;
        if (vkCreateBuffer(g_vkd.dev, &bi, NULL, &sb->buf)) { free(sb); return 0; }
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, sb->buf, &mr);
        VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        void *map = NULL;
        if (ma.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_vkd.dev, &ma, NULL, &sb->mem) || vkBindBufferMemory(g_vkd.dev, sb->buf, sb->mem, 0) || vkMapMemory(g_vkd.dev, sb->mem, 0, VK_WHOLE_SIZE, 0, &map)) {
            vkDestroyBuffer(g_vkd.dev, sb->buf, NULL); if (sb->mem) vkFreeMemory(g_vkd.dev, sb->mem, NULL); free(sb); return 0; }
        sb->after = after; sb->next = g_shr_all; g_shr_all = sb;   // a list of their own: the draw path may hold g_tmu here (the staging list's lock)
        g_shr = sb; g_shr_map = map; g_shr_off = 0; g_shr_cap = cap;
    }
    uint8_t *dst = g_shr_map + g_shr_off;
    memcpy(dst, (const void *)(uintptr_t)addr, size);   // memory as the CPU sees it, then every update not executed yet, in order (the deferred ones are among them)
    for (size_t i = 0; i < g_nulog; i++) {
        const ulog_t *u = &g_ulog[i]; uint64_t lo = u->dst > addr ? u->dst : addr, hi = u->dst + u->n < addr + size ? u->dst + u->n : addr + size;
        if (lo < hi) memcpy(dst + (lo - addr), u->data + (lo - u->dst), hi - lo);
    }
    *b = g_shr->buf; *off = g_shr_off; g_shr_off += need; g_shr->after = after; g_ub_shadows++;
    return 1;
}
static uint64_t g_rp_kept;   // target sets that kept the pass going (rt_same)
static int rt_same(const p_set_rt *s) {   // s binds exactly the targets of the pass in progress: the same images (by serial), views and depth
    int n = s->n < 0 ? 0 : s->n > 8 ? 8 : s->n;
    if (n != g_rt_n) return 0;
    for (int i = 0; i < n; i++) {
        const tex_t *a = s->color[i], *b = g_rt[i];
        if (!a != !b || (a && a->serial != b->serial) || s->has_cview[i] != g_rt_has_view[i] || (s->has_cview[i] && memcmp(&s->cview[i], &g_rt_view[i], sizeof s->cview[i]))) return 0;
    }
    const tex_t *a = s->depth, *b = g_rt_depth;
    return !a == !b && (!a || a->serial == b->serial) && s->has_dview == g_rt_has_dview && (!s->has_dview || !memcmp(&s->dview, &g_rt_dview, sizeof s->dview));
}
// The colour slots to leave unattached for the draw about to be made: a slot whose texture, level and layer repeat an earlier slot, and of which the channel mask writes
// nothing. On the Switch two slots on one texture do not interfere (the ROP writes only the masked channels); on this tiled GPU each attachment is stored back whole at the
// end of the pass, so an unwritten duplicate (loaded, never written) overwrites what the other slot wrote. Breath of the Wild's G-buffer passes attach the albedo and another
// buffer twice each (c1 = c2, c3 = c4), the second at mask 0: attached, Link and the shrine came out flat and dark. A written duplicate stays attached: dropping every one
// took the blue out of Mario Kart 8 Deluxe's race picture. The set is part of the pass: a draw that needs another set ends the pass (vk_replay's OP_DRAW), as nvn_mtl.m's
// rt_dup_slot leaves such slots out of its passes
static uint32_t g_rp_drop;
static uint32_t rp_drop_set(void) {
    uint32_t m = vkx_chmask(), drop = 0;
    for (int i = 1; i < g_rt_n && i < 8; i++) {
        if (!g_rt[i] || (m >> (4 * i) & 0xf)) continue;
        uint32_t li = g_rt_has_view[i] && g_rt_view[i].set_levels ? g_rt_view[i].base_level : 0, yi = g_rt_has_view[i] && g_rt_view[i].set_layers ? g_rt_view[i].min_layer : 0;
        for (int j = 0; j < i; j++) {
            if (!g_rt[j] || g_rt[j]->serial != g_rt[i]->serial) continue;
            uint32_t lj = g_rt_has_view[j] && g_rt_view[j].set_levels ? g_rt_view[j].base_level : 0, yj = g_rt_has_view[j] && g_rt_view[j].set_layers ? g_rt_view[j].min_layer : 0;
            if (lj == li && yj == yi) { drop |= 1u << i; break; }
        }
    }
    return drop;
}
static int rp_begin(VkCommandBuffer c) {
    if (g_rp_on) return 1;
    g_rp_drop = rp_drop_set();
    VkRenderingAttachmentInfo ca[8], da, sa; int nc = 0, have = 0, have_c = 0, samples = 1; uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = 1;
    const vtex *at[8] = {0}, *adt = NULL; uint32_t amip[8] = {0}, alay[8] = {0}, dmip = 0, dlay = 0;   // what each slot attaches, for a pending clear (pc_take)
    memset(ca, 0, sizeof ca); memset(&da, 0, sizeof da); memset(&sa, 0, sizeof sa);
    memset(&g_rp_diag, 0, sizeof g_rp_diag); g_rp_diag.ncolor = g_rt_n;
    pthread_mutex_lock(&g_tmu);
    for (int i = 0; i < g_rt_n; i++) {
        rp_att_diag *dg = &g_rp_diag.color[i];
        dg->bound = g_rt[i] != NULL; dg->dropped = (g_rp_drop >> i) & 1;
        if (g_rt[i]) { dg->serial = g_rt[i]->serial; dg->nvn_format = g_rt[i]->d.format; }
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        nc = i + 1;
        vtex *t = g_rt[i] ? tex_get(g_rt[i]) : NULL;
        if (t) { dg->resolved = 1; dg->vk_format = (int32_t)t->fmt; dg->usage = t->usage; }
        if (!t || !(t->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) continue;
        VkImageSubresourceRange r = view_range(t, &g_rt_view[i], g_rt_has_view[i], VK_IMAGE_ASPECT_COLOR_BIT);
        VkFormat rtf = vkr_view_format(t->fmt, &g_rt_view[i], g_rt_has_view[i]);
        if (g_rp_drop >> i & 1) continue;   // a duplicate slot this draw does not write (rp_drop_set)
        if (t->d > 1) {   // a 3D target: the view's slices (all of the level's without one), drawn as layers (the shaders' layer output picks the slice)
            uint32_t dl = t->d >> r.baseMipLevel ? t->d >> r.baseMipLevel : 1, z0 = 0, nz = dl;
            if (g_rt_has_view[i] && g_rt_view[i].set_layers) { z0 = g_rt_view[i].min_layer < dl ? g_rt_view[i].min_layer : dl - 1; nz = g_rt_view[i].nlayers ? g_rt_view[i].nlayers : 1; if (z0 + nz > dl) nz = dl - z0; }
            ca[i].imageView = att_view_n(t, r.baseMipLevel, z0, nz, VK_IMAGE_ASPECT_COLOR_BIT, rtf); if (nz > layers) layers = nz;
        } else { ca[i].imageView = att_view_n(t, r.baseMipLevel, r.baseArrayLayer, 1, VK_IMAGE_ASPECT_COLOR_BIT, rtf); at[i] = t; amip[i] = r.baseMipLevel; alay[i] = r.baseArrayLayer; }
        dg->view = ca[i].imageView != VK_NULL_HANDLE;
        if (!ca[i].imageView) { at[i] = NULL; continue; }
        vkr_tex_gpu_write(t);
        have = have_c = 1; samples = (int)t->samples;
        uint32_t lw = t->w >> r.baseMipLevel ? t->w >> r.baseMipLevel : 1, lh = t->h >> r.baseMipLevel ? t->h >> r.baseMipLevel : 1;
        if (lw < w) w = lw; if (lh < h) h = lh;
    }
    int has_d = 0, has_s = 0;
    rp_att_diag *dd = &g_rp_diag.depth;
    dd->bound = g_rt_depth != NULL;
    if (g_rt_depth) { dd->serial = g_rt_depth->serial; dd->nvn_format = g_rt_depth->d.format; }
    vtex *dt = g_rt_depth ? tex_get(g_rt_depth) : NULL;
    if (dt) { dd->resolved = 1; dd->vk_format = (int32_t)dt->fmt; dd->usage = dt->usage; }
    if (dt && (dt->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        VkImageSubresourceRange r = view_range(dt, &g_rt_dview, g_rt_has_dview, dt->aspect);
        VkImageView v = att_view(dt, r.baseMipLevel, r.baseArrayLayer, dt->aspect);
        dd->view = v != VK_NULL_HANDLE;
        if (v) {
            has_d = (dt->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0; has_s = (dt->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
            da = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            // the depth's sample count when no colour target is bound, also with colour slots that are all empty: DARK SOULS: REMASTERED's depth pre-pass binds four
            // empty slots over its 4x depth, and rasterized at 1 sample into it the pre-pass depth came out wrong in tile-shaped areas, where the material pass then failed its
            // less-or-equal test and the target's clear colour showed (flat grey walls and armour)
            sa = da; vkr_tex_gpu_write(dt); have = 1; if (!have_c) samples = (int)dt->samples;
            adt = dt; dmip = r.baseMipLevel; dlay = r.baseArrayLayer;
            uint32_t lw = dt->w >> r.baseMipLevel ? dt->w >> r.baseMipLevel : 1, lh = dt->h >> r.baseMipLevel ? dt->h >> r.baseMipLevel : 1;
            if (lw < w) w = lw; if (lh < h) h = lh;
        }
    }
    if (have && g_pcn) {   // pending clears of exactly these attachments become their load operation; the others are recorded before the pass begins
        for (int i = 0; i < nc; i++) { VkClearValue v; if (at[i] && pc_take(at[i], amip[i], alay[i], 1, w, h, &v)) { ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca[i].clearValue = v; } }
        if (adt) { VkClearValue v; VkImageAspectFlags a = pc_take(adt, dmip, dlay, 1, w, h, &v);
            if (a & VK_IMAGE_ASPECT_DEPTH_BIT) { da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.clearValue = v; }
            if (a & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.clearValue = v; } }
    }
    pthread_mutex_unlock(&g_tmu);
    if (!have) return 0;
    pc_flush(c);
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = layers, .colorAttachmentCount = (uint32_t)nc, .pColorAttachments = ca,
        .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    vkCmdBeginRendering(c, &ri);
    g_rp_on = 1; g_rp_drew = 0; g_rp_ncolor = nc; g_rp_w = w; g_rp_h = h; g_rp_layers = layers; g_rp_samples = samples; g_passes++;
    if (g_passlog) {   // each colour slot: the texture's serial, '-' when the slot is empty, '!' after it when no view could be attached
        char sl[256]; int o = 0; for (int i = 0; i < nc && o < (int)sizeof sl - 24; i++) o += snprintf(sl + o, sizeof sl - o, " %d:%s%llu%s", i, g_rt[i] ? "" : "-", g_rt[i] ? (unsigned long long)g_rt[i]->serial : 0ull, g_rt[i] && !ca[i].imageView ? "!" : "");
        LOG("[pass] %llu: %ux%u, %d colour (first serial %llu), depth %s; slots%s", (unsigned long long)g_passes, w, h, nc, g_rt[0] ? (unsigned long long)g_rt[0]->serial : 0ull, has_d || has_s ? "yes" : "no", sl);
    }
    return 1;
}
// NVN's clears are limited to the scissor rectangle, as the Mac's are (nvn_mtl.m's scissored clears); a whole-image clear would also wipe what a game drew outside it.
// A clear whose scissor leaves part of the level (w x h) uncovered runs inside the render pass with vkCmdClearAttachments over that rectangle: 1 = cleared there or nothing
// to clear (empty), 0 = clear the whole image
static int scissored(uint32_t w, uint32_t h, VkRect2D *rc) {
    int sc[4]; if (!vkx_scissor_get(sc)) return 0;
    int64_t x0 = sc[0] < 0 ? 0 : sc[0], y0 = sc[1] < 0 ? 0 : sc[1], x1 = (int64_t)sc[0] + sc[2], y1 = (int64_t)sc[1] + sc[3];
    if (x1 > (int64_t)w) x1 = w; if (y1 > (int64_t)h) y1 = h;
    if (x0 == 0 && y0 == 0 && x1 >= (int64_t)w && y1 >= (int64_t)h) return 0;   // covers the whole level
    *rc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {x1 > x0 ? (uint32_t)(x1 - x0) : 0, y1 > y0 ? (uint32_t)(y1 - y0) : 0}};
    return 1;
}
static int clear_in_pass(VkCommandBuffer c, VkImageAspectFlags asp, uint32_t att, const VkClearValue *v, VkRect2D rc) {   // 1 = done
    if (!rc.extent.width || !rc.extent.height) return 1;
    if (!rp_begin(c)) return 0;
    if (rc.offset.x + rc.extent.width > g_rp_w) rc.extent.width = (uint32_t)rc.offset.x < g_rp_w ? g_rp_w - (uint32_t)rc.offset.x : 0;
    if (rc.offset.y + rc.extent.height > g_rp_h) rc.extent.height = (uint32_t)rc.offset.y < g_rp_h ? g_rp_h - (uint32_t)rc.offset.y : 0;
    if (!rc.extent.width || !rc.extent.height) return 1;
    VkClearAttachment ca = {.aspectMask = asp, .colorAttachment = att, .clearValue = *v};
    VkClearRect cr = {rc, 0, g_rp_layers ? g_rp_layers : 1};
    vkCmdClearAttachments(c, 1, &ca, 1, &cr); g_clears++;
    return 1;
}
static void op_clear_color(VkCommandBuffer c, const p_clear_color *p) {
    if (p->index >= 0 && p->index < g_rt_n && g_rt[p->index]) {
        pthread_mutex_lock(&g_tmu); vtex *t = tex_get(g_rt[p->index]); VkImageSubresourceRange r = {0};
        if (t) r = view_range(t, &g_rt_view[p->index], g_rt_has_view[p->index], VK_IMAGE_ASPECT_COLOR_BIT);
        pthread_mutex_unlock(&g_tmu);
        VkRect2D rc; VkClearValue v; memcpy(v.color.float32, p->rgba, sizeof v.color.float32);
        if (t && (t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (t->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            && scissored(t->w >> r.baseMipLevel ? t->w >> r.baseMipLevel : 1, t->h >> r.baseMipLevel ? t->h >> r.baseMipLevel : 1, &rc)
            && clear_in_pass(c, VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)p->index, &v, rc)) return;
    }
    if (pc_on()) rp_end_keep(c); else rp_end(c);
    if (p->index < 0 || p->index >= g_rt_n || !g_rt[p->index]) return;
    pthread_mutex_lock(&g_tmu);
    vtex *t = tex_get(g_rt[p->index]);
    // multisampled targets too (vkCmdClearColorImage / vkCmdClearDepthStencilImage take any sample count): Pokémon Shining Pearl draws its scene into a 4x target whose depth,
    // never cleared, failed its reversed-Z GEQUAL test everywhere, and its background and characters were not drawn
    if (t && (t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        vkr_tex_gpu_write(t);
        tex_ready(c, t);
        if (g_passlog) { int sc[4], hs = vkx_scissor_get(sc); LOG("[pass] clear colour %d serial %llu to %g %g %g %g mask %#x scissor %s %d %d %d %d (image %ux%u)", p->index, (unsigned long long)t->serial,
                                p->rgba[0], p->rgba[1], p->rgba[2], p->rgba[3], p->mask, hs ? "on" : "off", sc[0], sc[1], sc[2], sc[3], t->w, t->h); }
        VkClearColorValue v; memcpy(v.float32, p->rgba, sizeof v.float32);   // a float clear; integer targets take the same bits (open: NVN's integer clears)
        VkImageSubresourceRange r = view_range(t, &g_rt_view[p->index], g_rt_has_view[p->index], VK_IMAGE_ASPECT_COLOR_BIT);
        if (pc_on()) { VkClearValue cv; cv.color = v; pc_add(c, t, r, cv, VK_IMAGE_ASPECT_COLOR_BIT); }
        else { vkCmdClearColorImage(c, t->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r); full_barrier(c); g_clears++; }
    }
    pthread_mutex_unlock(&g_tmu);
}
static void op_clear_ds(VkCommandBuffer c, const p_clear_ds *p) {
    if (!g_rt_depth) { rp_end(c); return; }
    {   // a scissored clear (see scissored)
        pthread_mutex_lock(&g_tmu); vtex *t = tex_get(g_rt_depth); VkImageSubresourceRange r = {0};
        if (t) r = view_range(t, &g_rt_dview, g_rt_has_dview, t->aspect);
        pthread_mutex_unlock(&g_tmu);
        VkImageAspectFlags asp = 0;
        if (t && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && p->depth_mask) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (t && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && p->stencil_mask) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        VkRect2D rc; VkClearValue v = {.depthStencil = {p->depth < 0 ? 0 : p->depth > 1 ? 1 : p->depth, (uint32_t)p->stencil}};
        if (t && asp && !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (t->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
            && scissored(t->w >> r.baseMipLevel ? t->w >> r.baseMipLevel : 1, t->h >> r.baseMipLevel ? t->h >> r.baseMipLevel : 1, &rc)
            && clear_in_pass(c, asp, 0, &v, rc)) return;
    }
    if (pc_on()) rp_end_keep(c); else rp_end(c);
    pthread_mutex_lock(&g_tmu);
    vtex *t = tex_get(g_rt_depth);
    if (g_passlog) LOG("[pass] clear depth %g (mask %d) stencil %d (mask %#x) serial %llu%s", p->depth, p->depth_mask, p->stencil, p->stencil_mask, t ? (unsigned long long)t->serial : 0ull, t ? "" : ": no image");
    if (t && !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {   // multisampled too (op_clear_color)
        VkImageAspectFlags asp = 0;
        if ((t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && p->depth_mask) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if ((t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && p->stencil_mask) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        if (asp) {
            vkr_tex_gpu_write(t);
            tex_ready(c, t);
            VkClearDepthStencilValue v = {p->depth < 0 ? 0 : p->depth > 1 ? 1 : p->depth, (uint32_t)p->stencil};
            VkImageSubresourceRange r = view_range(t, &g_rt_dview, g_rt_has_dview, asp);
            if (pc_on()) { VkClearValue cv = {.depthStencil = v}; pc_add(c, t, r, cv, asp); }
            else { vkCmdClearDepthStencilImage(c, t->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r); full_barrier(c); g_clears++; }
        }
    }
    pthread_mutex_unlock(&g_tmu);
}
// ---- copies and texture clears (the records nvn_obj.h shares with nvn_mtl.m) ----
// Each runs outside a render pass and is followed by a full barrier. A texture a copy writes becomes the GPU's (never reloaded from guest memory); one that still holds
// packaged content is loaded first, so a copy into part of it keeps the rest
static int64_t g_copy_row, g_copy_img;   // SetCopyRowStride / SetCopyImageStride (0 = tightly packed)
static uint64_t g_copies, g_copy_fail;
static void copy_fail(const char *what) { g_copy_fail++; if (!(g_copy_fail & (g_copy_fail - 1))) LOG("[RED] vulkan replay: %llu copies not made (latest: %s)", (unsigned long long)g_copy_fail, what); }
// The subresource and offset / extent of a copy region on an image (layers from the view and the region's z for layered images, depth slices for a 3D one)
static void region_of(const vtex *t, const view_t *v, int has_v, const NVNcopyRegion *r, VkImageSubresourceLayers *sl, VkOffset3D *o, VkExtent3D *e) {
    VkImageSubresourceRange vr = view_range(t, v, has_v, t->aspect);
    *sl = (VkImageSubresourceLayers){t->aspect, vr.baseMipLevel, 0, 1};
    uint32_t w = r->width > 0 ? (uint32_t)r->width : 1, h = r->height > 0 ? (uint32_t)r->height : 1, d = r->depth > 0 ? (uint32_t)r->depth : 1;
    if (t->d > 1) { *o = (VkOffset3D){r->xoffset, r->yoffset, r->zoffset}; *e = (VkExtent3D){w, h, d}; }
    else { sl->baseArrayLayer = vr.baseArrayLayer + (uint32_t)(r->zoffset > 0 ? r->zoffset : 0); sl->layerCount = d; *o = (VkOffset3D){r->xoffset, r->yoffset, 0}; *e = (VkExtent3D){w, h, 1}; }
    if (sl->baseArrayLayer + sl->layerCount > t->layers) sl->layerCount = sl->baseArrayLayer < t->layers ? t->layers - sl->baseArrayLayer : 0;
}
static void op_copy_t2t(VkCommandBuffer c, const p_copy_t2t *p) {
    rp_end(c);
    pthread_mutex_lock(&g_tmu);
    vtex *s = tex_sampled(p->src), *d = tex_get(p->dst);
    if (!s || !d) { pthread_mutex_unlock(&g_tmu); copy_fail("texture to texture: an image is missing"); return; }
    if (g_passlog) LOG("[pass] copy texture serial %llu %ux%u -> serial %llu %ux%u%s (objects: serial %llu storage %p -> serial %llu storage %p)", (unsigned long long)s->serial, s->w, s->h, (unsigned long long)d->serial, d->w, d->h,
                       p->flags & DM_COPY_RESOLVE ? " (resolve)" : "", (unsigned long long)p->src->serial, (const void *)p->src->mem, (unsigned long long)p->dst->serial, (const void *)p->dst->mem);
    if (!d->gpu_owned && !d->uploaded) upload(d, p->dst);
    vkr_tex_gpu_write(d);
    VkImageSubresourceLayers ss, ds; VkOffset3D so, dof; VkExtent3D se, de;
    region_of(s, &p->sv, p->has_sv, &p->sr, &ss, &so, &se); region_of(d, &p->dv, p->has_dv, &p->dr, &ds, &dof, &de);
    if (!ss.layerCount || !ds.layerCount) { pthread_mutex_unlock(&g_tmu); copy_fail("texture to texture: layers out of range"); return; }
    if (p->flags & DM_COPY_RESOLVE) {
        VkImageResolve rr = {ss, so, ds, dof, se};
        vkCmdResolveImage(c, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rr);
    } else if (s->fmt == d->fmt && se.width == de.width && se.height == de.height && se.depth == de.depth && ss.layerCount == ds.layerCount && s->samples == d->samples) {
        VkImageCopy ic = {ss, so, ds, dof, se};
        vkCmdCopyImage(c, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    } else if ((s->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (d->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && s->samples == 1 && d->samples == 1) {
        if (ss.layerCount != ds.layerCount) ss.layerCount = ds.layerCount = ss.layerCount < ds.layerCount ? ss.layerCount : ds.layerCount;
        VkImageBlit bl = {ss, {so, {so.x + (int32_t)se.width, so.y + (int32_t)se.height, so.z + (int32_t)se.depth}}, ds, {dof, {dof.x + (int32_t)de.width, dof.y + (int32_t)de.height, dof.z + (int32_t)de.depth}}};
        vkCmdBlitImage(c, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, (p->flags & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);   // NVN_COPY_FLAGS_LINEAR_FILTER_BIT
    } else { pthread_mutex_unlock(&g_tmu); copy_fail("texture to texture: depth or multisampled images of different sizes or formats"); return; }
    pthread_mutex_unlock(&g_tmu);
    full_barrier(c); g_copies++;
}
// Buffer ↔ texture: the buffer side is linear, rows g_copy_row bytes apart (tight when 0)
static void op_copy_bt(VkCommandBuffer c, const p_copy_b2t *p, int to_tex) {
    rp_end(c);
    pthread_mutex_lock(&g_tmu);
    vtex *t = to_tex ? tex_get(p->tex) : tex_sampled(p->tex);
    if (!t) { pthread_mutex_unlock(&g_tmu); copy_fail("buffer and texture: no image"); return; }
    if (to_tex) { if (!t->gpu_owned && !t->uploaded) upload(t, p->tex); vkr_tex_gpu_write(t); }
    VkImageSubresourceLayers sl; VkOffset3D o; VkExtent3D e; region_of(t, &p->view, p->has_view, &p->r, &sl, &o, &e);
    int bw, bh, bpb = nvn_fmt_bpb(p->tex->d.format, &bw, &bh);
    if (!bpb || !sl.layerCount) { pthread_mutex_unlock(&g_tmu); copy_fail("buffer and texture: no block size"); return; }
    uint64_t row = g_copy_row > 0 ? (uint64_t)g_copy_row : (uint64_t)((e.width + bw - 1) / bw) * bpb, rows = (e.height + bh - 1) / bh;
    uint64_t img = g_copy_img > 0 ? (uint64_t)g_copy_img : row * rows, n = img * (e.depth > 1 ? e.depth : sl.layerCount);
    VkBuffer b; VkDeviceSize off; VkDeviceAddress g;
    if (row % bpb || !vkm_lookup((const void *)(uintptr_t)p->src, n, &b, &off, &g)) { pthread_mutex_unlock(&g_tmu); copy_fail("buffer and texture: buffer not in GPU-visible memory or stride not a whole block"); return; }
    // Ocarina reads decoder output into a codec buffer but uploads a distinct,
    // contiguous Y+UV presentation allocation.  The resolved image/region is
    // authoritative here (raw texture descriptors can encode their extent in
    // more than one form).  With the exact-title bridge enabled, populate both
    // adjacent planes after the GPU mapping is resolved and before sampling it.
    if (to_tex && dm_codec_movie_nv12_present && p->tex->d.format == NVN_FORMAT_R8 &&
        t->fmt == VK_FORMAT_R8_UNORM && t->w == 1920 && t->h == 1080 && sl.mipLevel == 0 &&
        sl.layerCount == 1 && o.x == 0 && o.y == 0 && o.z == 0 &&
        e.width == 1920 && e.height == 1080 && e.depth == 1)
        dm_codec_movie_nv12_present((void *)(uintptr_t)p->src, 1920, 1080);
    if (g_passlog && to_tex) {
        const uint8_t *src = (const uint8_t *)(uintptr_t)p->src;
        uint8_t lo = 255, hi = 0; uint64_t sum = 0, ns = 0;
        uint64_t step = n > 4096 ? n / 4096 : 1;
        for (uint64_t k = 0; k < n; k += step) {
            uint8_t v = src[k]; if (v < lo) lo = v; if (v > hi) hi = v;
            sum += v; ns++;
        }
        LOG("[pass] copy buffer %p +%#llx (%llu bytes, row %llu image %llu, sample %u..%u mean %.2f) -> texture serial %llu %ux%u fmt %#x storage %p",
            src, (unsigned long long)off, (unsigned long long)n, (unsigned long long)row,
            (unsigned long long)img, lo, hi, ns ? (double)sum / ns : 0.0,
            (unsigned long long)t->serial, t->w, t->h, p->tex->d.format, p->tex->mem);
    }
    VkBufferImageCopy bc = {off, (uint32_t)(row / bpb * bw), (uint32_t)(img / row * bh), sl, o, e};
    if (to_tex) vkCmdCopyBufferToImage(c, b, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bc);
    else vkCmdCopyImageToBuffer(c, t->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &bc);
    pthread_mutex_unlock(&g_tmu);
    full_barrier(c); g_copies++;
}
static void op_clear_tex(VkCommandBuffer c, const p_clear_tex *p) {
    rp_end(c);
    pthread_mutex_lock(&g_tmu);
    vtex *t = tex_get(p->tex);
    if (!t || !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) { pthread_mutex_unlock(&g_tmu); copy_fail("texture clear: no colour image"); return; }
    if (g_passlog) LOG("[pass] clear texture serial %llu %ux%u region %d %d %d %d", (unsigned long long)t->serial, t->w, t->h, p->r.xoffset, p->r.yoffset, p->r.width, p->r.height);
    if (!t->gpu_owned && !t->uploaded) upload(t, p->tex);
    vkr_tex_gpu_write(t);
    VkImageSubresourceLayers sl; VkOffset3D o; VkExtent3D e; region_of(t, &p->view, p->has_view, &p->r, &sl, &o, &e);
    VkClearColorValue v; memcpy(&v, &p->c, sizeof v);
    uint32_t lw = t->w >> sl.mipLevel ? t->w >> sl.mipLevel : 1, lh = t->h >> sl.mipLevel ? t->h >> sl.mipLevel : 1;
    if (!o.x && !o.y && e.width >= lw && e.height >= lh && t->d == 1) {   // the whole of each layer: a plain clear
        VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, sl.mipLevel, 1, sl.baseArrayLayer, sl.layerCount};
        vkCmdClearColorImage(c, t->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rr);
    } else if (t->d == 1 && (t->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {   // part of it: a rectangle cleared in a pass over each layer
        for (uint32_t L = 0; L < sl.layerCount; L++) {
            VkImageView iv = att_view(t, sl.mipLevel, sl.baseArrayLayer + L, VK_IMAGE_ASPECT_COLOR_BIT); if (!iv) continue;
            VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = iv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {lw, lh}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
            vkCmdBeginRendering(c, &ri);
            VkClearAttachment cl = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = v}}; VkClearRect cr = {{{o.x, o.y}, {e.width, e.height}}, 0, 1};
            vkCmdClearAttachments(c, 1, &cl, 1, &cr);
            vkCmdEndRendering(c);
        }
    } else { pthread_mutex_unlock(&g_tmu); copy_fail("texture clear: part of an image that cannot be a render target"); return; }
    pthread_mutex_unlock(&g_tmu);
    full_barrier(c);
}

void mtl_exec(const uint8_t *handle) {
    if (!g_on || !g_open) return;
    if (g_prof < 0) g_prof = getenv("DM_VK_PROF") != NULL;
    VkCommandBuffer c = g_cb[g_cb_i];
    const uint8_t *p = handle, *ret[16]; int nret = 0;
    for (uint64_t guard = 0; p && guard < 50000000; guard++) {
        int descriptor_split = vkx_descriptor_pressure();
        if (!g_noflush && ((g_flush_draws >= FLUSH_DRAWS && !g_rp_on) || descriptor_split)) {
            // Ordinary batching waits for the game to end dynamic rendering,
            // but an NVN pass may legitimately contain thousands of draws.
            // Bound descriptor lifetime directly: attachments use LOAD/STORE,
            // so a pressured pass can be stored here and resumed by the next
            // draw without changing its contents.
            if (descriptor_split) rp_end(c);
            uint64_t ft = g_prof ? mono_ns() : 0;
            vkr_flush(); c = g_cb[g_cb_i];
            if (g_prof) g_opns[64] += mono_ns() - ft;
        }
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (h.op < 64) g_ops[h.op]++;
        uint64_t pt = g_prof ? mono_ns() : 0;
        switch (h.op) {
        case OP_END: if (nret) { p = ret[--nret]; continue; } return;
        case OP_JUMP: memcpy(&p, a, 8); continue;
        case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); if (nret == 16) { LOG("[RED] vulkan replay: command buffer calls nested deeper than 16"); return; } ret[nret++] = p + h.size; p = t; continue; }
        case OP_SET_RT: {
            const p_set_rt *s = a;
            // The targets already bound, views and all: the pass goes on. Unity sets its targets before nearly every draw (Pokemon Shining Pearl's town: 531 draws in 506
            // passes), and each pass loads and stores its attachments on this tiled GPU (a 4x 1280x720 colour and depth, a 4096x4096 shadow map): 55 ms of GPU a frame
            if (g_rp_on && rt_same(s)) { g_rp_kept++; break; }
            rp_end(c);
            g_rt_n = s->n < 0 ? 0 : s->n > 8 ? 8 : s->n;
            for (int i = 0; i < 8; i++) { g_rt[i] = i < g_rt_n ? s->color[i] : NULL; g_rt_view[i] = s->cview[i]; g_rt_has_view[i] = s->has_cview[i]; }
            g_rt_depth = s->depth; g_rt_dview = s->dview; g_rt_has_dview = s->has_dview;
            break;
        }
        case OP_CLEAR_COLOR: if (!g_rc_skip) op_clear_color(c, a); break;
        case OP_CLEAR_DS: if (!g_rc_skip) op_clear_ds(c, a); break;
        case OP_RENDER_COND: rc_eval(a); break;   // recorded only with the profile key render_cond (hle_nvn.c)
        case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: case OP_BIND_IMG: {   // textures are made ready to sample now; the handles go to the draw state
            p_bind_handles t1; const p_bind_handles *b = bind_rec(h.op, a, &t1);
            if (b->n < 0 || b->n > 32) break;
            uint64_t hh[32]; memcpy(hh, b->h, 8 * (size_t)b->n);
            if (b->defer && b->src) memcpy(hh, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);   // BindTexturesDeferred: the handles as they are at execution
            vkx_bind_handles(h.op == OP_BIND_SAMP ? 1 : h.op == OP_BIND_IMG ? 2 : 0, b, hh);
            if (h.op == OP_BIND_TEX || h.op == OP_BIND_TEX1) op_bind_tex(h.op, a);
            break;
        }
        case OP_BIND_PROGRAM: vkx_bind_program(((const p_bind_program *)a)->prog, ((const p_bind_program *)a)->stages); break;
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tb; vkx_bind_buf(0, bind_buf_rec(h.op, a, &tb)); break; }
        case OP_BIND_SSBO: vkx_bind_buf(1, a); break;
        case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: vkx_set_pool(h.op == OP_SET_TEX_POOL, nvn_dpool_base(((const p_set_pool *)a)->pool, h.op == OP_SET_TEX_POOL)); break;
        case OP_DRAW: case OP_DRAW_IDX: {
            // Mesh/task is proof-gated below; unsupported multi-draw is still
            // counted. ind/mdi_cnt exist only when the record includes them.
            int32_t prim; memcpy(&prim, a, 4);
            size_t ind_off = h.op == OP_DRAW ? offsetof(p_draw, ind) : offsetof(p_draw_idx, ind), mdi_off = h.op == OP_DRAW ? offsetof(p_draw, mdi_cnt) : offsetof(p_draw_idx, mdi_cnt);
            uint64_t ind = 0, mdi = 0;
            if (h.size >= 8 + ind_off + 8) memcpy(&ind, (const uint8_t *)a + ind_off, 8);
            if (h.size >= 8 + mdi_off + 8) memcpy(&mdi, (const uint8_t *)a + mdi_off, 8);
            if (prim == DM_PRIM_MESH_TASKS) {
                p_draw md = {0}; size_t payload = h.size > 8 ? h.size - 8 : 0; if (payload > sizeof md) payload = sizeof md; memcpy(&md, a, payload);
                vkx_mesh_draw_telemetry(&md);
                if (g_rc_skip) break;
                if (g_passlog && g_passstop && g_passes - g_pass0 > (uint64_t)g_passstop) break;
                if (!vkd_mesh_active()) {
                    SKIP_ONCE(g_skip_mesh, "mesh/task draw (DrawMeshTasks%s) skipped: VK_EXT_mesh_shader runtime proof gate is closed", ind ? "Indirect" : ""); break;
                }
                // An indirect mesh draw copies its count and first-instance words out of guest memory with transfer commands, which must be outside dynamic
                // rendering: the pass ends for it. A direct one writes its record from the host and records nothing, so it stays in the pass the game's
                // other draws are in. Ending the pass before every mesh draw split Ocarina of Time's 1600x1600 shadow pass into 23 hardware passes and
                // two G-buffer passes into 29 (51 tile stores and reloads a frame; biosed-agent's audit of a Kokiri frame). DM_VK_MESH_KEEP_PASS=0
                // ends it for every mesh draw as before
                static int keep_pass = -1; if (keep_pass < 0) { const char *e = getenv("DM_VK_MESH_KEEP_PASS"); keep_pass = !(e && !strcmp(e, "0")); }
                if (ind || !keep_pass) rp_end(c);
                // A long render pass can consume most of this command
                // buffer's descriptor pools before its first mesh draw.  The
                // mesh bridge has just provided a legal split point; submit
                // the old part before allocating the task/mesh/fragment sets.
                if (!g_rp_on && g_flush_draws >= FLUSH_DRAWS && !g_noflush) {   // only at a split point: with the pass still open the next pass end flushes
                    uint64_t ft = g_prof > 0 ? mono_ns() : 0;
                    vkr_flush(); c = g_cb[g_cb_i];
                    if (g_prof > 0) g_opns[64] += mono_ns() - ft;
                }
                vkx_mesh_submit submit;
                if (!vkx_mesh_bridge(c, &md, &submit)) {
                    vkx_mesh_abort();
                    atomic_fetch_add(&g_skip_mesh, 1);
                    if (atomic_fetch_add(&g_mesh_fail_bridge, 1) == 0)
                        LOG("[RED] vulkan replay: mesh/task draw (DrawMeshTasks%s) failed the guest-to-Vulkan mesh bridge", ind ? "Indirect" : "");
                    break;
                }
                if (!rp_begin(c)) {
                    vkx_mesh_abort();
                    mesh_rp_failure_log(ind != 0);
                    break;
                }
                uint64_t mesh_seq = atomic_fetch_add(&g_mesh_issue_seq, 1) + 1;
                if (mesh_log_on())
                    LOG("vulkan mesh command: issue=%llu slot=%d mode=%s%s ts=%s ms=%s ps=%s guest-indirect=%#llx vkCmdDrawMeshTasksEXT begin",
                        (unsigned long long)mesh_seq, g_cb_i,
                        submit.command.indirect ? "indirect-" : "direct-",
                        submit.command.have_task ? "task" : "taskless",
                        submit.task_key[0] ? submit.task_key : "-", submit.mesh_key,
                        submit.frag_key[0] ? submit.frag_key : "-",
                        (unsigned long long)submit.guest_indirect);
                int mesh_ok = vkx_draw_mesh(c, &submit, g_rp_ncolor, g_rp_w, g_rp_h, g_rp_samples);
                if (mesh_log_on())
                    LOG("vulkan mesh command: issue=%llu slot=%d vkCmdDrawMeshTasksEXT returned ok=%d",
                        (unsigned long long)mesh_seq, g_cb_i, mesh_ok);
                if (!mesh_ok) {
                    vkx_mesh_abort();
                    atomic_fetch_add(&g_skip_mesh, 1);
                    if (atomic_fetch_add(&g_mesh_fail_draw, 1) == 0)
                        LOG("[RED] vulkan replay: mesh/task draw (DrawMeshTasks%s) reached rendering but failed program preparation or command issue", ind ? "Indirect" : "");
                    break;
                }
                g_rp_drew = 1;
                g_cb_mesh_seq[g_cb_i] = mesh_seq;
                if (mesh_log_on())
                    LOG("vulkan mesh command: issue=%llu recorded slot=%d prior-signal=%llu",
                        (unsigned long long)mesh_seq, g_cb_i, (unsigned long long)atomic_load(&g_cb_val[g_cb_i]));
                g_flush_draws++;
                // Diagnostic isolation for the software task/mesh path.  N=1
                // is the original strict proof; a small N keeps scratch
                // bounded while amortising queue/timeline round-trips.
                unsigned mesh_flush = mesh_flush_every();
                if (mesh_flush && ++g_mesh_batch_draws >= mesh_flush) {
                    rp_end(c);
                    vkr_flush();
                    c = g_cb[g_cb_i];
                }
                break;
            }
            if (mdi) { SKIP_ONCE(g_skip_mdi, "indirect-count multi-draw skipped: not implemented"); break; }
            if (ind && atomic_fetch_add(&g_ind_seen, 1) == 0) LOG("vulkan replay: ordinary indirect draws are forwarded to Vulkan");
        }
            if (g_rc_skip) break;
            if (g_passlog && g_passstop && g_passes - g_pass0 > (uint64_t)g_passstop) break;   // DM_VK_PASSSTOP: the traced frame's later passes draw nothing
            if (g_rp_on && rp_drop_set() != g_rp_drop) rp_end(c);   // another set of duplicate slots to leave out (rp_drop_set)
            if (rp_begin(c) && vkx_draw(c, h.op, a, h.size, g_rp_ncolor, g_rp_w, g_rp_h, g_rp_samples)) g_rp_drew = 1; g_flush_draws++; break;
        case OP_BARRIER: rp_end(c); full_barrier(c); break;
        case OP_COPY_T2T: op_copy_t2t(c, a); break;
        case OP_COPY_B2T: op_copy_bt(c, a, 1); break;
        case OP_COPY_T2B: op_copy_bt(c, a, 0); break;
        case OP_CLEAR_TEX: op_clear_tex(c, a); break;
        case OP_COPY_STRIDE: { const p_copy_stride *s = a; if (s->image) g_copy_img = s->stride; else g_copy_row = s->stride; break; }
        case OP_DISPATCH: rp_end(c); vkx_dispatch(c, a); full_barrier(c); g_flush_draws++; break;
        case OP_UPDATE_UB: {   // {dst, n, data}: the bytes written into the buffer at this point of the GPU's work (UpdateUniformBuffer)
            uint64_t dst, n; memcpy(&dst, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8);
            VkBuffer b; VkDeviceSize off; VkDeviceAddress g;
            if (n && n <= 65536 && !(n & 3) && !(dst & 3) && vkm_lookup((const void *)(uintptr_t)dst, n, &b, &off, &g)) {
                ulog_add(dst, n, (const uint8_t *)a + 16);
                if (g_rp_on && ub_defer_on()) {   // inside a pass: deferred to its end, the draws until then read shadows (vkr_ub_shadow)
                    if (g_nubd == g_cubd) { size_t nc = g_cubd ? 2 * g_cubd : 64; ubd_t *nd = realloc(g_ubd, nc * sizeof *nd); if (nd) { g_ubd = nd; g_cubd = nc; } }
                    if (g_nubd < g_cubd) { g_ubd[g_nubd++] = (ubd_t){dst, n, (const uint8_t *)a + 16}; g_ub_deferred++; g_ub_updates++; break; }
                }
                rp_end(c); vkCmdUpdateBuffer(c, b, off, n, (const uint8_t *)a + 16); full_barrier(c); g_ub_updates++;
                if (g_passlog) { const float *f = (const float *)((const uint8_t *)a + 16); LOG("[pass] uniform update %#llx + %llu bytes; first %g %g %g %g", (unsigned long long)dst, (unsigned long long)n, n >= 16 ? f[0] : 0.f, n >= 16 ? f[1] : 0.f, n >= 16 ? f[2] : 0.f, n >= 16 ? f[3] : 0.f);
                    static unsigned seq; const char *dd = getenv("DM_VK_SBUFDUMP");   // the update's bytes too, in order: <dir>/u<sequence>_<address>.bin
                    if (dd) { char path[600]; snprintf(path, sizeof path, "%s/u%05u_%llx.bin", dd, seq++, (unsigned long long)dst); FILE *o = fopen(path, "wb"); if (o) { fwrite(f, 1, n, o); fclose(o); } } }
            } else { static int w; if (!w++) LOG("[RED] vulkan replay: uniform update of %llu bytes at %#llx not made (not 4-aligned or not in GPU-visible memory)", (unsigned long long)n, (unsigned long long)dst); }
            break;
        }
        case OP_COPY_B2B: {   // {src, dst, n}: CopyBufferToBuffer
            uint64_t src, dst, n; memcpy(&src, a, 8); memcpy(&dst, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8);
            VkBuffer sb, db; VkDeviceSize so, dof; VkDeviceAddress g;
            if (n && vkm_lookup((const void *)(uintptr_t)src, n, &sb, &so, &g) && vkm_lookup((const void *)(uintptr_t)dst, n, &db, &dof, &g)) {
                rp_end(c); VkBufferCopy bc = {so, dof, n}; vkCmdCopyBuffer(c, sb, db, 1, &bc); full_barrier(c);
            } else { static int w; if (!w++) LOG("[RED] vulkan replay: buffer copy of %llu bytes %#llx -> %#llx not made (not in GPU-visible memory)", (unsigned long long)n, (unsigned long long)src, (unsigned long long)dst); }
            break;
        }
        case OP_BIND_STATE: vkx_bind_state(a); break;
        case OP_VIEWPORTS: vkx_viewports(a); break;
        case OP_SCISSOR: vkx_scissor(a); break;
        case OP_DEPTH_RANGE: vkx_depth_range(a); break;
        case OP_STENCIL: vkx_stencil(a); break;
        case OP_POLY_OFFSET: vkx_poly_offset(a); break;
        case OP_BLEND_COLOR: vkx_blend_color(a); break;
        case OP_RAST_DISCARD: vkx_rast_discard(((const p_rast_discard *)a)->on); break;
        case OP_DEPTH_CLAMP: vkx_depth_clamp(((const p_depth_clamp *)a)->on); break;
        case OP_PRIM_RESTART: vkx_prim_restart((int)((const p_prim_restart *)a)->on); break;
        case OP_BIND_VATTR: vkx_vattr(a); break;
        case OP_BIND_VSTREAM: vkx_vstream(a); break;
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tv; vkx_vbuf(vbuf_rec(h.op, a, &tv)); break; }
        case OP_SIGNAL_EVENT: { gpost o = {.kind = 1}; memcpy(&o.val, a, 8); memcpy(&o.mode, (const uint8_t *)a + 8, 4); memcpy(&o.v, (const uint8_t *)a + 12, 4); gp_add(o); break; }
        case OP_REPORT_COUNTER: { gpost o = {.kind = 2, .q = -1}; memcpy(&o.type, a, 8); memcpy(&o.addr, (const uint8_t *)a + 8, 8);
            if (o.type == 0 || o.type == 0x10) o.q = seg_query(c, o.type);
            if (DM_GAME_RENDER_COND && o.type != 0 && o.type != 0x10) rc_occ_add(o.addr);   // its value is not known at replay (rc_value)
            // Samples passed: the replay makes no occlusion queries yet, but a report that is never written stalls a game that waits for it (Ocarina of Time at the
            // first Kokiri Forest frame: no present after the first skipped mesh draw). nvn_mtl.m writes real visibility results and notes that always writing 0 means
            // "fully occluded" (campfires culled); so the report is written when the GPU is done, as "visible" (gp_run: exact Android-parity count 1), which only costs culled work
            if (o.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) { static int w; if (!w++) LOG("vulkan replay: samples-passed counter reports use conservative visible placeholders (no occlusion queries yet)"); atomic_fetch_add(&g_samples_placeholder, 1); }
            gp_add(o); break; }
        case OP_FENCE_SYNC: { sync_t *sy; memcpy(&sy, a, 8); sy->fence = atomic_load(&g_fence) + 1; sy->fenced = 1; break; }   // done when the part being recorded is
        case OP_CLEAR_BUF: {   // ClearBuffer: the range filled with a 32-bit value on the GPU
            const p_clear_buf *cb = a; VkBuffer db; VkDeviceSize dof; VkDeviceAddress g;
            if (cb->size && !(cb->size & 3) && vkm_lookup((const void *)(uintptr_t)cb->addr, cb->size, &db, &dof, &g)) {
                rp_end(c); vkCmdFillBuffer(c, db, dof, cb->size, cb->value); full_barrier(c);
            } else { static int w; if (!w++) LOG("[RED] vulkan replay: buffer clear of %llu bytes at %#llx not made (not 4-aligned or not in GPU-visible memory)", (unsigned long long)cb->size, (unsigned long long)cb->addr); }
            break;
        }
        case OP_PUSH_CONST: {   // NVN2 push constants: the bytes land in the stage group's block (vk_push.h), bound as c[24] at the draw / dispatch (vk_draw.c push_fill)
            const p_push_const *b = a;
            if (vkx_push_apply(&g_push, b)) atomic_fetch_add(&g_push_applied, 1);
            else SKIP_ONCE(g_push_dropped, "push constants of %u bytes at %u exceed the %u-byte block: ignored", b->size, b->offset, DMSH_PUSH_SIZE);
            break; }
        case OP_MULTI_DRAW: SKIP_ONCE(g_skip_multi, "MultiDrawArraysIndirectCount record skipped: not implemented"); break;
        case OP_DLSS: SKIP_ONCE(g_skip_dlss, "DLSS record ignored (no upscaler): the game draws over its input picture"); break;
        case OP_DRAW_TEX: SKIP_ONCE(g_skip_draw_tex, "DrawTexture record skipped: textured rectangle lowering is not implemented"); break;
        case OP_ALPHA_REF: SKIP_ONCE(g_skip_alpha, "alpha-reference state ignored: alpha-test shader plumbing is not implemented"); break;
        default: break;   // counted only, for now
        }
        if (g_prof) g_opns[h.op < 64 ? h.op : 63] += mono_ns() - pt;
        if (h.size < 8) { LOG("[RED] vulkan replay: record of op %u with size %u: the stream is broken", h.op, h.size); return; }
        p += h.size;
    }
}
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }

// ---- present ----
static void shot_check(const tex_t *t);
static int final_select(const tex_t *presented, tex_t *out) {
    int mode = final_mode();
    if (!mode || !presented || !presented->serial) return 0;
    int found = 0; uint32_t dw = 0, dh = 0;
    pthread_mutex_lock(&g_tmu);
    vtex *dst = tex_find(presented->serial);
    if (dst) { dw = dst->w; dh = dst->h; }
    pthread_mutex_unlock(&g_tmu);
    pthread_mutex_lock(&g_final_mu);
    uint32_t n = g_final_n > FINAL_HISTORY ? FINAL_HISTORY : g_final_n;
    for (uint32_t j = 0; dw && dh && j < n; j++) {
        final_candidate *v = &g_final_hist[(g_final_n - 1 - j) % FINAL_HISTORY];
        if ((mode == 2 && !v->depth) || v->tex.serial == presented->serial ||
            (uint64_t)v->w * 4 > (uint64_t)dw * 3 || (uint64_t)v->h * 4 > (uint64_t)dh * 3 ||
            (uint64_t)v->w * 5 < dw || (uint64_t)v->h * 5 < dh) continue;
        uint64_t a = (uint64_t)v->w * dh, b = (uint64_t)v->h * dw;
        uint64_t delta = a > b ? a - b : b - a;
        if (!b || delta * 100 > b) continue;
        pthread_mutex_lock(&g_tmu);
        vtex *src = tex_find(v->tex.serial);
        int usable = src && src->gpu_owned && (src->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        pthread_mutex_unlock(&g_tmu);
        if (!usable) continue;
        *out = v->tex; found = 1;
        static _Atomic uint64_t selected_n;
        uint64_t k = atomic_fetch_add(&selected_n, 1) + 1;
        if (k == 1 || !(k & (k - 1)))
            LOG("vulkan replay: output bypass selected serial %llu (%ux%u, %s) for %llu (%ux%u), present %llu",
                (unsigned long long)v->tex.serial, v->w, v->h, v->depth ? "depth-backed" : "post",
                (unsigned long long)presented->serial, dw, dh, (unsigned long long)k);
        break;
    }
    g_final_n = 0;
    pthread_mutex_unlock(&g_final_mu);
    return found;
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    tex_t selected; const tex_t *show = g_on > 0 && final_select(t, &selected) ? &selected : t;
    shot_check(show);
    if (vkw_active()) return vkw_present(g_on > 0 ? show : NULL, g_on > 0 ? atomic_load(&g_fence) : 0, crop, period_s);
    static _Atomic uint64_t n; uint64_t k = atomic_fetch_add(&n, 1) + 1;
    if (k == 1 || !(k & (k - 1))) LOG("[headless] present %llu (no window: frames are not drawn)", (unsigned long long)k);
    return k;
}

// ---- screenshots: DM_CTL "shot <path>" (dm_gfx_shot_at) and DM_SHOT=<present number>:<path> ----
// The presented texture is blitted to an RGBA8 image (any colour format converts), copied to host memory once the frame's submissions are done, and written as a PNG
// (stored deflate blocks: no compressor needed)
static uint32_t crc_tab[256];
static uint32_t crc32b(uint32_t c, const uint8_t *p, size_t n) {
    if (!crc_tab[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t k = i; for (int j = 0; j < 8; j++) k = k & 1 ? 0xEDB88320u ^ (k >> 1) : k >> 1; crc_tab[i] = k; }
    c = ~c; while (n--) c = crc_tab[(c ^ *p++) & 255] ^ (c >> 8); return ~c;
}
static void put32(FILE *f, uint32_t v) { uint8_t b[4] = {v >> 24, v >> 16, v >> 8, v}; fwrite(b, 1, 4, f); }
static void chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    put32(f, n); fwrite(type, 1, 4, f); if (n) fwrite(d, 1, n, f);
    uint32_t c = crc32b(0, (const uint8_t *)type, 4); c = crc32b(c, d, n); put32(f, c);
}
static int write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    size_t raw = (size_t)h * (1 + 4 * (size_t)w), nblk = (raw + 65534) / 65535, zn = 2 + raw + 5 * nblk + 4;
    uint8_t *z = malloc(zn), *q = z; if (!z) return 0;
    *q++ = 0x78; *q++ = 0x01;
    uint32_t a = 1, b = 0; size_t left = raw, row = 1 + 4 * (size_t)w, pos = 0;
    while (left) {
        uint16_t n = left > 65535 ? 65535 : (uint16_t)left; left -= n;
        *q++ = left ? 0 : 1; *q++ = n & 255; *q++ = n >> 8; *q++ = ~n & 255; *q++ = (uint16_t)~n >> 8;
        for (uint16_t i = 0; i < n; i++, pos++) {
            size_t y = pos / row, x = pos % row; uint8_t v = x ? rgba[y * 4 * w + x - 1] : 0;   // filter byte 0 at each row start
            *q++ = v; a = (a + v) % 65521; b = (b + a) % 65521;
        }
    }
    *q++ = b >> 8; *q++ = b; *q++ = a >> 8; *q++ = a;
    FILE *f = fopen(path, "wb"); if (!f) { free(z); return 0; }
    fwrite("\x89PNG\r\n\x1a\n", 1, 8, f);
    uint8_t ih[13] = {w >> 24, w >> 16, w >> 8, w, h >> 24, h >> 16, h >> 8, h, 8, 6, 0, 0, 0};
    chunk(f, "IHDR", ih, 13); chunk(f, "IDAT", z, (uint32_t)(q - z)); chunk(f, "IEND", NULL, 0);
    free(z); return !fclose(f);
}
static uint32_t pick_mem(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_vkd.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (g_vkd.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
static void shoot_vt_flip(vtex *t, uint32_t layer, int flip, uint64_t fence, const char *path) {
    if (!t || !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) { LOG("[RED] screenshot %s: no colour image", path); return; }
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vkd.pd, t->fmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) { LOG("[RED] screenshot %s: format %d cannot be blitted to RGBA8 here", path, (int)t->fmt); return; }
    uint32_t w = t->w, h = t->h; VkDevice d = g_vkd.dev;
    // DM_VK_RTFLOAT=1 (debugging): the values as 32-bit floats, unclamped, in <path>.f32 beside the PNG (u32 width, u32 height, then RGBA per pixel)
    static int rtf = -1; if (rtf < 0) rtf = getenv("DM_VK_RTFLOAT") != NULL;
    const uint32_t bpp = rtf ? 16 : 4;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = rtf ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1,
        .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VkImage im; vkCreateImage(d, &ci, NULL, &im); VkMemoryRequirements mr; vkGetImageMemoryRequirements(d, im, &mr);
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory imm; vkAllocateMemory(d, &ma, NULL, &imm); vkBindImageMemory(d, im, imm, 0);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * bpp, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer buf; vkCreateBuffer(d, &bi, NULL, &buf); vkGetBufferMemoryRequirements(d, buf, &mr);
    ma.allocationSize = mr.size; ma.memoryTypeIndex = pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    VkDeviceMemory bm; vkAllocateMemory(d, &ma, NULL, &bm); vkBindBufferMemory(d, buf, bm, 0);
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer c; vkAllocateCommandBuffers(d, &ai, &c);
    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(c, &cbi);
    pthread_mutex_lock(&g_tmu); tex_ready(c, t); pthread_mutex_unlock(&g_tmu);
    VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer < t->layers ? layer : 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    if (flip) { bl.dstOffsets[0].y = (int32_t)h; bl.dstOffsets[1].y = 0; }   // a presented image of a lower-left-origin game is bottom-up
    vkCmdBlitImage(c, t->img, VK_IMAGE_LAYOUT_GENERAL, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(c, im, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &bc);
    VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(c);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &fence};
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ts, .waitSemaphoreCount = fence ? 1 : 0, .pWaitSemaphores = &g_tl, .pWaitDstStageMask = &ws, .commandBufferCount = 1, .pCommandBuffers = &c};
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; vkCreateFence(d, &fci, NULL, &fe);
    pthread_mutex_lock(&g_vkd_qlock); VkResult r = vkQueueSubmit(g_vkd.q, 1, &si, fe); pthread_mutex_unlock(&g_vkd_qlock);
    if (!r) vkWaitForFences(d, 1, &fe, VK_TRUE, UINT64_MAX);
    void *px = NULL; vkMapMemory(d, bm, 0, VK_WHOLE_SIZE, 0, &px);
    int ok = !r && px;
    if (ok && rtf) {
        char fp32[600]; snprintf(fp32, sizeof fp32, "%s.f32", path); FILE *o = fopen(fp32, "wb");
        if (o) { fwrite(&w, 4, 1, o); fwrite(&h, 4, 1, o); fwrite(px, bpp, (size_t)w * h, o); fclose(o); }
        uint8_t *q = malloc((size_t)w * h * 4); const float *f = px;   // the PNG: the floats clamped to 0..1
        if (q) { for (size_t i = 0; i < (size_t)w * h * 4; i++) { float v = f[i]; q[i] = (uint8_t)(v != v || v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + 0.5f); } ok = write_png(path, q, w, h); free(q); }
    } else if (ok) ok = write_png(path, px, w, h);
    LOG("%sscreenshot %s: %ux%u (image format %d)%s", ok ? "" : "[RED] ", path, w, h, (int)t->fmt, ok ? "" : " not written");
    vkUnmapMemory(d, bm); vkDestroyFence(d, fe, NULL); vkFreeCommandBuffers(d, g_pool, 1, &c);
    vkDestroyBuffer(d, buf, NULL); vkFreeMemory(d, bm, NULL); vkDestroyImage(d, im, NULL); vkFreeMemory(d, imm, NULL);
}
static void shoot_vt_layer(vtex *t, uint32_t layer, uint64_t fence, const char *path) { shoot_vt_flip(t, layer, 0, fence, path); }
static void shoot_vt(vtex *t, uint64_t fence, const char *path) { shoot_vt_layer(t, 0, fence, path); }
static void shoot(const tex_t *o, uint64_t fence, const char *path) { pthread_mutex_lock(&g_tmu); vtex *t = tex_get(o); pthread_mutex_unlock(&g_tmu); shoot_vt_flip(t, 0, vkx_origin_ll(), fence, path); }   // as presented
// DM_VK_TEXDUMP: the textures loaded during this submission, once it is done
static void dump_uploads(uint64_t fence) {
    static int n;
    for (int i = 0; i < g_ndump; i++) {
        pthread_mutex_lock(&g_tmu); vtex *t = tex_find(g_dump_ser[i]); pthread_mutex_unlock(&g_tmu);
        char p[1024]; snprintf(p, sizeof p, "%s/tex%03d_ser%llu_%ux%u_vkfmt%d.png", g_dump_dir, ++n, (unsigned long long)g_dump_ser[i], t ? t->w : 0, t ? t->h : 0, t ? (int)t->fmt : 0);
        if (t) shoot_vt(t, fence, p);
    }
    g_ndump = 0;
}
static pthread_mutex_t g_shot_mu = PTHREAD_MUTEX_INITIALIZER; static long g_shot_k; static char g_shot_path[1024]; static uint64_t g_presents;
void dm_gfx_shot_at(long k, const char *path) { pthread_mutex_lock(&g_shot_mu); g_shot_k = (long)g_presents + k; snprintf(g_shot_path, sizeof g_shot_path, "%s", path); pthread_mutex_unlock(&g_shot_mu); }
static int env_shot_init;
static void shot_check(const tex_t *t) {
    if (!env_shot_init) { env_shot_init = 1; const char *e = getenv("DM_SHOT"); char *c; long k; if (e && (k = strtol(e, &c, 10)) > 0 && *c == ':') dm_gfx_shot_at(k, c + 1); }
    pthread_mutex_lock(&g_shot_mu);
    g_presents++;
    { static long pk = -1; if (pk < 0) { const char *e = getenv("DM_VK_PASSLOG"); pk = e ? atol(e) : 0; } g_passlog = pk > 0 && (long)g_presents + 1 == pk; vkx_trace(g_passlog); g_pass0 = g_passes;
      { static int se; if (!se) { se = 1; const char *e = getenv("DM_VK_PASSSTOP"); g_passstop = e ? atol(e) : 0; } } if (g_passlog) LOG("[pass] present %llu begins", (unsigned long long)g_presents + 1); }
    int now = g_shot_k && (long)g_presents >= g_shot_k; char path[1024]; if (now) { snprintf(path, sizeof path, "%s", g_shot_path); g_shot_k = 0; }
    pthread_mutex_unlock(&g_shot_mu);
    if (now && g_on > 0) shoot(t, atomic_load(&g_fence), path);
    // DM_VK_RTDUMP=<present>:<dir>: at that present, every colour image the GPU rendered (render targets), as <dir>/rt_<serial>_<w>x<h>.png
    static long rtk = -1; static char rtdir[900];
    if (rtk < 0) { const char *e = getenv("DM_VK_RTDUMP"); char *c; rtk = 0; if (e && (rtk = strtol(e, &c, 10)) > 0 && *c == ':') snprintf(rtdir, sizeof rtdir, "%s", c + 1); else rtk = 0; }
    if (rtk > 0 && (long)g_presents == rtk && g_on > 0) {
        vtex *all[256]; int na = 0;
        pthread_mutex_lock(&g_tmu); for (int i = 0; i < 4096 && na < 256; i++) for (vtex *t = g_stab[i]; t && na < 256; t = t->next) if (t->gpu_owned && (t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) all[na++] = t; pthread_mutex_unlock(&g_tmu);
        for (int i = 0; i < na; i++) for (uint32_t ly = 0; ly < all[i]->layers && ly < 12; ly++) {   // layered images: each layer (the first 12)
            char p[1100]; snprintf(p, sizeof p, ly ? "%s/rt_%06llu_%ux%u_f%d_L%u.png" : "%s/rt_%06llu_%ux%u_f%d.png", rtdir, (unsigned long long)all[i]->serial, all[i]->w, all[i]->h, (int)all[i]->fmt, ly);
            shoot_vt_layer(all[i], ly, atomic_load(&g_fence), p); }
    }
    // DM_SHOT_EVERY=<n>:<dir>: every n-th presented frame as <dir>/present<number>.png (at most 40)
    static long every = -1; static char edir[900]; static int shots;
    if (every < 0) { const char *e = getenv("DM_SHOT_EVERY"); char *c; every = 0; if (e && (every = strtol(e, &c, 10)) > 0 && *c == ':') snprintf(edir, sizeof edir, "%s", c + 1); else every = 0; }
    if (every > 0 && g_on > 0 && shots < 40 && g_presents % (uint64_t)every == 0) { char p[1000]; snprintf(p, sizeof p, "%s/present%05llu.png", edir, (unsigned long long)g_presents); shots++; shoot(t, atomic_load(&g_fence), p); }
}

// ---- a picture from host memory (an OpenGL game, linux/gl_egl.c) ----
// RGBA8 rows bottom-up, as glReadPixels gives them: shown in the window (vkw_present_rgba), counted as a present for DM_SHOT / DM_SHOT_EVERY, which write it upright
uint64_t vkr_present_pixels(const uint8_t *rgba, uint32_t w, uint32_t h) {
    if (!env_shot_init) { env_shot_init = 1; const char *e = getenv("DM_SHOT"); char *c; long k; if (e && (k = strtol(e, &c, 10)) > 0 && *c == ':') dm_gfx_shot_at(k, c + 1); }
    static long every = -1; static char edir[900]; static int shots;
    if (every < 0) { const char *e = getenv("DM_SHOT_EVERY"); char *c; every = 0; if (e && (every = strtol(e, &c, 10)) > 0 && *c == ':') snprintf(edir, sizeof edir, "%s", c + 1); else every = 0; }
    pthread_mutex_lock(&g_shot_mu);
    uint64_t n = ++g_presents; char path[1024] = "";
    if (g_shot_k && (long)n >= g_shot_k) { snprintf(path, sizeof path, "%s", g_shot_path); g_shot_k = 0; }
    pthread_mutex_unlock(&g_shot_mu);
    if (!path[0] && every > 0 && shots < 40 && n % (uint64_t)every == 0) { snprintf(path, sizeof path, "%s/present%05llu.png", edir, (unsigned long long)n); shots++; }
    if (path[0]) {
        uint8_t *up = malloc((size_t)w * h * 4);
        if (up) { for (uint32_t y = 0; y < h; y++) memcpy(up + (size_t)y * w * 4, rgba + (size_t)(h - 1 - y) * w * 4, (size_t)w * 4);
                  if (write_png(path, up, w, h)) LOG("screenshot %s: %ux%u (OpenGL frame)", path, w, h); free(up); }
    }
    if (vkw_active()) return vkw_present_rgba(rgba, w, h);
    if (n == 1 || !(n & (n - 1))) LOG("[headless] present %llu (no window: frames are not drawn)", (unsigned long long)n);
    return n;
}

// ---- for vk_draw.c ----
vtex *vkr_tex_sampled(const tex_t *o, VkCommandBuffer unused) { (void)unused; return tex_sampled(o); }
vtex *vkr_tex_target(const tex_t *o) { return tex_get(o); }
vtex *vkr_tex_sampled_budgeted(const tex_t *o, uint64_t total_budget, uint64_t single_max) {
    uint64_t old_budget = t_spec_budget, old_max = t_spec_max;
    t_spec_budget = total_budget; t_spec_max = single_max;
    vtex *t = tex_sampled(o);
    t_spec_budget = old_budget; t_spec_max = old_max;
    return t;
}
int vkr_treg(int id, tex_t *t, view_t *v, int *has_v) { treg_t r; if (!treg_get(id, &r)) return 0; *t = r.t; *v = r.v; *has_v = r.has_v; return 1; }
int vkr_sreg(int id, samp_par *p) { pthread_mutex_lock(&g_rmu); int ok = id > 0 && id < g_nsreg && g_sreg_ok[id]; if (ok) *p = g_sreg[id]; pthread_mutex_unlock(&g_rmu); return ok; }
static int reg_take(int *ids, int cap, int *list, int *n, uint8_t *flag) {
    int k = 0;
    pthread_mutex_lock(&g_rmu);
    while (*n && k < cap) { int id = list[--*n]; flag[id] = 0; ids[k++] = id; }
    pthread_mutex_unlock(&g_rmu);
    return k;
}
int vkr_bindless_take_tex_dirty(int *ids, int cap) { return cap > 0 ? reg_take(ids, cap, g_tdirty, &g_ntdirty, g_tdflag) : 0; }
int vkr_bindless_take_samp_dirty(int *ids, int cap) { return cap > 0 ? reg_take(ids, cap, g_sdirty, &g_nsdirty, g_sdflag) : 0; }
uint64_t vkr_submission_count(void) { return atomic_load(&g_subs); }
uint64_t vkr_mesh_issue_count(void) { return atomic_load(&g_mesh_issue_seq); }
uint32_t vkr_pick_mem(uint32_t bits, VkMemoryPropertyFlags want) { return pick_mem(bits, want); }
uint64_t vkr_fence_next(void) { return atomic_load(&g_fence) + 1; }
uint64_t vkr_image_bytes(void) { return atomic_load(&g_image_bytes); }
int vkr_slot(void) { return g_cb_i; }
VkCommandBuffer vkr_cmd(void) { return g_cb[g_cb_i]; }
