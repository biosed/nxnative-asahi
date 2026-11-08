// The mtl_* calls current nxnative (48d0b37) added to the backend interface after nxasahi's base (nxnative 40b643b): statistics the NVN layer's periodic
// log line reads, pacing hooks of the display link, program prefetch, queue bookkeeping and memory hints. The nxasahi Vulkan replay (vk_replay.c, vk_window.c)
// has no state behind them yet, so each answers as the null backend does -- zeros, "nothing outstanding", no-op -- which keeps hle_nvn.c's callers on their
// existing paths. Each becomes real when the replay grows the feature (the comment on each names the Metal behaviour it stands in for).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "vk_draw.h"
#include "vk_mem.h"
#include "vk_replay.h"
#include "vk_window.h"

// ---- statistics for hle_nvn.c's log line (nvn_mtl.m counts these in its replay) ----
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = *full = 0; }                       // barriers skipped / issued in full
uint64_t mtl_blit_splits(void) { return 0; }                                                        // copies split by the blitter's limits
uint64_t mtl_copy_cut_count(void) { return 0; }                                                     // copies cut at a pass boundary
unsigned mtl_cpu_tex_count(void) { return 0; }                                                      // textures read back for the CPU
uint64_t mtl_cut_count(void) { return 0; }                                                          // render passes cut
void mtl_cshoist_stats(uint64_t *moved, uint64_t *depth, uint64_t stops[8]) { *moved = *depth = 0; memset(stops, 0, 8 * sizeof *stops); }   // compute hoisting
void mtl_fb_stats(uint64_t *snaps, uint64_t *reused, uint64_t *tile, uint64_t *waited) { *snaps = *reused = *tile = *waited = 0; }          // same-pass feedback
void mtl_geohoist_stats(uint64_t *moved, uint64_t *runs, uint64_t stops[6]) { *moved = *runs = 0; memset(stops, 0, 6 * sizeof *stops); }   // geometry hoisting
void mtl_geomesh_stats(uint64_t *made, uint64_t *waited, uint64_t *run_draws, uint64_t *runs, uint64_t ends[4]) {                           // geometry-as-mesh runs
    *made = *waited = *run_draws = *runs = 0; memset(ends, 0, 4 * sizeof *ends);
}
void mtl_lc_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }                                        // load/clear merges
void mtl_mem_stat(unsigned long long a, unsigned long long b) { (void)a; (void)b; }                 // the host's memory figures, for the log line
void mtl_res_stats(char *out, size_t n) { if (n) snprintf(out, n, "vulkan (nxasahi replay)"); }    // resource summary text
void mtl_sm_stats(uint64_t *passes, uint64_t *contended, uint64_t *waited) { *passes = *contended = *waited = 0; }   // submit-lock contention
void mtl_ubd_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }                                       // uniform-block diffing
void mtl_ubp_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = *c = 0; }                     // in-pass uniform partial copies (ubp_get)
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }

// ---- display link pacing (nvn_pace.h): the SDL window presents with FIFO vsync, so there is no link to tick; "no tick" keeps the pacer on its timer path ----
int mtl_link_rate(void) { return 60; }                                                              // the display's rate (front_screen_max_fps covers the window)
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) {
    (void)n; if (target) *target = 0; if (lead_ms) *lead_ms = 0; if (ticks) *ticks = 0; if (gap_ms) *gap_ms = 0; return 0;
}
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_probe_ramp_on(void) { return 0; }                                                           // DM_FSTEP_PROBE_RAMP: no frame-step probe ramp here
// Presents in flight for hle_vi.c are the SDL/Vulkan present thread's ticket counters.
void mtl_present_progress(uint64_t *queued, uint64_t *done) {
    if (vkw_active()) vkw_present_progress(queued, done); else *queued = *done = 0;
}
void mtl_present_layer(uint32_t w, uint32_t h) { vkw_present_layer(w, h); }
void mtl_present_screen(uint32_t w, uint32_t h) { vkw_present_screen(w, h); }
void mtl_prepare(void) {}                                                                           // the "Preparing shaders" warm-up: pipelines are built on first use

// ---- programs, queues, images ----
void mtl_prog_prefetch(const prog_t *p) { (void)p; }                                                // background shader-library load: done on first draw instead
void mtl_prog_forget(const prog_t *p) { vkx_prog_forget(p); }
static uint64_t g_active_queue;
void mtl_queue_activate(uint64_t serial) { g_active_queue = serial; vkx_queue_select(serial); }   // the queue about to submit: its own bound state (vk_draw.c)
void mtl_queue_forget(uint64_t serial) { if (g_active_queue == serial) g_active_queue = 0; vkx_queue_forget(serial); }
void mtl_set_queue_origin_lower_left(int on) { mtl_set_origin_lower_left(on); }
void mtl_reg_image(int id, const tex_t *t, const view_t *v) {
    if (!t) { mtl_reg_tex(id, NULL, v); return; }
    tex_t image = *t; image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; mtl_reg_tex(id, &image, v);
}
int mtl_tex_gpu_owned(uint64_t serial) { return vkr_tex_gpu_owned(serial); }

// ---- memory hints ----
void mtl_pool_addressed(const void *mem, uint64_t size) {
    // In pool mode the heap/alias reservations stay anonymous until NVN gives
    // the GPU an address. This is the same semantic boundary as the modern
    // Android backend: ordinary texture-only storage is copied into VkImages,
    // while buffers/raw-global ranges are placed here before the address can
    // reach a command buffer or shader.
    vkm_addressed(mem, size);
    if (!vkm_pool_commit(mem, size)) {
        static _Atomic int nred;
        if (size && atomic_fetch_add(&nred, 1) < 8)
            LOG("[RED] vulkan: addressed NVN memory %p+%#llx is outside placed guest memory", mem, (unsigned long long)size);
    }
}
void mtl_pool_gpu_written(const void *pool) {
    pool_t *p = (pool_t *)pool;
    if (p) atomic_store_explicit(&p->gpu_written, 1, memory_order_relaxed);
}
// vk_mem.c intentionally retains placed-memory chunks for the process lifetime.
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
