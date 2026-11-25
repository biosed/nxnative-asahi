// Device-free regression test for NVN CPU-write texture tracking (g_ft):
// verifies that nvnMemoryPoolFinalize drops registered textures from g_ft and
// serializes with nvn_ft_poll so unmapped storage and freed pool_t are never read.
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "dm.h"
#include "nvn_obj.h"
#include "nvn_protos.h"
#include "nvn_mtl.h"
#include "nvn_dirty.h"

#define mtl_tex_cpu_written null_mtl_tex_cpu_written
#include "../../null/nvn_null.c"
#undef mtl_tex_cpu_written

static _Atomic uint64_t g_cpu_written_calls;
static _Atomic uint64_t g_cpu_written_last_serial;
void mtl_tex_cpu_written(const tex_t *t) {
    atomic_fetch_add(&g_cpu_written_calls, 1);
    atomic_store(&g_cpu_written_last_serial, t ? t->serial : 0);
}

#include "../../../gfx/nvn/hle_nvn.c"

uint64_t g_t0_ns;
uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
uint64_t guest_ns(void) { return now_ns(); }
module_t *module_of(const void *p) { (void)p; return NULL; }
void dm_backtrace(uint64_t fp) { (void)fp; }
void dm_cheats_present(void) {}
int dm_detclock_enabled(void) { return 0; }
void dm_guest_frame_tick(void) {}
void dm_guest_write_prepare(void *addr, size_t len) { (void)addr; (void)len; }
void dm_vi_frame_queued(uint64_t f) { (void)f; }
void dm_vi_layer_crop(int c[4]) { c[0] = c[1] = c[2] = c[3] = 0; }
int front_screen_max_fps(void) { return 60; }
void dm_res_report_register(void (*fn)(uint32_t tick, const char *why)) { (void)fn; }
int dm_suspend_check(void) { return 0; }
void nvn_unimpl_entry(void) {}
void nvn_big_entry(void) {}
void nvn_prof_entry(void) {}
void svc_thread_dump(void) {}
void dm_sys_prof(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { (void)a; (void)b; (void)c; (void)d; }
_Atomic uint64_t dm_present_ns;
uint8_t *dm_fps_state;
struct dm_flags g_flag;

static int (*g_fault_fn)(void *, int, void *);
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) { g_fault_fn = fn; }
void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len)) { (void)fn; }

static void segv_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig;
    if (g_fault_fn && g_fault_fn(si->si_addr, 1, uc)) return;
    fprintf(stderr, "t_nvn_ft: unhandled SIGSEGV at %p\n", si->si_addr);
    _exit(2);
}

static void install_fault_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = segv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, NULL) != 0) { perror("sigaction"); exit(2); }
}

int dm_guest_as_contains(uintptr_t address) { (void)address; return 0; }

static int bad;
static void check(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    install_fault_handler();

    NVNdevice dev;
    NVNdeviceBuilder db;
    h_nvnDeviceBuilderSetDefaults(&db);
    check(h_nvnDeviceInitialize(&dev, &db), "device initializes");

    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    size_t pool_sz = 4 * pg > 0x40000 ? 4 * pg : 0x40000;
    uint8_t *mem = mmap(NULL, pool_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(mem != MAP_FAILED, "pool backing maps");

    NVNmemoryPool *pool = calloc(1, sizeof *pool);
    check(pool != NULL, "pool struct allocates");
    NVNmemoryPoolBuilder pb;
    h_nvnMemoryPoolBuilderSetDefaults(&pb);
    h_nvnMemoryPoolBuilderSetDevice(&pb, &dev);
    h_nvnMemoryPoolBuilderSetStorage(&pb, mem, pool_sz);
    h_nvnMemoryPoolBuilderSetFlags(&pb, NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
    check(h_nvnMemoryPoolInitialize(pool, &pb), "memory pool initializes");

    NVNtextureBuilder tb;
    h_nvnTextureBuilderSetDefaults(&tb);
    h_nvnTextureBuilderSetDevice(&tb, &dev);
    h_nvnTextureBuilderSetTarget(&tb, NVN_TEXTURE_TARGET_2D);
    h_nvnTextureBuilderSetFormat(&tb, NVN_FORMAT_RGBA8);
    h_nvnTextureBuilderSetSize2D(&tb, 256, 256);
    h_nvnTextureBuilderSetStorage(&tb, pool, 0);

    NVNtexture tex;
    check(h_nvnTextureInitialize(&tex, &tb), "texture initializes");
    uint64_t serial = ((const tex_t *)&tex)->serial;
    check(nvn_ft_tracked(serial), "new texture on CPU_CACHED pool is tracked in g_ft");

    mem[0] = 0x5a;
    h_nvnMemoryPoolFinalize(pool);
    munmap(mem, pool_sz);
    free(pool);

    check(!nvn_ft_tracked(serial), "pool finalize drops texture from g_ft");
    nvn_ft_poll();
    check(atomic_load(&g_cpu_written_calls) == 0,
          "nvn_ft_poll skips texture from finalized pool without touching unmapped storage");

    printf("t_nvn_ft: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
