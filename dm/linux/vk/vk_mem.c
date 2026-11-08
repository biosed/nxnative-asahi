// Linux build: guest memory backed by Vulkan memory (vk_mem.h; the design is checked on the device by tests/t_placed.c).
// hle_svc.c names its regions (dm_gpu_region) and calls dm_gpu_commit before it opens memory in them. A chunk is backed the first time a commit touches it:
// device memory is mapped in place over the chunk (VK_EXT_map_memory_placed replaces the reservation's pages), and bound into the region's sparse buffer at the
// chunk's offset. Pages that were readable keep their bytes (copied out first, through process_vm_readv, which fails on an inaccessible page rather than faulting)
// and their access; the others are zeroed, as fresh pages are, and made inaccessible again, so hle_svc.c's own protection of the region is unchanged.
// The device exists before the game runs (front_sdl.c), so in practice no page of a chunk is open before it is backed, and no other thread can write it meanwhile.
// Chunks stay backed for the rest of the run (open: give back a chunk none of whose pages is committed)
#include "dm.h"
#include "dmsh_abi.h"
#include "vk_dev.h"
#include "vk_mem.h"
#include "nvn_dirty.h"
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>

#define VKM_CHUNK_DEFAULT (32ull << 20)
#define VKM_MAXREG 4
typedef struct {
    uint8_t *base; uint64_t size;
    VkBuffer buf; VkDeviceAddress gpu; uint32_t type;   // made when the first chunk is backed
    VkDeviceMemory *chunk;                               // one per vkm_granule(), VK_NULL_HANDLE until backed
    uint8_t *referenced;                                 // chunks reached by an explicit renderer GPU buffer lookup
    uint8_t *addressed;                                  // chunks in ranges NVN has handed to the GPU
} vkm_region;
static vkm_region g_reg[VKM_MAXREG]; static int g_nreg;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_chunks, g_bytes; static _Atomic uint64_t g_refchunks;
static _Atomic uint64_t g_generation = 1;
enum { VKM_PROFILE_CHUNKS = 4096 };
static uint8_t g_profile[VKM_MAXREG][VKM_PROFILE_CHUNKS];
static int g_profile_loaded, g_profile_on;

// A learned profile is a set of 32-MiB region/chunk pairs, one per line. It is
// consumed only by the normal pre-open dm_gpu_commit path: selected chunks are
// placed before SetHeapSize/MapPhysicalMemory opens their first CPU page. No
// live mapping is ever replaced in this mode.
static int profile_mode(void) {
    if (g_profile_loaded) return g_profile_on;
    g_profile_loaded = 1;
    const char *path = getenv("DM_VK_GMEM_PROFILE");
    if (!path || !*path) return 0;
    FILE *f = fopen(path, "re");
    if (!f) { LOG("[RED] vulkan: cannot open guest-memory profile %s (errno=%d); using full backing", path, errno); return 0; }
    char line[256]; unsigned n = 0;
    while (fgets(line, sizeof line, f)) {
        int region = -1; unsigned long long first = 0, last = 0;
        if (line[0] == '#' || line[0] == '\n') continue;
        int fields = sscanf(line, "%d %llu-%llu", &region, &first, &last);
        if (fields == 2) last = first;
        if (fields < 2 || region < 0 || region >= VKM_MAXREG || first > last || last >= VKM_PROFILE_CHUNKS) {
            LOG("[RED] vulkan: bad guest-memory profile line: %s", line); fclose(f); memset(g_profile, 0, sizeof g_profile); return 0;
        }
        for (unsigned long long chunk = first; chunk <= last; chunk++)
            if (!g_profile[region][chunk]) { g_profile[region][chunk] = 1; n++; }
    }
    fclose(f);
    g_profile_on = n != 0;
    if (g_profile_on) LOG("vulkan: guest-memory profile %s selects %u pre-open 32-MiB chunks", path, n);
    else LOG("[RED] vulkan: guest-memory profile %s is empty; using full backing", path);
    return g_profile_on;
}

// The original Linux bring-up backs every byte accepted by SetHeapSize and
// MapPhysicalMemory.  A Switch title commonly reserves far more CPU heap than
// it gives to NVN, and a host-visible VkDeviceMemory allocation commits all of
// it. Pool mode keeps those reservations anonymous and backs a granule only
// when NVN gives its address to the GPU (mtl_pool_addressed).
static int pool_mode(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_GMEM_POOLS"); v = e && strcmp(e, "0"); }
    return v;
}

uint64_t vkm_granule(void) {
    static uint64_t v;
    if (!v) {
        v = VKM_CHUNK_DEFAULT;
        const char *e = getenv("DM_VK_GMEM_GRANULE_MB");
        if (pool_mode() && e && *e) {
            char *end = NULL; unsigned long long mb = strtoull(e, &end, 10);
            uint64_t n = mb > UINT64_MAX >> 20 ? 0 : (uint64_t)mb << 20;
            if (end != e && !*end && n >= (1ull << 20) && !(n & (n - 1))) v = n;
            else LOG("[RED] vulkan: DM_VK_GMEM_GRANULE_MB=%s is not a power of two MiB; using 32 MiB", e);
        }
    }
    return v;
}

// Fresh Honeykrisp host-visible allocations are demand-zero shmem.  Leaving a
// closed guest page untouched keeps it uncommitted until either the CPU opens
// it or the GPU actually reads it.  The original bring-up explicitly wrote
// zero to every closed page in every 32-MiB chunk, which charged the complete
// 6.2-GiB heap and every mapped alias chunk even when the title never used
// those pages.  This is opt-in until the physical selftest and title have both
// exercised it; open pages are still copied byte-for-byte.
static int demand_zero_mode(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DM_VK_GMEM_DEMAND_ZERO");
        v = e && strcmp(e, "0");
        if (v) LOG("vulkan: closed guest pages stay demand-zero instead of being prefaulted (DM_VK_GMEM_DEMAND_ZERO=1)");
    }
    return v;
}

// A late placed map must not lose a CPU write or briefly expose zeroes.  Make
// the open pages read-only, fill the allocation through an ordinary temporary
// map, then place the already-filled allocation over the old pages in one map.
// Guest writes fault while this range is read-only; host-side writes use the
// companion prepare hook.  Both wait until the atomic replacement is done.
static _Atomic uintptr_t g_swap_lo, g_swap_hi;
static _Atomic long g_swap_owner;
static long this_tid(void) { return (long)syscall(SYS_gettid); }
static int swap_overlap(uintptr_t lo, uintptr_t hi) {
    uintptr_t shi = atomic_load_explicit(&g_swap_hi, memory_order_acquire);
    uintptr_t slo = atomic_load_explicit(&g_swap_lo, memory_order_relaxed);
    return slo < shi && lo < shi && hi > slo;
}
static int swap_fault(void *addr, int is_write, void *uc) {
    (void)uc;
    uintptr_t a = (uintptr_t)addr;
    if (!is_write || !swap_overlap(a, a + 1)) return 0;
    if (atomic_load_explicit(&g_swap_owner, memory_order_relaxed) == this_tid()) return 0;
    while (swap_overlap(a, a + 1)) sched_yield();
    return 1;
}
static void swap_prepare(void *addr, size_t len) {
    uintptr_t lo = (uintptr_t)addr, hi = lo + len;
    if (hi < lo) hi = UINTPTR_MAX;
    if (atomic_load_explicit(&g_swap_owner, memory_order_relaxed) == this_tid()) return;
    while (len && swap_overlap(lo, hi)) sched_yield();
}
static void swap_hooks_once(void) {
    dm_fault_hook_register(swap_fault);
    dm_guest_write_prep_register(swap_prepare);
}
static void swap_hooks(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, swap_hooks_once); }
static void swap_begin(uintptr_t lo, uintptr_t hi) {
    atomic_store_explicit(&g_swap_lo, lo, memory_order_relaxed);
    atomic_store_explicit(&g_swap_owner, this_tid(), memory_order_relaxed);
    atomic_store_explicit(&g_swap_hi, hi, memory_order_release);
}
static void swap_end(void) {
    atomic_store_explicit(&g_swap_hi, 0, memory_order_release);
    atomic_store_explicit(&g_swap_lo, 0, memory_order_relaxed);
    atomic_store_explicit(&g_swap_owner, 0, memory_order_relaxed);
}

static void protect_runs(uint8_t *at, const uint8_t *open, size_t npg, size_t pg, int open_pages, int prot) {
    for (size_t p = 0; p < npg;) {
        if (!!open[p] != !!open_pages) { p++; continue; }
        size_t q = p + 1; while (q < npg && !!open[q] == !!open_pages) q++;
        if (mprotect(at + p * pg, (q - p) * pg, prot))
            die("vulkan: guest chunk %p: mprotect errno=%d", (void *)at, errno);
        p = q;
    }
}

static void ref_report(void) {
    uint64_t granule = vkm_granule();
    LOG("vulkan: guest memory explicit GPU references reached %llu of %llu backed chunks (%llu of %llu MiB); raw-global pointer reads are additional",
        (unsigned long long)atomic_load(&g_refchunks), (unsigned long long)g_chunks,
        (unsigned long long)(atomic_load(&g_refchunks) * (granule >> 20)),
        (unsigned long long)(g_bytes >> 20));
}

void dm_gpu_region(void *base, size_t size) {
    static int report_registered;
    static int pool_reported;
    if (pool_mode()) {
        swap_hooks();
        if (!pool_reported++) LOG("vulkan: guest heap/alias use anonymous pages until NVN memory pools expose them (DM_VK_GMEM_POOLS=1)");
    }
    pthread_mutex_lock(&g_mu);
    if (g_nreg < VKM_MAXREG && size) {
        vkm_region *r = &g_reg[g_nreg++];
        r->base = base; r->size = size;
        uint64_t granule = vkm_granule();
        size_t n = (size + granule - 1) / granule;
        r->chunk = calloc(n, sizeof *r->chunk);
        r->referenced = calloc(n, 1);
        r->addressed = calloc(n, 1);
        if (!report_registered) { atexit(ref_report); report_registered = 1; }
    }
    pthread_mutex_unlock(&g_mu);
}

void vkm_addressed(const void *addr, uint64_t len) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("DM_VK_GMEM_ADDR_LOG") != NULL;
    if (!addr || !len) return;
    const uint8_t *a = addr; uint64_t granule = vkm_granule();
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nreg; i++) {
        vkm_region *r = &g_reg[i];
        if (a < r->base || (uint64_t)(a - r->base) >= r->size) continue;
        uint64_t lo = (uint64_t)(a - r->base), hi = len > r->size - lo ? r->size : lo + len;
        for (uint64_t c = lo / granule; c * granule < hi; c++) {
            if (!r->addressed[c]) {
                r->addressed[c] = 1;
                if (enabled) LOG("[gmem-profile] region %d chunk %llu (offset %#llx, %llu MiB granule)", i,
                                 (unsigned long long)c, (unsigned long long)(c * granule),
                                 (unsigned long long)(granule >> 20));
            }
            if (profile_mode() && !r->chunk[c]) {
                static _Atomic int misses;
                if (atomic_fetch_add(&misses, 1) < 32)
                    LOG("[RED] vulkan: guest-memory profile miss: region %d chunk %llu (offset %#llx)", i,
                        (unsigned long long)c, (unsigned long long)(c * granule));
            }
        }
        break;
    }
    pthread_mutex_unlock(&g_mu);
}

static int region_buffer(vkm_region *r) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT, .size = r->size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult vr = vkCreateBuffer(g_vkd.dev, &bi, NULL, &r->buf);
    if (vr) { LOG("[RED] vulkan: guest memory region %p+%#llx: no sparse buffer (%d)", (void *)r->base, (unsigned long long)r->size, (int)vr); return 0; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, r->buf, &mr);
    // host visible, coherent and cached: the CPU side is the game's own memory
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    r->type = UINT32_MAX;
    for (uint32_t i = 0; i < g_vkd.mem.memoryTypeCount && r->type == UINT32_MAX; i++)
        if ((mr.memoryTypeBits >> i & 1) && (g_vkd.mem.memoryTypes[i].propertyFlags & want) == want) r->type = i;
    if (r->type == UINT32_MAX || (vkm_granule() % mr.alignment) || ((uintptr_t)r->base % mr.alignment)) {
        LOG("[RED] vulkan: guest memory region %p+%#llx: no host-visible coherent cached memory type, or sparse alignment %#llx does not fit", (void *)r->base,
            (unsigned long long)r->size, (unsigned long long)mr.alignment);
        vkDestroyBuffer(g_vkd.dev, r->buf, NULL); r->buf = VK_NULL_HANDLE; return 0;
    }
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = r->buf};
    r->gpu = vkGetBufferDeviceAddress(g_vkd.dev, &ai);
    LOG("vulkan: guest memory region %p+%#llx is GPU buffer %#llx (memory type %u)", (void *)r->base, (unsigned long long)r->size, (unsigned long long)r->gpu, r->type);
    return 1;
}

static int back_chunk(vkm_region *r, uint64_t ci) {
    uint64_t granule = vkm_granule();
    uint8_t *at = r->base + ci * granule;
    uint64_t len = r->size - ci * granule < granule ? r->size - ci * granule : granule;
    size_t pg = (size_t)getpagesize(), npg = len / pg;
    int swapping = pool_mode();
    if (swapping) { nvn_dirty_hold(at, len); swap_begin((uintptr_t)at, (uintptr_t)at + len); }
    // the pages that are open now, and their bytes
    uint8_t *open = calloc(npg, 1), *save = NULL; size_t nopen = 0;
    for (size_t p = 0; p < npg; p++) {
        static __thread uint8_t probe;
        struct iovec l = {&probe, 1}, rv = {at + p * pg, 1};
        if (process_vm_readv(getpid(), &l, 1, &rv, 1, 0) == 1) { open[p] = 1; nopen++; }
    }
    if (swapping) protect_runs(at, open, npg, pg, 1, PROT_READ);
    else if (nopen) {
        save = malloc(nopen * pg); size_t k = 0;
        for (size_t p = 0; p < npg; p++) if (open[p]) memcpy(save + pg * k++, at + p * pg, pg);
    }
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = len, .memoryTypeIndex = r->type};
    VkDeviceMemory m; VkResult vr = vkAllocateMemory(g_vkd.dev, &ma, NULL, &m);
    if (vr) {
        if (swapping) { protect_runs(at, open, npg, pg, 1, PROT_READ | PROT_WRITE); swap_end(); nvn_dirty_unhold(); }
        LOG("[RED] vulkan: no device memory for guest chunk %p+%#llx (%d)", (void *)at, (unsigned long long)len, (int)vr); free(open); free(save); return 0;
    }
    if (swapping) {
        // Populate away from the guest address.  Mapping it placed only after
        // it is complete makes concurrent readers see old or new bytes.
        void *tmp = NULL; vr = vkMapMemory(g_vkd.dev, m, 0, VK_WHOLE_SIZE, 0, &tmp);
        if (vr || !tmp) {
            vkFreeMemory(g_vkd.dev, m, NULL);
            protect_runs(at, open, npg, pg, 1, PROT_READ | PROT_WRITE); swap_end(); nvn_dirty_unhold();
            LOG("[RED] vulkan: temporary map for guest chunk %p+%#llx failed (%d)", (void *)at, (unsigned long long)len, (int)vr); free(open); return 0;
        }
        for (size_t p = 0; p < npg; p++) {
            if (open[p]) memcpy((uint8_t *)tmp + p * pg, at + p * pg, pg);
            else if (!demand_zero_mode()) memset((uint8_t *)tmp + p * pg, 0, pg);
        }
        vkUnmapMemory(g_vkd.dev, m);
    }
    VkMemoryMapPlacedInfoEXT pl = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT, .pPlacedAddress = at};
    VkMemoryMapInfoKHR mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, .pNext = &pl, .flags = VK_MEMORY_MAP_PLACED_BIT_EXT, .memory = m, .offset = 0, .size = VK_WHOLE_SIZE};
    void *p0 = NULL; vr = g_vkd.map2(g_vkd.dev, &mi, &p0);
    if (vr || p0 != at) die("vulkan: placing guest chunk %p+%#llx failed (%d, at %p): the region's pages are gone", (void *)at, (unsigned long long)len, (int)vr, p0);
    if (!swapping) {
        size_t k = 0;
        for (size_t p = 0; p < npg; p++) {
            if (open[p]) memcpy(at + p * pg, save + pg * k++, pg);
            else if (!demand_zero_mode()) memset(at + p * pg, 0, pg);
        }
    }
    protect_runs(at, open, npg, pg, 0, PROT_NONE);
    if (swapping) { swap_end(); nvn_dirty_unhold(); }
    free(open); free(save);
    VkSparseMemoryBind sb = {ci * granule, len, m, 0, 0};
    VkSparseBufferMemoryBindInfo sbi = {r->buf, 1, &sb};
    VkBindSparseInfo bsi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .bufferBindCount = 1, .pBufferBinds = &sbi};
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; vkCreateFence(g_vkd.dev, &fci, NULL, &f);
    pthread_mutex_lock(&g_vkd_qlock); vr = vkQueueBindSparse(g_vkd.q, 1, &bsi, f); pthread_mutex_unlock(&g_vkd_qlock);
    if (!vr) vr = vkWaitForFences(g_vkd.dev, 1, &f, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g_vkd.dev, f, NULL);
    if (vr) die("vulkan: binding guest chunk %p into its region's buffer failed (%d)", (void *)at, (int)vr);
    r->chunk[ci] = m; g_chunks++; g_bytes += len;
    atomic_fetch_add_explicit(&g_generation, 1, memory_order_release);
    if (!(g_chunks & (g_chunks - 1)) || (nopen && granule == VKM_CHUNK_DEFAULT)) LOG("vulkan: guest memory backed: %llu chunks, %llu MiB (chunk %p: %zu of %zu pages were open)", (unsigned long long)g_chunks,
                                                  (unsigned long long)(g_bytes >> 20), (void *)at, nopen, npg);
    return 1;
}

void dm_gpu_commit(void *addr, size_t size) {
    if (!vkd_ok() || !g_vkd.placed || !size) return;
    if (pool_mode()) { swap_prepare(addr, size); return; }
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nreg; i++) {
        vkm_region *r = &g_reg[i]; uint8_t *a = addr;
        if (a < r->base || a - r->base >= (ptrdiff_t)r->size) continue;
        if (!r->buf && !region_buffer(r)) break;
        uint64_t granule = vkm_granule();
        uint64_t lo = (uint64_t)(a - r->base), hi = lo + size > r->size ? r->size : lo + size;
        for (uint64_t c = lo / granule; c * granule < hi; c++) {
            if (profile_mode() && (i >= VKM_MAXREG || c >= VKM_PROFILE_CHUNKS || !g_profile[i][c])) continue;
            if (!r->chunk[c] && !back_chunk(r, c)) break;
        }
        break;
    }
    pthread_mutex_unlock(&g_mu);
}

int vkm_pool_commit(const void *addr, uint64_t len) {
    if (!pool_mode() || !len) return 1;
    if (!vkd_ok() || !g_vkd.placed) return 0;
    const uint8_t *a = addr;
    int hit = 0, ok = 1;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nreg; i++) {
        vkm_region *r = &g_reg[i];
        if (a < r->base || (uint64_t)(a - r->base) >= r->size) continue;
        uint64_t granule = vkm_granule();
        uint64_t lo = (uint64_t)(a - r->base), hi = len > r->size - lo ? r->size : lo + len;
        if (!r->buf && !region_buffer(r)) { ok = 0; break; }
        for (uint64_t c = lo / granule; c * granule < hi; c++)
            if (!r->chunk[c] && !back_chunk(r, c)) { ok = 0; break; }
        hit = 1;
        break;
    }
    pthread_mutex_unlock(&g_mu);
    return hit && ok;
}

int vkm_lookup(const void *addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceAddress *gpu) {
    const uint8_t *a = addr;
    for (int i = 0; i < g_nreg; i++) {
        vkm_region *r = &g_reg[i];
        if (!r->buf || a < r->base || (uint64_t)(a - r->base) + len > r->size) continue;
        uint64_t granule = vkm_granule();
        uint64_t lo = (uint64_t)(a - r->base), hi = lo + (len ? len : 1);
        for (uint64_t c = lo / granule; c * granule < hi; c++) if (!r->chunk[c]) return 0;
        for (uint64_t c = lo / granule; c * granule < hi; c++) if (!__atomic_exchange_n(&r->referenced[c], 1, __ATOMIC_RELAXED)) {
            uint64_t n = atomic_fetch_add_explicit(&g_refchunks, 1, memory_order_relaxed) + 1;
            if (!(n & (n - 1))) LOG("vulkan: guest memory explicitly GPU-referenced: %llu of %llu backed chunks", (unsigned long long)n, (unsigned long long)g_chunks);
        }
        *buf = r->buf; *off = lo; *gpu = r->gpu + lo; return 1;
    }
    return 0;
}
uint64_t vkm_generation(void) { return atomic_load_explicit(&g_generation, memory_order_acquire); }

int vkm_segments(uint64_t *lo, uint64_t *hi, uint64_t *gpu, int max) {
    if (!lo || !hi || !gpu || max <= 0) return 0;
    pthread_mutex_lock(&g_mu);
    int n = 0, more = 0;
    uint64_t granule = vkm_granule();
    for (int i = 0; i < g_nreg; i++) {
        const vkm_region *r = &g_reg[i];
        if (!r->buf) continue;
        uint64_t nc = (r->size + granule - 1) / granule;
        for (uint64_t c = 0; c < nc;) {
            while (c < nc && !r->chunk[c]) c++;
            if (c == nc) break;
            uint64_t c0 = c; while (c < nc && r->chunk[c]) c++;
            if (n == max) { more = 1; continue; }
            uint64_t off = c0 * granule, end = c * granule;
            if (end > r->size) end = r->size;
            lo[n] = (uint64_t)(uintptr_t)r->base + off;
            hi[n] = (uint64_t)(uintptr_t)r->base + end;
            gpu[n] = (uint64_t)r->gpu + off;
            n++;
        }
    }
    // Regions are registered by the host, not necessarily in address order;
    // the shader's raw-global lookup binary-searches this table.
    for (int i = 1; i < n; i++) {
        uint64_t l = lo[i], h = hi[i], g = gpu[i]; int j = i;
        while (j && lo[j - 1] > l) { lo[j] = lo[j - 1]; hi[j] = hi[j - 1]; gpu[j] = gpu[j - 1]; j--; }
        lo[j] = l; hi[j] = h; gpu[j] = g;
    }
    pthread_mutex_unlock(&g_mu);
    if (more) { static _Atomic int warned; if (!atomic_exchange(&warned, 1)) LOG("[RED] vulkan: raw-global map has more than %d backed runs; later runs are not visible", max); }
    return n;
}

int vkm_gmem_map(void *dst, size_t size, uint64_t *generation) {
    const size_t need = (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    if (!dst || size < need) return -1;
    static uint64_t lo[DMSH_GMEM_MAP_ENTRIES], hi[DMSH_GMEM_MAP_ENTRIES], gpu[DMSH_GMEM_MAP_ENTRIES];
    uint64_t gen; int n;
    do { gen = vkm_generation(); n = vkm_segments(lo, hi, gpu, DMSH_GMEM_MAP_ENTRIES - 1); }
    while (gen != vkm_generation());
    uint8_t *d = dst;
    memset(d, 0xff, need); memset(d, 0, DMSH_GMEM_MAP_ENTRY);
    for (int i = 0; i < n; i++) {
        uint64_t e[4] = {lo[i], hi[i], gpu[i] - lo[i], 0};
        memcpy(d + (size_t)(i + 1) * DMSH_GMEM_MAP_ENTRY, e, sizeof e);
    }
    if (generation) *generation = gen;
    return n;
}
void vkm_stats(uint64_t *chunks, uint64_t *bytes) { pthread_mutex_lock(&g_mu); *chunks = g_chunks; *bytes = g_bytes; pthread_mutex_unlock(&g_mu); }
