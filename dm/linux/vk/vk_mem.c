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
#include "vk_gmem_map.h"
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
    uint8_t *backed_by;                                  // VKM_BACK_*: the path that first placed each chunk
    uint8_t *raw_visible;                                // chunks exported in a raw-global lookup table
    VkDeviceMemory hole;                                 // DM_VK_GMEM_FLAT: one granule bound over every chunk not backed yet
    int flat;                                            // every byte of the region is bound: exported as one raw-global entry
} vkm_region;
static vkm_region g_reg[VKM_MAXREG]; static int g_nreg;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_chunks, g_bytes; static _Atomic uint64_t g_refchunks;
static uint64_t g_back_open_chunks, g_back_open_pages;
static _Atomic uint64_t g_generation = 1;
enum { VKM_BACK_CPU = 1, VKM_BACK_PROFILE = 2, VKM_BACK_POOL = 3 };
enum { VKM_PROFILE_CHUNKS = 4096 };
static uint8_t g_profile[VKM_MAXREG][VKM_PROFILE_CHUNKS];
static uint8_t g_profile_miss_reported[VKM_MAXREG][VKM_PROFILE_CHUNKS];
static unsigned g_profile_miss_reports;
static int g_profile_loaded, g_profile_on;
static int pool_mode(void);

// A learned profile is a set of 32-MiB region/chunk pairs, one per line. It is
// consumed by the normal pre-open dm_gpu_commit path: selected chunks are
// placed before SetHeapSize/MapPhysicalMemory opens their first CPU page. In
// pool mode it is audit-only: NVN-addressed ranges are checked against it, but
// only the address exposure may place memory.
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
    if (g_profile_on && pool_mode())
        LOG("vulkan: guest-memory profile %s lists %u 32-MiB chunks (audit only; NVN addresses drive placement)", path, n);
    else if (g_profile_on)
        LOG("vulkan: guest-memory profile %s selects %u pre-open 32-MiB chunks", path, n);
    else if (pool_mode())
        LOG("[RED] vulkan: guest-memory profile %s is empty; pool coverage audit is disabled", path);
    else
        LOG("[RED] vulkan: guest-memory profile %s is empty; using full backing", path);
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

// A guest chunk's closed pages must read zero when the guest opens them later (SetHeapSize opens the whole heap and the title's allocator trusts it).
// The first bring-up wrote zero into every closed page of every 32-MiB chunk, which committed the complete 6.2-GiB heap and every mapped alias chunk
// whether or not the title ever touched them: Ocarina of Time jumped from 7.9 to 12.3 GiB in five seconds at its first area load and stalled in reclaim
// under a 12-GiB guard. Device memory is not zero by contract (a driver may hand back a recycled allocation: with the writes simply left out the title
// aborted at startup on a stale allocator word, Result 0xd401), so the pages are left untouched only when the memory was allocated with
// VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT (VK_EXT_zero_initialize_device_memory), which guarantees zero and lets the driver keep it uncommitted.
// DM_VK_GMEM_DEMAND_ZERO=0 writes the zeroes as before; open pages are copied byte for byte either way.
static int demand_zero_mode(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DM_VK_GMEM_DEMAND_ZERO");
        v = g_vkd.zero_init && !(e && !strcmp(e, "0"));
        LOG("vulkan: closed guest pages %s (VK_EXT_zero_initialize_device_memory %s; DM_VK_GMEM_DEMAND_ZERO=0 writes them)",
            v ? "stay demand-zero: the chunk memory is allocated zero-initialized" : "are written with zero", g_vkd.zero_init ? "present" : "absent");
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

static uint64_t chunk_size(const vkm_region *r, uint64_t c) {
    uint64_t off = c * vkm_granule();
    return r->size - off < vkm_granule() ? r->size - off : vkm_granule();
}

static void ref_report(void) {
    uint64_t granule = vkm_granule();
    uint64_t by[4] = {0}, by_bytes[4] = {0};
    uint64_t addressed = 0, addressed_bytes = 0, referenced = 0, referenced_bytes = 0;
    uint64_t raw = 0, raw_bytes = 0, no_explicit = 0, no_explicit_bytes = 0;
    uint64_t neither = 0, neither_bytes = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nreg; i++) {
        const vkm_region *r = &g_reg[i];
        uint64_t nc = (r->size + granule - 1) / granule;
        uint64_t rb = 0, rr = 0, ra = 0, rw = 0, rn = 0;
        uint64_t rbb = 0, rrb = 0, rab = 0, rwb = 0, rnb = 0;
        for (uint64_t c = 0; c < nc; c++) {
            if (!r->chunk[c]) continue;
            uint64_t n = chunk_size(r, c); unsigned cause = r->backed_by[c];
            if (cause < 4) { by[cause]++; by_bytes[cause] += n; }
            rb++; rbb += n;
            if (r->referenced[c]) { referenced++; referenced_bytes += n; rr++; rrb += n; }
            else { no_explicit++; no_explicit_bytes += n; }
            if (r->addressed[c]) { addressed++; addressed_bytes += n; ra++; rab += n; }
            if (r->raw_visible[c]) { raw++; raw_bytes += n; rw++; rwb += n; }
            if (!r->referenced[c] && !r->addressed[c]) { neither++; neither_bytes += n; rn++; rnb += n; }
        }
        if (rb) LOG("vulkan: guest memory provenance region %d: backed %llu chunks/%llu MiB; explicit %llu/%llu MiB; NVN-addressed %llu/%llu MiB; raw-table-visible %llu/%llu MiB; neither explicit nor addressed %llu/%llu MiB",
                    i, (unsigned long long)rb, (unsigned long long)(rbb >> 20),
                    (unsigned long long)rr, (unsigned long long)(rrb >> 20),
                    (unsigned long long)ra, (unsigned long long)(rab >> 20),
                    (unsigned long long)rw, (unsigned long long)(rwb >> 20),
                    (unsigned long long)rn, (unsigned long long)(rnb >> 20));
    }
    LOG("vulkan: guest memory provenance total: %llu chunks/%llu MiB backed (CPU full %llu/%llu MiB, CPU profile %llu/%llu MiB, live pool %llu/%llu MiB); explicit %llu/%llu MiB; NVN-addressed %llu/%llu MiB; raw-table-visible %llu/%llu MiB; not explicit %llu/%llu MiB; neither explicit nor addressed %llu/%llu MiB; %llu chunks had %llu CPU-open pages when placed",
        (unsigned long long)g_chunks, (unsigned long long)(g_bytes >> 20),
        (unsigned long long)by[VKM_BACK_CPU], (unsigned long long)(by_bytes[VKM_BACK_CPU] >> 20),
        (unsigned long long)by[VKM_BACK_PROFILE], (unsigned long long)(by_bytes[VKM_BACK_PROFILE] >> 20),
        (unsigned long long)by[VKM_BACK_POOL], (unsigned long long)(by_bytes[VKM_BACK_POOL] >> 20),
        (unsigned long long)referenced, (unsigned long long)(referenced_bytes >> 20),
        (unsigned long long)addressed, (unsigned long long)(addressed_bytes >> 20),
        (unsigned long long)raw, (unsigned long long)(raw_bytes >> 20),
        (unsigned long long)no_explicit, (unsigned long long)(no_explicit_bytes >> 20),
        (unsigned long long)neither, (unsigned long long)(neither_bytes >> 20),
        (unsigned long long)g_back_open_chunks, (unsigned long long)g_back_open_pages);
    pthread_mutex_unlock(&g_mu);
}

void dm_gpu_region(void *base, size_t size) {
    static int report_registered;
    static int pool_reported;
    if (pool_mode()) {
        swap_hooks();
        if (!pool_reported++) LOG("[RED] vulkan: experimental live guest-memory placement enabled (DM_VK_GMEM_POOLS=1); not a gameplay baseline");
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
        r->backed_by = calloc(n, 1);
        r->raw_visible = calloc(n, 1);
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
        }
        // The learned profile is always indexed in 32-MiB units, independent
        // of the experimental pool placement granule. Report each uncovered
        // profile chunk once; observing it must never place memory.
        if (profile_mode()) {
            for (uint64_t c = lo / VKM_CHUNK_DEFAULT;
                 c * VKM_CHUNK_DEFAULT < hi; c++) {
                int missing = c >= VKM_PROFILE_CHUNKS || !g_profile[i][c];
                int fresh = c >= VKM_PROFILE_CHUNKS || !g_profile_miss_reported[i][c];
                if (missing && fresh) {
                    if (c < VKM_PROFILE_CHUNKS)
                        g_profile_miss_reported[i][c] = 1;
                    if (g_profile_miss_reports++ < 32)
                        LOG("[RED] vulkan: guest-memory profile miss: region %d chunk %llu (offset %#llx)", i,
                            (unsigned long long)c,
                            (unsigned long long)(c * VKM_CHUNK_DEFAULT));
                }
            }
        }
        break;
    }
    pthread_mutex_unlock(&g_mu);
}

// DM_VK_GMEM_FLAT=1 (experimental, default off): every region is one sparse buffer at one contiguous device address, so a guest address inside it is
// the region's buffer address plus its offset. The raw-global table nevertheless lists each backed run, and the translated shaders binary-search it
// (13 steps and range checks per access, inlined hundreds of times in the mesh producers). Here the chunks that are not backed yet are all bound to one
// shared zero-initialised granule (this G13 reports residencyNonResidentStrict = false, so an unbound range must not be read), the table holds one entry
// per region, and entry 0 carries DMSH_GMEM_FLAT_TAG so a translator fast path can use those few entries with no search. A backed chunk replaces its hole
// binding as before. A stray GPU access to a hole reads zero or another hole's stray bytes instead of being refused; any write there is lost to the CPU.
static int flat_mode(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DM_VK_GMEM_FLAT"); v = e && *e && strcmp(e, "0");
        if (v) LOG("[RED] vulkan: experimental flat raw-global map (DM_VK_GMEM_FLAT=1): unbacked guest chunks share one zero granule; not a gameplay baseline");
    }
    return v;
}
static void flat_cover_holes(vkm_region *r) {
    uint64_t granule = vkm_granule(), nc = (r->size + granule - 1) / granule;
    VkMemoryAllocateFlagsInfo zf = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT};
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = g_vkd.zero_init ? &zf : NULL, .allocationSize = granule, .memoryTypeIndex = r->type};
    VkResult vr = vkAllocateMemory(g_vkd.dev, &ma, NULL, &r->hole);
    if (vr) { LOG("[RED] vulkan: flat raw-global map: no hole granule for region %p (%d); region keeps the run table", (void *)r->base, (int)vr); r->hole = VK_NULL_HANDLE; return; }
    if (!g_vkd.zero_init) {   // not zero by contract: write the zeroes once
        void *p = NULL;
        if (vkMapMemory(g_vkd.dev, r->hole, 0, VK_WHOLE_SIZE, 0, &p) || !p) { vkFreeMemory(g_vkd.dev, r->hole, NULL); r->hole = VK_NULL_HANDLE; LOG("[RED] vulkan: flat raw-global map: hole granule not mappable"); return; }
        memset(p, 0, granule); vkUnmapMemory(g_vkd.dev, r->hole);
    }
    VkSparseMemoryBind *sb = calloc(nc, sizeof *sb); uint32_t n = 0;
    if (!sb) die("vulkan: flat raw-global map allocation failed");
    for (uint64_t c = 0; c < nc; c++) if (!r->chunk[c]) sb[n++] = (VkSparseMemoryBind){c * granule, chunk_size(r, c), r->hole, 0, 0};
    VkSparseBufferMemoryBindInfo sbi = {r->buf, n, sb};
    VkBindSparseInfo bsi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .bufferBindCount = n ? 1 : 0, .pBufferBinds = &sbi};
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; vkCreateFence(g_vkd.dev, &fci, NULL, &f);
    pthread_mutex_lock(&g_vkd_qlock); vr = vkQueueBindSparse(g_vkd.q, 1, &bsi, f); pthread_mutex_unlock(&g_vkd_qlock);
    if (!vr) vr = vkWaitForFences(g_vkd.dev, 1, &f, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g_vkd.dev, f, NULL); free(sb);
    if (vr) die("vulkan: flat raw-global map: binding the hole granule into region %p failed (%d)", (void *)r->base, (int)vr);
    r->flat = 1;
    atomic_fetch_add_explicit(&g_generation, 1, memory_order_release);
    LOG("vulkan: flat raw-global map: region %p+%#llx is one entry (%u chunks bound to the zero granule)", (void *)r->base, (unsigned long long)r->size, n);
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
    if (flat_mode()) flat_cover_holes(r);
    return 1;
}

static int back_chunk(vkm_region *r, uint64_t ci, unsigned cause) {
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
    VkMemoryAllocateFlagsInfo zf = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT};
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = demand_zero_mode() ? &zf : NULL, .allocationSize = len, .memoryTypeIndex = r->type};
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
    { static int zc = -1; if (zc < 0) { const char *e = getenv("DM_VK_GMEM_ZERO_CHECK"); zc = e && atoi(e) != 0; }   // diagnostic: do the untouched closed pages read zero? (reading commits them)
      if (zc && demand_zero_mode()) { size_t bad = 0; for (size_t p = 0; p < npg; p++) if (!open[p]) { const uint64_t *w = (const uint64_t *)(at + p * pg); for (size_t i = 0; i < pg / 8; i++) if (w[i]) { bad++; break; } }
          static _Atomic uint64_t tot, chunks; uint64_t t = atomic_fetch_add(&tot, bad) + bad, c = atomic_fetch_add(&chunks, 1) + 1;
          if (bad || !(c & (c - 1))) LOG("%svulkan: zero check: chunk %p has %zu non-zero closed pages of %zu (%llu in %llu chunks so far)", bad ? "[RED] " : "", (void *)at, bad, npg - nopen, (unsigned long long)t, (unsigned long long)c); } }
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
    r->chunk[ci] = m; r->backed_by[ci] = (uint8_t)cause; g_chunks++; g_bytes += len;
    if (nopen) { g_back_open_chunks++; g_back_open_pages += nopen; }
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
        int profile = profile_mode();
        for (uint64_t c = lo / granule; c * granule < hi; c++) {
            if (profile && (i >= VKM_MAXREG || c >= VKM_PROFILE_CHUNKS || !g_profile[i][c])) continue;
            if (!r->chunk[c] && !back_chunk(r, c, profile ? VKM_BACK_PROFILE : VKM_BACK_CPU)) break;
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
            if (!r->chunk[c] && !back_chunk(r, c, VKM_BACK_POOL)) { ok = 0; break; }
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
        vkm_region *r = &g_reg[i];
        if (!r->buf) continue;
        uint64_t nc = (r->size + granule - 1) / granule;
        if (r->flat) {   // every chunk is bound (a backed one or the hole granule): the whole region is one entry
            if (n == max) { more = 1; continue; }
            memset(r->raw_visible, 1, nc);
            lo[n] = (uint64_t)(uintptr_t)r->base; hi[n] = (uint64_t)(uintptr_t)r->base + r->size; gpu[n] = (uint64_t)r->gpu; n++;
            continue;
        }
        for (uint64_t c = 0; c < nc;) {
            while (c < nc && !r->chunk[c]) c++;
            if (c == nc) break;
            uint64_t c0 = c; while (c < nc && r->chunk[c]) c++;
            if (n == max) { more = 1; continue; }
            for (uint64_t k = c0; k < c; k++) r->raw_visible[k] = 1;
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
    int out = vkx_gmem_map_pack(dst, size, lo, hi, gpu, n);
    if (out < 0) return -1;
    // Flat only when every exported entry is a whole, fully bound region (a region that could not get its hole granule keeps its runs)
    int flat = flat_mode() && out > 0 && out <= VKX_GMEM_FLAT_MAX;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nreg && flat; i++) if (g_reg[i].buf && !g_reg[i].flat) flat = 0;
    pthread_mutex_unlock(&g_mu);
    if (flat) vkx_gmem_map_mark_flat(dst, size, out);
    if (generation) *generation = gen;
    return out;
}
void vkm_stats(uint64_t *chunks, uint64_t *bytes) { pthread_mutex_lock(&g_mu); *chunks = g_chunks; *bytes = g_bytes; pthread_mutex_unlock(&g_mu); }
