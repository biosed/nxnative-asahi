// Game-free contract test for vk_mem.c's hle_svc seam. It reserves a guest
// region, opens and writes one CPU page, registers/commits it through the same
// two callbacks as the runtime, and verifies both byte preservation and a GPU
// copy from the sparse placed buffer. It also proves that dm_gpu_commit leaves
// CPU memory independent in pool mode: vkm_lookup rejects each chunk until the
// explicit NVN pool commit places it.
#include "dm.h"
#include "dmsh_abi.h"
#include "vk_dev.h"
#include "vk_mem.h"

#include <sys/mman.h>

#define COPY_SIZE 4096
#define VK_CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) \
    die("vk_mem selftest: %s failed (%d)", #call, (int)r_); } while (0)

// The game host supplies these hook registries.  This single-threaded physical
// test needs no fault arbitration, but links the same pool-placement path.
void dm_fault_hook_register(int (*fn)(void *, int, void *)) { (void)fn; }
void dm_guest_write_prep_register(void (*fn)(void *, size_t)) { (void)fn; }
void nvn_dirty_hold(const void *addr, size_t len) { (void)addr; (void)len; }
void nvn_dirty_unhold(void) {}

static void *reserve_aligned(size_t size, size_t align) {
    size_t total = size + align;
    uint8_t *whole = mmap(NULL, total, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (whole == MAP_FAILED) return NULL;
    uintptr_t a = ((uintptr_t)whole + align - 1) & ~(uintptr_t)(align - 1);
    size_t pre = a - (uintptr_t)whole;
    size_t post = total - pre - size;
    if (pre) munmap(whole, pre);
    if (post) munmap((void *)(a + size), post);
    return (void *)a;
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_vkd.mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) &&
            (g_vkd.mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

int main(void) {
    if (setenv("DM_VK_GMEM_POOLS", "1", 1) ||
        setenv("DM_VK_GMEM_DEMAND_ZERO", "0", 1))
        die("vk_mem selftest: could not select the pool/explicit-zero contract");
    if (!vkd_init(0) || !g_vkd.placed)
        die("vk_mem selftest: placed-memory Vulkan device is unavailable");

    const size_t test_chunk = (size_t)vkm_granule();
    const size_t test_size = 2 * test_chunk;
    uint8_t *guest = reserve_aligned(test_size, test_chunk);
    if (!guest) die("vk_mem selftest: could not reserve aligned guest region");
    size_t pg = (size_t)getpagesize();
    if (mprotect(guest, pg, PROT_READ | PROT_WRITE))
        die("vk_mem selftest: could not open the source page (errno=%d)", errno);
    for (uint32_t i = 0; i < COPY_SIZE / 4; i++)
        ((uint32_t *)guest)[i] = 0x6d000000u ^ i;

    dm_gpu_region(guest, test_size);
    dm_gpu_commit(guest, COPY_SIZE);
    uint64_t chunks = 0, bytes = 0;
    vkm_stats(&chunks, &bytes);
    if (chunks || bytes)
        die("vk_mem selftest: CPU commit placed %llu chunks/%llu bytes before NVN exposed the pool",
            (unsigned long long)chunks, (unsigned long long)bytes);
    if (!vkm_pool_commit(guest, COPY_SIZE))
        die("vk_mem selftest: first pool commit failed");
    vkm_stats(&chunks, &bytes);
    if (chunks != 1 || bytes != test_chunk)
        die("vk_mem selftest: first pool commit placed %llu chunks/%llu bytes, expected 1/%llu",
            (unsigned long long)chunks, (unsigned long long)bytes,
            (unsigned long long)test_chunk);

    VkBuffer src = VK_NULL_HANDLE; VkDeviceSize src_off = 0;
    VkDeviceAddress src_gpu = 0;
    if (!vkm_lookup(guest, COPY_SIZE, &src, &src_off, &src_gpu) || !src_gpu)
        die("vk_mem selftest: committed first chunk is not visible");
    uint64_t slo[4], shi[4], sgpu[4];
    if (vkm_segments(slo, shi, sgpu, 4) != 1 || slo[0] != (uint64_t)(uintptr_t)guest ||
        shi[0] != (uint64_t)(uintptr_t)(guest + test_chunk) || sgpu[0] != src_gpu)
        die("vk_mem selftest: first raw-global segment is wrong");
    const size_t map_size = (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    uint8_t *map = malloc(map_size); uint64_t map_gen0 = 0;
    if (!map || vkm_gmem_map(map, map_size, &map_gen0) != 1)
        die("vk_mem selftest: could not encode the first raw-global map");
    uint64_t e0[4], e1[4], e2[4];
    memcpy(e0, map, sizeof e0);
    memcpy(e1, map + DMSH_GMEM_MAP_ENTRY, sizeof e1);
    memcpy(e2, map + 2 * DMSH_GMEM_MAP_ENTRY, sizeof e2);
    if (e0[0] || e0[1] || e0[2] || e0[3] || e1[0] != slo[0] || e1[1] != shi[0] || e1[2] != sgpu[0] - slo[0] || e1[3])
        die("vk_mem selftest: raw-global entry bytes are wrong");
    // Unused entries use UINT64_MAX only for the binary-search lower bound;
    // their end and delta stay zero, and word 6 may hold a hash hint.
    if (e2[0] != UINT64_MAX || e2[1] || e2[2])
        die("vk_mem selftest: unused raw-global entry has the wrong sentinel");
    if (vkm_lookup(guest + test_chunk, COPY_SIZE, &src, &src_off, &src_gpu))
        die("vk_mem selftest: uncommitted second chunk is visible");
    for (uint32_t i = 0; i < COPY_SIZE / 4; i++)
        if (((uint32_t *)guest)[i] != (0x6d000000u ^ i))
            die("vk_mem selftest: CPU bytes changed at word %u", i);

    VkBuffer dst = VK_NULL_HANDLE;
    VkDeviceMemory dst_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = COPY_SIZE, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK_CHECK(vkCreateBuffer(g_vkd.dev, &bi, NULL, &dst));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, dst, &mr);
    uint32_t mt = mem_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) die("vk_mem selftest: no coherent host-visible memory");
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size, .memoryTypeIndex = mt};
    VK_CHECK(vkAllocateMemory(g_vkd.dev, &ma, NULL, &dst_mem));
    VK_CHECK(vkBindBufferMemory(g_vkd.dev, dst, dst_mem, 0));
    void *dst_map = NULL; VK_CHECK(vkMapMemory(g_vkd.dev, dst_mem, 0, COPY_SIZE, 0, &dst_map));
    memset(dst_map, 0, COPY_SIZE);

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = g_vkd.qf};
    VK_CHECK(vkCreateCommandPool(g_vkd.dev, &pi, NULL, &pool));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    VK_CHECK(vkAllocateCommandBuffers(g_vkd.dev, &cai, &cb));
    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    VkBufferCopy copy = {.srcOffset = src_off, .size = COPY_SIZE};
    vkCmdCopyBuffer(cb, src, dst, 1, &copy);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(g_vkd.dev, &fi, NULL, &fence));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cb};
    pthread_mutex_lock(&g_vkd_qlock);
    VkResult qr = vkQueueSubmit(g_vkd.q, 1, &si, fence);
    pthread_mutex_unlock(&g_vkd_qlock);
    if (qr != VK_SUCCESS) die("vk_mem selftest: vkQueueSubmit failed (%d)", (int)qr);
    VK_CHECK(vkWaitForFences(g_vkd.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    if (memcmp(guest, dst_map, COPY_SIZE))
        die("vk_mem selftest: GPU copy did not see preserved guest bytes");

    dm_gpu_commit(guest + test_chunk, COPY_SIZE);
    vkm_stats(&chunks, &bytes);
    if (chunks != 1 || bytes != test_chunk)
        die("vk_mem selftest: second CPU commit changed GPU placement to %llu chunks/%llu bytes",
            (unsigned long long)chunks, (unsigned long long)bytes);
    if (vkm_lookup(guest + test_chunk, COPY_SIZE, &src, &src_off, &src_gpu))
        die("vk_mem selftest: second CPU commit made an unaddressed chunk visible");
    if (!vkm_pool_commit(guest + test_chunk, COPY_SIZE))
        die("vk_mem selftest: second pool commit failed");
    // Read a page the CPU has never opened through the sparse Vulkan buffer
    // before faulting it into the guest mapping. Explicit-zero mode must expose
    // fresh placed memory as zero to both GPU and guest CPU.
    if (!vkm_lookup(guest + test_chunk, COPY_SIZE, &src, &src_off, &src_gpu))
        die("vk_mem selftest: explicitly zeroed page is not GPU-visible");
    memset(dst_map, 0xa5, COPY_SIZE);
    VK_CHECK(vkResetCommandPool(g_vkd.dev, pool, 0));
    VK_CHECK(vkResetFences(g_vkd.dev, 1, &fence));
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    copy.srcOffset = src_off;
    vkCmdCopyBuffer(cb, src, dst, 1, &copy);
    VK_CHECK(vkEndCommandBuffer(cb));
    pthread_mutex_lock(&g_vkd_qlock);
    qr = vkQueueSubmit(g_vkd.q, 1, &si, fence);
    pthread_mutex_unlock(&g_vkd_qlock);
    if (qr != VK_SUCCESS)
        die("vk_mem selftest: explicit-zero copy submit failed (%d)", qr);
    VK_CHECK(vkWaitForFences(g_vkd.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    for (size_t i = 0; i < COPY_SIZE; i++)
        if (((const uint8_t *)dst_map)[i])
            die("vk_mem selftest: GPU saw nonzero fresh memory at byte %zu", i);
    // A closed page was populated only in the temporary Vulkan map. Opening it
    // through the placed guest address must expose the same fresh-memory value.
    if (mprotect(guest + test_chunk, pg, PROT_READ | PROT_WRITE))
        die("vk_mem selftest: could not open the explicitly zeroed page (errno=%d)", errno);
    for (size_t i = 0; i < pg; i++)
        if (guest[test_chunk + i])
            die("vk_mem selftest: fresh guest page is not zero at byte %zu", i);
    if (!vkm_lookup(guest + test_chunk - pg, 2 * pg,
                    &src, &src_off, &src_gpu))
        die("vk_mem selftest: range across two committed chunks is not visible");
    if (vkm_segments(slo, shi, sgpu, 4) != 1 || slo[0] != (uint64_t)(uintptr_t)guest ||
        shi[0] != (uint64_t)(uintptr_t)(guest + test_size))
        die("vk_mem selftest: adjacent raw-global chunks did not coalesce");
    uint64_t map_gen1 = 0;
    if (vkm_gmem_map(map, map_size, &map_gen1) != 1 || map_gen1 == map_gen0)
        die("vk_mem selftest: raw-global generation did not rebuild");
    memcpy(e1, map + DMSH_GMEM_MAP_ENTRY, sizeof e1);
    if (e1[0] != slo[0] || e1[1] != shi[0] || e1[2] != sgpu[0] - slo[0])
        die("vk_mem selftest: rebuilt raw-global entry is wrong");
    vkm_stats(&chunks, &bytes);
    if (chunks != 2 || bytes != test_size)
        die("vk_mem selftest: expected 2 chunks/%llu bytes, got %llu/%llu",
            (unsigned long long)test_size, (unsigned long long)chunks,
            (unsigned long long)bytes);

    LOG("vk_mem selftest passed: CPU commits allocated no Vulkan chunks, bytes preserved, GPU copy verified, "
        "unaddressed chunk rejected, raw-global bytes/generation verified, cross-chunk lookup accepted (%llu MiB backed)",
        (unsigned long long)(bytes >> 20));
    free(map);
    return 0;
}
