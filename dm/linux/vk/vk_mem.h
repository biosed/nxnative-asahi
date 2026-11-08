// Linux build: guest memory the GPU sees (vk_mem.c). The heap and the alias region (hle_svc.c) are backed by Vulkan memory in chunks mapped in place at their own
// addresses, each region one sparse buffer the chunks are bound into at the same offsets: the game and the GPU use the same bytes, as on the Switch
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
// Host/backend seam: hle_svc.c names each reserved GPU-addressable guest
// region, then commits ranges before making their CPU pages accessible.
void dm_gpu_region(void *base, size_t size);
void dm_gpu_commit(void *addr, size_t size);
// Sparse placed-memory allocation/bind granularity. 32 MiB by default;
// DM_VK_GMEM_GRANULE_MB may select a smaller power-of-two pool-mode granule.
uint64_t vkm_granule(void);
// Record the guest-memory chunks covered by an address NVN gives to the GPU.
// DM_VK_GMEM_ADDR_LOG=1 emits one replayable region/chunk line per new chunk.
void vkm_addressed(const void *addr, uint64_t len);
// DM_VK_GMEM_POOLS=1: SetHeapSize/MapPhysicalMemory leave ordinary guest
// pages anonymous; MemoryPoolInitialize calls this before dirty tracking so
// only chunks covered by an NVN pool become placed Vulkan memory.
int vkm_pool_commit(const void *addr, uint64_t len);
// What the GPU uses for [addr, addr + len): the region's buffer and the offset in it, and its GPU address. 0 when the range is not all in backed chunks of one region
int vkm_lookup(const void *addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceAddress *gpu);
// Contiguous backed runs, sorted by guest address. Raw-global shaders use
// their guest bounds and buffer device address to build dmsh_abi.h's map.
// Returns at most max runs; generation changes whenever that map can change.
int vkm_segments(uint64_t *lo, uint64_t *hi, uint64_t *gpu, int max);
uint64_t vkm_generation(void);
// Encode dmsh_abi.h's raw-global table into dst. Returns the number of backed
// runs (not counting empty entry zero), or -1 when dst is too small. The
// returned generation names the exact snapshot encoded.
int vkm_gmem_map(void *dst, size_t size, uint64_t *generation);
void vkm_stats(uint64_t *chunks, uint64_t *bytes);
