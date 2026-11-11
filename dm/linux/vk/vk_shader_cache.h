#ifndef DM_LINUX_VK_SHADER_CACHE_H
#define DM_LINUX_VK_SHADER_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define VKX_SHADER_CACHE_MAGIC UINT32_C(0x32636873) /* "shc2" */

typedef struct {
    uint32_t magic;
    uint32_t binary_version;
    uint8_t binary_uuid[16];
    uint64_t source_hash;
    uint64_t binary_size;
} vkx_shader_cache_header;

/* The cache is local and non-security-sensitive.  FNV-1a gives a stable,
 * inexpensive identity for the exact fitted SPIR-V bytes sent to the driver. */
static inline uint64_t vkx_shader_source_hash(const void *data, size_t size)
{
    const uint8_t *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static inline int vkx_shader_cache_header_compatible(
    const vkx_shader_cache_header *h, uint32_t binary_version,
    const uint8_t binary_uuid[16], uint64_t stored_binary_size)
{
    return h && h->magic == VKX_SHADER_CACHE_MAGIC &&
           h->binary_version == binary_version &&
           !memcmp(h->binary_uuid, binary_uuid, 16) &&
           h->binary_size == stored_binary_size;
}

static inline int vkx_shader_cache_header_matches(
    const vkx_shader_cache_header *h, uint32_t binary_version,
    const uint8_t binary_uuid[16], uint64_t source_hash,
    uint64_t stored_binary_size)
{
    return vkx_shader_cache_header_compatible(
               h, binary_version, binary_uuid, stored_binary_size) &&
           h->source_hash == source_hash;
}

#endif
