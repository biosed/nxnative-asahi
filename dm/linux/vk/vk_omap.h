// Fragment-output-map helpers shared by the Vulkan draw path and its
// device-free regression test.
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint32_t vkx_json_u32_or(const char *txt, const char *name,
                                       uint32_t fallback)
{
    const char *p = txt ? strstr(txt, name) : NULL;
    if (!p || !(p = strchr(p + strlen(name), ':'))) return fallback;
    do p++; while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n');

    // System.Text.Json serializes the translator's 32-bit output map as a
    // signed integer.  Accept both that representation (-1 is 0xffffffff)
    // and an explicitly unsigned value.
    char *end = NULL;
    long long v = strtoll(p, &end, 0);
    if (end == p || v < -2147483648ll || v > (long long)UINT32_MAX)
        return fallback;
    return (uint32_t)v;
}

static inline uint32_t vkx_fragment_write_mask(uint32_t nvn_mask,
                                                uint32_t frag_out_map,
                                                unsigned target)
{
    if (target >= 8) return 0;
    return ((nvn_mask & frag_out_map) >> (4 * target)) & 0xfu;
}
