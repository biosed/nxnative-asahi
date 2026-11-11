#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vk_shader_cache.h"

static int bad;

static void check(int condition, const char *what)
{
    if (!condition) { printf("FAIL: %s\n", what); bad++; }
}

int main(void)
{
    static const uint8_t uuid[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    };
    const uint64_t hash = vkx_shader_source_hash("abc", 3);
    check(hash == UINT64_C(0xe71fa2190541574b),
          "source hash has the stable FNV-1a value");

    vkx_shader_cache_header h = {
        .magic = VKX_SHADER_CACHE_MAGIC,
        .binary_version = 36,
        .source_hash = hash,
        .binary_size = 1234,
    };
    memcpy(h.binary_uuid, uuid, sizeof uuid);
    check(vkx_shader_cache_header_compatible(&h, 36, uuid, 1234),
          "matching binary format is compatible");
    check(vkx_shader_cache_header_matches(&h, 36, uuid, hash, 1234),
          "matching fitted SPIR-V accepts the binary");
    check(!vkx_shader_cache_header_matches(&h, 36, uuid, hash ^ 1, 1234),
          "changed fitted SPIR-V rejects the binary");
    check(!vkx_shader_cache_header_compatible(&h, 37, uuid, 1234),
          "changed driver binary version rejects the binary");
    check(!vkx_shader_cache_header_compatible(&h, 36, uuid, 1235),
          "truncated or extended binary is rejected");
    h.magic = UINT32_C(0x31636873);
    check(!vkx_shader_cache_header_compatible(&h, 36, uuid, 1234),
          "legacy header without a source hash is rejected");

    printf("t_vk_shader_cache: %s (%d failures)\n",
           bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
