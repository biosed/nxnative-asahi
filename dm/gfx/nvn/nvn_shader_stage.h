// NVN2 stage identity kept beside program keys.  Task and mesh SPHs can look
// like legacy graphics stages, so control +0x56 is authoritative for 5/6/7;
// the remaining graphics stages use SPH word 0 at the code-area start.
#pragma once
#include <stdint.h>
#include <string.h>

static inline int nvn2_shader_stage(const void *control, const void *code) {
    const uint8_t *c = control; uint32_t magic = 0, ver = 0;
    if (!c) return -1;
    memcpy(&magic, c, 4); memcpy(&ver, c + 4, 4);
    if (magic != 0x98761234u || ver != 0xc9) return -1;
    if (c[0x56] >= 5 && c[0x56] <= 7) return c[0x56];
    if (!code) return -1;
    uint32_t w; memcpy(&w, code, 4); unsigned sph = w >> 10 & 0xf;
    if (!sph && (w & 0x1f) == 1) return 0;
    static const signed char stage[6] = {-1, 0, 3, 4, 2, 1};
    return sph < 6 ? stage[sph] : -1;
}

