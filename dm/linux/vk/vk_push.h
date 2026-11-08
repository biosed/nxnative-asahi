// NVN2 push constants in the replay: the record stream's OP_PUSH_CONST bytes (hle_nvn.c push_const: a graphics block and a compute block of DMSH_PUSH_SIZE bytes,
// bound to the shaders as c[24], DMSH_PUSH_BANK) kept as the stream leaves them. Pure, so vk/tests/t_push.c checks it without a device.
#pragma once
#include <stdint.h>
#include <string.h>
#include "nvn_obj.h"

typedef struct { uint8_t block[2][DMSH_PUSH_SIZE]; uint64_t applied, dropped; } vkx_push_st;   // [0] graphics stages, [1] compute

// One OP_PUSH_CONST record: 1 when its bytes were applied, 0 when the record lies outside the block (dropped, as hle_nvn.c would have)
static inline int vkx_push_apply(vkx_push_st *p, const p_push_const *b) {
    if ((uint32_t)b->offset + b->size > DMSH_PUSH_SIZE || b->compute > 1) { p->dropped++; return 0; }
    memcpy(p->block[b->compute] + b->offset, b->data, b->size); p->applied++; return 1;
}
// The block a stage reads: compute for stage 5, graphics for every other
static inline const uint8_t *vkx_push_block(const vkx_push_st *p, int stage) { return p->block[stage == 5 ? 1 : 0]; }
