// NVN2 push constants, without a device (vk/vk_push.h): records as hle_nvn.c push_const lays them out land in the right block at the right offset, partial pushes
// keep the rest of the block, the compute block is separate, an out-of-block record is dropped, and a stage maps to its block as vk_draw.c push_fill expects.
// Build and run: make -C dm/linux GAME=<id> t_push
#include <stdio.h>
#include <stdlib.h>
#include "vk_push.h"
static int bad; static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
static p_push_const *rec(uint8_t *buf, int compute, unsigned off, unsigned n, uint8_t fill) {
    p_push_const *r = (p_push_const *)buf; r->compute = (uint16_t)compute; r->offset = (uint16_t)off; r->size = n; for (unsigned i = 0; i < n; i++) r->data[i] = (uint8_t)(fill + i); return r;
}
int main(void) {
    static vkx_push_st p; static uint8_t buf[sizeof(p_push_const) + DMSH_PUSH_SIZE];
    check(vkx_push_apply(&p, rec(buf, 0, 0x10, 16, 0x40)) == 1, "graphics push applied");
    check(p.block[0][0x10] == 0x40 && p.block[0][0x1f] == 0x4f && p.block[0][0x0f] == 0 && p.block[0][0x20] == 0, "graphics bytes at the offset only");
    check(vkx_push_apply(&p, rec(buf, 1, 0, 8, 0x80)) == 1, "compute push applied");
    check(p.block[1][0] == 0x80 && p.block[1][7] == 0x87 && p.block[0][0] == 0, "compute block separate from graphics");
    check(vkx_push_apply(&p, rec(buf, 0, 0x14, 4, 0xa0)) == 1, "partial overwrite applied");
    check(p.block[0][0x13] == 0x43 && p.block[0][0x14] == 0xa0 && p.block[0][0x17] == 0xa3 && p.block[0][0x18] == 0x48, "partial overwrite keeps the rest");
    check(vkx_push_apply(&p, rec(buf, 0, DMSH_PUSH_SIZE - 4, 8, 0)) == 0, "out-of-block record dropped");
    check(vkx_push_apply(&p, rec(buf, 2, 0, 4, 0)) == 0, "unknown block dropped");
    check(p.applied == 3 && p.dropped == 2, "applied and dropped records are counted separately (applied records are not skips)");
    check(vkx_push_block(&p, 5) == p.block[1] && vkx_push_block(&p, 0) == p.block[0] && vkx_push_block(&p, 1) == p.block[0], "stage to block");
    check(DMSH_PUSH_BANK - 3 == 21, "c[24] is cbuf reg 21 (dmsh_spv.py binding 19)");
    printf("t_push: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
