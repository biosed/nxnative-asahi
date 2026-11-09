// The compute-dispatch skip rule (vk/vk_skipcs.h), without a device: the DM_VK_SKIP_CS list parses (commas, whitespace, case, junk dropped), only exact keys
// match, unset changes nothing, and every selected dispatch is counted. Uses Ocarina of Time's four NDVE keys (Runtime Cycle 20).
// Build and run: make -C dm/linux GAME=<id> t_skipcs
#include <stdio.h>
#include "vk_skipcs.h"
static int bad; static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
int main(void) {
    vkx_skipcs c;
    check(vkx_skipcs_parse(&c, NULL) == 0 && !c.on, "unset: off");
    check(!vkx_skipcs_match(&c, "d15370a382630d4e"), "unset: no dispatch skipped");
    check(vkx_skipcs_parse(&c, "") == 0 && !c.on &&
          !vkx_skipcs_match(&c, "d15370a382630d4e"), "empty launcher default: no dispatch skipped");
    int dropped = vkx_skipcs_parse(&c, "d15370a382630d4e,E6BE82D835FBE4D3\t9442573ee4ace011\n,bad,0a56002cdc7e1fce,0a56002cdc7e1fceXX");
    check(c.on && c.n == 4 && dropped == 2, "list: four keys taken, two junk entries dropped");
    check(!strcmp(c.key[1], "e6be82d835fbe4d3"), "upper-case key lowered");
    check(vkx_skipcs_match(&c, "d15370a382630d4e") && vkx_skipcs_match(&c, "e6be82d835fbe4d3") &&
          vkx_skipcs_match(&c, "9442573ee4ace011") && vkx_skipcs_match(&c, "0a56002cdc7e1fce"), "the four exact NDVE keys skipped");
    check(!vkx_skipcs_match(&c, "5a7d69b067a7f011") && !vkx_skipcs_match(&c, "ba079f68ae288dba") &&
          !vkx_skipcs_match(&c, "d15370a382630d4f"), "nearby and unrelated compute keys kept");
    check(c.skipped == 4, "selected dispatch count");
    check(vkx_skipcs_parse(&c, "d15370a382630d4") == 1 && !c.on && !vkx_skipcs_match(&c, "d15370a382630d4e"), "a key prefix is invalid, not a broad match");
    printf("t_skipcs: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
