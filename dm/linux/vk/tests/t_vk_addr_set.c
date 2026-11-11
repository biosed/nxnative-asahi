#include <stdio.h>
#include "vk_addr_set.h"

static int bad;

static void check(int condition, const char *what)
{
    if (!condition) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

int main(void)
{
    vkx_addr_set s = {.max_cap = 1u << 20};
    check(!vkx_addr_set_has(&s, 0x1000), "an empty set holds nothing");

    /* Report addresses 32 bytes apart, more than the old 4096-slot table held: all must stay present. */
    const uint64_t base = 0xfffbd6c60000ull;
    for (uint32_t i = 0; i < 20000; i++) vkx_addr_set_add(&s, base + 32ull * i);
    int all = 1;
    for (uint32_t i = 0; i < 20000; i++) all &= vkx_addr_set_has(&s, base + 32ull * i);
    check(all, "every added address stays present past 4096 entries");
    check(s.n == 20000, "each address is counted once");
    vkx_addr_set_add(&s, base);
    check(s.n == 20000, "adding a present address again changes nothing");
    check(!vkx_addr_set_has(&s, base + 16), "an address never added is absent");
    check(!vkx_addr_set_has(&s, base + 32ull * 20000), "the next address past the added range is absent");
    check(2 * s.n <= s.cap, "the table stays at most half full");

    /* Addresses 4 KiB apart all hit one bucket of the old a >> 4 hash. */
    vkx_addr_set p = {.max_cap = 1u << 20};
    for (uint32_t i = 0; i < 3000; i++) vkx_addr_set_add(&p, 0x80000000ull + 4096ull * i);
    check(vkx_addr_set_has(&p, 0x80000000ull + 4096ull * 2999), "page-strided addresses are all present");

    check(!vkx_addr_set_has(&s, 0), "address 0 is absent until added");
    vkx_addr_set_add(&s, 0);
    check(vkx_addr_set_has(&s, 0), "address 0 can be added (0 marks a free slot)");

    /* Past its cap the set saturates and answers present for everything: callers then take their safe answer. */
    vkx_addr_set c = {.max_cap = 1024};
    int added = 1;
    for (uint32_t i = 1; i <= 512; i++) added &= vkx_addr_set_add(&c, 64ull * i);
    check(added && !c.saturated, "a set fills to half its cap without saturating");
    check(!vkx_addr_set_add(&c, 64ull * 513), "the add past the cap reports saturation");
    check(c.saturated && vkx_addr_set_has(&c, 0xdead0000ull), "a saturated set reports every address as present");

    free(s.slot); free(p.slot); free(c.slot);
    printf("t_vk_addr_set: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
