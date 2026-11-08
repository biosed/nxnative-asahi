#include <stdio.h>
#include <string.h>
#include "vk_missing_log.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

int main(void) {
    vkx_missing_log_set s = {0};
    check(vkx_missing_log_once(&s, UINT64_C(0x0123456789abcdef)) == 1 && s.count == 1,
          "first key is admitted");
    check(vkx_missing_log_once(&s, UINT64_C(0x0123456789abcdef)) == 0 && s.count == 1,
          "duplicate key is suppressed");
    check(vkx_missing_log_once(&s, 0) == 1 && vkx_missing_log_once(&s, 0) == 0,
          "zero is a valid deduplicated key");
    for (uint64_t k = 1; s.count < VKX_MISSING_LOG_MAX; k++)
        (void)vkx_missing_log_once(&s, k);
    check(s.count == VKX_MISSING_LOG_MAX && vkx_missing_log_once(&s, UINT64_C(0xfedcba9876543210)) == -1,
          "the key set fails closed at its fixed bound");

    char line[40];
    int n = vkx_missing_log_line(line, sizeof line, "0123456789abcdef", 5);
    check(n == 19 && !memcmp(line, "0123456789abcdef\t5\n", 19),
          "sink line contains only the key and program index");
    check(vkx_missing_log_line(line, 8, "0123456789abcdef", 5) == -1,
          "truncated output is rejected");

    printf("t_missing_log: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
