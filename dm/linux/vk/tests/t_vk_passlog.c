#include <stdio.h>
#include "vk_passlog.h"

static int bad;
static void check(int condition, const char *what) {
    if (!condition) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    vkp_passlog p = {0};
    check(!vkp_passlog_next(&p), "idle frames are not traced");

    vkp_passlog_request(&p, 2);
    check(vkp_passlog_next(&p) && vkp_passlog_next(&p) && !vkp_passlog_next(&p),
          "a request traces exactly its bounded frame count");

    vkp_passlog_request(&p, VKP_PASSLOG_MAX_FRAMES + 100);
    int traced = 0;
    while (vkp_passlog_next(&p)) traced++;
    check(traced == VKP_PASSLOG_MAX_FRAMES, "large requests are capped");

    vkp_passlog_request(&p, 3);
    check(vkp_passlog_next(&p), "an active request begins");
    vkp_passlog_request(&p, 1);
    check(vkp_passlog_next(&p) && !vkp_passlog_next(&p),
          "a newer request replaces the active request");

    vkp_passlog_request(&p, 0);
    vkp_passlog_request(&p, -1);
    check(!vkp_passlog_next(&p), "non-positive requests are ignored");

    printf("t_vk_passlog: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
