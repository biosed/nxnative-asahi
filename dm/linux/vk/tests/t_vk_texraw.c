#include <stdio.h>
#include <string.h>
#include "vk_texraw.h"

static int bad;
static void check(int condition, const char *what) {
    if (!condition) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    vkt_texraw r = VKT_TEXRAW_INIT;
    char path[VKT_TEXRAW_PATH_MAX];
    check(!vkt_texraw_take(&r, path, sizeof path), "an idle request is empty");

    vkt_texraw_request(&r, 41, "/tmp/first.png");
    check(vkt_texraw_take(&r, path, sizeof path) == 41 &&
          !strcmp(path, "/tmp/first.png"), "a request carries its serial and path");
    check(!vkt_texraw_take(&r, path, sizeof path), "a request is consumed once");

    vkt_texraw_request(&r, 42, "/tmp/old.png");
    vkt_texraw_request(&r, 43, "/tmp/new.png");
    check(vkt_texraw_take(&r, path, sizeof path) == 43 &&
          !strcmp(path, "/tmp/new.png"), "a newer request replaces a pending one");

    vkt_texraw_request(&r, 0, "/tmp/zero.png");
    vkt_texraw_request(&r, 44, "");
    check(!vkt_texraw_take(&r, path, sizeof path), "invalid requests are ignored");

    char long_path[VKT_TEXRAW_PATH_MAX + 80];
    memset(long_path, 'x', sizeof long_path);
    long_path[sizeof long_path - 1] = 0;
    vkt_texraw_request(&r, 45, long_path);
    check(vkt_texraw_take(&r, path, sizeof path) == 45 &&
          strlen(path) == VKT_TEXRAW_PATH_MAX - 1,
          "an overlong path is bounded and terminated");

    char short_path[5];
    vkt_texraw_request(&r, 46, "/tmp/short-copy.png");
    check(vkt_texraw_take(&r, short_path, sizeof short_path) == 46 &&
          !strcmp(short_path, "/tmp"), "the consumer's path buffer is bounded");

    printf("t_vk_texraw: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
