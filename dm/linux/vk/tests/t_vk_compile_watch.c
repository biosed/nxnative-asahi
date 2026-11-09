// The in-flight compile table behind the compile watcher and the heartbeat's "compiling" suffix (vk/vk_compile_watch.h)
#include <stdio.h>
#include "vk_compile_watch.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

int main(void) {
    static vkx_cw_table t;   // zero-initialized, as the host's table is
    char line[160];
    const uint64_t s = 1000000000ull;

    check(vkx_cw_text(&t, 10 * s, 2 * s, line, sizeof line) == 0 && line[0] == 0, "an empty table has no text");

    int a = vkx_cw_begin(&t, "d15370a382630d4e", 5, 0, 7424476, 10 * s);
    check(a >= 0, "a compile takes a slot");
    check(vkx_cw_text(&t, 11 * s, 2 * s, line, sizeof line) == 0 && line[0] == 0, "a compile younger than the threshold is not reported");
    check(vkx_cw_text(&t, 41 * s, 2 * s, line, sizeof line) == 1, "a compile older than the threshold is reported");
    check(!strcmp(line, "1 compiling, oldest d15370a382630d4e stage 5 gl 0 7.42 MB SPIR-V for 31.0 s"), "the line names key, stage, form, size and age");
    if (strcmp(line, "1 compiling, oldest d15370a382630d4e stage 5 gl 0 7.42 MB SPIR-V for 31.0 s")) printf("  got: %s\n", line);

    int b = vkx_cw_begin(&t, "e2eebd7ba18fe35f", 5, 1, 1000000, 30 * s);
    check(b >= 0 && b != a, "a second compile takes another slot");
    check(vkx_cw_text(&t, 41 * s, 2 * s, line, sizeof line) == 2 && !strncmp(line, "2 compiling, oldest d15370a382630d4e", 36), "the count covers both and the oldest is named");
    vkx_cw_end(&t, a);
    check(vkx_cw_text(&t, 41 * s, 2 * s, line, sizeof line) == 1 && !strncmp(line, "1 compiling, oldest e2eebd7ba18fe35f stage 5 gl 1", 49), "an ended compile leaves; the other becomes the oldest");
    vkx_cw_end(&t, b);
    check(vkx_cw_text(&t, 41 * s, 2 * s, line, sizeof line) == 0, "all ended: no text");

    int slots[VKX_CW_SLOTS];
    for (int i = 0; i < VKX_CW_SLOTS; i++) slots[i] = vkx_cw_begin(&t, "0123456789abcdef", 0, 0, 1, (uint64_t)i * s);
    int distinct = 1; for (int i = 0; i < VKX_CW_SLOTS; i++) for (int j = 0; j < i; j++) if (slots[i] < 0 || slots[i] == slots[j]) distinct = 0;
    check(distinct, "every slot can be taken once");
    check(vkx_cw_begin(&t, "ffffffffffffffff", 0, 0, 1, 0) == -1 && atomic_load(&t.dropped) == 1, "a full table refuses and counts the drop");
    check(vkx_cw_text(&t, 100 * s, 0, line, sizeof line) == VKX_CW_SLOTS, "a full table reports every slot");
    vkx_cw_end(&t, -1);   // the unwatched compile's end is harmless
    for (int i = 0; i < VKX_CW_SLOTS; i++) vkx_cw_end(&t, slots[i]);
    check(vkx_cw_begin(&t, "k", 1, 0, 2, 0) >= 0, "freed slots are reused");
    check(vkx_cw_text(&t, 5 * s, 0, line, 8) == 1 && strlen(line) == 7, "a short buffer is cut, not overrun");

    check(vkx_cw_text(&t, 0, 0, line, sizeof line) == 1, "a clock that has not moved reports an age of zero, not a wrap");
    printf("t_vk_compile_watch: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
