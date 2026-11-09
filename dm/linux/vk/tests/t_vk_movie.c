#include <stdio.h>
#include "vk_movie.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

int main(void) {
    check(vkx_movie_classify(0, 1, 1, 3) == VKX_MOVIE_NONE, "unnamed shader is not a movie stage");
    check(vkx_movie_classify(1, 2, 1, 0) == VKX_MOVIE_NONE, "colour conversion without storage buffers still runs");
    check(vkx_movie_classify(1, 0, 0, 3) == VKX_MOVIE_STAGE, "network storage-buffer pass is an intermediate stage");
    check(vkx_movie_classify(1, 1, 1, 3) == VKX_MOVIE_LAST, "one-texture one-image pass is the final stage");

    vkx_movie_state s = {0};
    check(!vkx_movie_skip_stage(&s) && !s.stages_left, "first frame keeps intermediate network stages");
    vkx_movie_copy_result(&s, 1);
    check(vkx_movie_skip_stage(&s) && s.stages_left == 1, "successful copy permits the next frame's stage skip");
    vkx_movie_copy_result(&s, 0);
    check(!vkx_movie_skip_stage(&s) && !s.stages_left, "failed copy restores the full network");

    printf("t_vk_movie: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
