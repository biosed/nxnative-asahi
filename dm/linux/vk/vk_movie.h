// A movie enhancement network's reflection class and fail-closed skip state.
#pragma once

enum { VKX_MOVIE_NONE, VKX_MOVIE_STAGE, VKX_MOVIE_LAST };

static inline int vkx_movie_classify(int named, int textures, int images,
                                     int storage_buffers) {
    if (!named || !storage_buffers) return VKX_MOVIE_NONE;
    return textures == 1 && images == 1 ? VKX_MOVIE_LAST : VKX_MOVIE_STAGE;
}

typedef struct { int prior_copy_ok; unsigned stages_left; } vkx_movie_state;

static inline int vkx_movie_skip_stage(vkx_movie_state *s) {
    if (!s->prior_copy_ok) return 0;
    s->stages_left++;
    return 1;
}

static inline void vkx_movie_copy_result(vkx_movie_state *s, int ok) {
    s->prior_copy_ok = ok != 0;
    s->stages_left = 0;
}
