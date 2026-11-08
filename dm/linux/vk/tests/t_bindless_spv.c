// Device-free proof of the NVN texture/sampler-pool SPIR-V canonicalisation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spv_reflect.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
static void put(uint32_t *w, size_t *n, const uint32_t *x, size_t nx) { memcpy(w + *n, x, nx * 4); *n += nx; }
static uint32_t decoration(const uint32_t *w, size_t n, uint32_t id, uint32_t dec);
#define EMIT(...) do { const uint32_t x_[] = {__VA_ARGS__}; put(w, &n, x_, sizeof x_ / sizeof *x_); } while (0)

static size_t module(uint32_t *w) {
    size_t n = 0; uint32_t h[] = {0x07230203, 0x00010300, 0, 64, 0}; put(w, &n, h, 5);
    // Four pool variables in the translator's original set/bindings and one
    // ordinary descriptor which must remain untouched.
    for (uint32_t id = 30; id <= 34; id++) { EMIT((4u<<16)|71, id, 34, 1); }
    EMIT((4u<<16)|71, 30, 33, 564); EMIT((4u<<16)|71, 31, 33, 564);
    EMIT((4u<<16)|71, 32, 33, 566); EMIT((4u<<16)|71, 33, 33, 660);
    EMIT((4u<<16)|71, 34, 33, 64);
    EMIT((4u<<16)|21, 1, 32, 0);                    // u32
    EMIT((3u<<16)|22, 2, 32);                       // f32
    EMIT((4u<<16)|43, 1, 3, 1048576);               // texture count
    EMIT((4u<<16)|43, 1, 4, 4096);                  // sampler count
    EMIT((9u<<16)|25, 10, 2, 1, 0, 0, 0, 1, 0);    // image2D
    EMIT((9u<<16)|25, 11, 2, 1, 0, 1, 0, 1, 0);    // image2DArray
    EMIT((9u<<16)|25, 12, 2, 1, 1, 0, 0, 1, 0);    // depth image2D
    EMIT((2u<<16)|26, 13);                          // sampler
    EMIT((4u<<16)|28, 20, 10, 3); EMIT((4u<<16)|28, 21, 11, 3);
    EMIT((4u<<16)|28, 22, 12, 3); EMIT((4u<<16)|28, 23, 13, 4);
    EMIT((4u<<16)|32, 24, 0, 20); EMIT((4u<<16)|32, 25, 0, 21);
    EMIT((4u<<16)|32, 26, 0, 22); EMIT((4u<<16)|32, 27, 0, 23);
    EMIT((4u<<16)|32, 28, 0, 10);
    EMIT((4u<<16)|59, 24, 30, 0); EMIT((4u<<16)|59, 25, 31, 0);
    EMIT((4u<<16)|59, 26, 32, 0); EMIT((4u<<16)|59, 27, 33, 0);
    EMIT((4u<<16)|59, 28, 34, 0);
    return n;
}
#undef EMIT

static size_t feedback_module(uint32_t *w) {
    size_t n = 0; uint32_t h[] = {0x07230203, 0x00010400, 0, 64, 0}; put(w, &n, h, 5);
#define FEMIT(...) do { const uint32_t x_[] = {__VA_ARGS__}; put(w, &n, x_, sizeof x_ / sizeof *x_); } while (0)
    FEMIT((2u<<16)|17, 1);                         // Shader capability
    FEMIT((3u<<16)|14, 0, 1);                     // Logical, GLSL450
    FEMIT((5u<<16)|15, 4, 20, 0x6e69616d, 0);    // Fragment main, no interface yet
    FEMIT((4u<<16)|71, 30, 34, 5);               // pool set 5
    FEMIT((4u<<16)|71, 30, 33, 0);               // texture-array binding 0
    FEMIT((2u<<16)|19, 2);                        // void
    FEMIT((4u<<16)|21, 1, 32, 0);                // u32
    FEMIT((3u<<16)|22, 3, 32);                    // f32
    FEMIT((4u<<16)|43, 1, 4, 8);                 // array length
    FEMIT((4u<<16)|43, 1, 5, 0); FEMIT((4u<<16)|43, 1, 6, 1);
    FEMIT((9u<<16)|25, 10, 3, 1, 0, 0, 0, 1, 0);// image2D
    FEMIT((4u<<16)|28, 11, 10, 4);               // image2D[8]
    FEMIT((4u<<16)|32, 12, 0, 11);               // UniformConstant array pointer
    FEMIT((4u<<16)|32, 14, 0, 10);               // UniformConstant image pointer
    FEMIT((4u<<16)|59, 12, 30, 0);               // pool variable
    FEMIT((3u<<16)|33, 13, 2);                    // void function type
    FEMIT((5u<<16)|54, 2, 20, 0, 13);            // main
    FEMIT((2u<<16)|248, 21);                      // label
    FEMIT((5u<<16)|128, 1, 40, 5, 6);            // dynamic-typed index
    FEMIT((5u<<16)|65, 14, 41, 30, 40);          // pool[index]
    FEMIT((1u<<16)|253); FEMIT((1u<<16)|56);
#undef FEMIT
    return n;
}

static uint32_t feedback_var(const uint32_t *w, size_t n) {
    for (size_t i = 5; i < n;) { uint32_t z = w[i] >> 16; if (!z) break;
        if ((w[i] & 0xffff) == 71 && z >= 4 && w[i + 2] == 34 && w[i + 3] == 6) {
            uint32_t id = w[i + 1];
            if (decoration(w, n, id, 33) == 0) return id;
        }
        i += z;
    }
    return 0;
}

static int entry_has(const uint32_t *w, size_t n, uint32_t id) {
    for (size_t i = 5; i < n;) { uint32_t z = w[i] >> 16; if (!z) break;
        if ((w[i] & 0xffff) == 15) for (uint32_t j = 3; j < z; j++) if (w[i + j] == id) return 1;
        i += z;
    }
    return 0;
}

static uint32_t decoration(const uint32_t *w, size_t n, uint32_t id, uint32_t dec) {
    for (size_t i = 5; i < n;) { uint32_t z = w[i] >> 16; if (!z) break;
        if ((w[i] & 0xffff) == 71 && z >= 4 && w[i + 1] == id && w[i + 2] == dec) return w[i + 3]; i += z; }
    return UINT32_MAX;
}
static uint32_t constant(const uint32_t *w, size_t n, uint32_t id) {
    for (size_t i = 5; i < n;) { uint32_t z = w[i] >> 16; if (!z) break;
        if ((w[i] & 0xffff) == 43 && z >= 4 && w[i + 2] == id) return w[i + 3]; i += z; }
    return UINT32_MAX;
}

static int rewrite_file(const char *src, const char *dst) {
    FILE *f = fopen(src, "rb"); if (!f) return 2;
    fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)nb); size_t got = w ? fread(w, 1, (size_t)nb, f) : 0; fclose(f);
    if (!w || got != (size_t)nb || nb % 4) { free(w); return 2; }
    size_t no = 0; unsigned patched = 0; uint32_t *o = spvr_bindless(w, (size_t)nb / 4, 5, 32768, &no, &patched); free(w);
    if (!o) return 3;
    f = fopen(dst, "wb"); int ok = 0;
    if (f) { ok = fwrite(o, 4, no, f) == no; if (fclose(f)) ok = 0; }
    free(o); if (!ok) return 4;
    printf("t_bindless_spv: rewrote %u pool variables in %s -> %s\n", patched, src, dst);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3) return rewrite_file(argv[1], argv[2]);
    uint32_t w[256]; size_t n = module(w), no = 0; unsigned patched = 0;
    uint32_t *o = spvr_bindless(w, n, 5, 32768, &no, &patched);
    check(o && no == n && patched == 4, "all four pool variables recognised");
    check(o && decoration(o, no, 30, 33) == 0 && decoration(o, no, 31, 33) == 1 && decoration(o, no, 32, 33) == 2,
          "image pools canonicalised by 2D/array/depth shape");
    check(o && decoration(o, no, 33, 33) == 3, "sampler pool canonicalised");
    check(o && decoration(o, no, 30, 34) == 5 && decoration(o, no, 31, 34) == 5 && decoration(o, no, 32, 34) == 5 && decoration(o, no, 33, 34) == 5,
          "pool variables moved to bindless set 5");
    check(o && constant(o, no, 3) == 32768 && constant(o, no, 4) == 4096, "texture count bounded and sampler count preserved");
    check(o && decoration(o, no, 34, 33) == 64 && decoration(o, no, 34, 34) == 1, "ordinary descriptor untouched");
    free(o);
    uint32_t fw[256]; size_t fn = feedback_module(fw), fno = 0; patched = 0;
    o = spvr_bindless_feedback(fw, fn, 5, 6, &fno, &patched);
    uint32_t fv = o ? feedback_var(o, fno) : 0;
    check(o && patched == 1 && fv, "one bindless texture access receives feedback instrumentation");
    check(o && entry_has(o, fno, fv), "SPIR-V 1.4 entry point names the injected feedback SSBO");
    free(o);
    printf("t_bindless_spv: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
