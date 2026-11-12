#include <stdio.h>
#include "vk_cube.h"

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
    check(vkx_cube_compatible(0, 256, 256, 6, 1), "a 2D array of six square layers can be a cube");
    check(vkx_cube_compatible(0, 64, 64, 12, 1), "two cubes' worth of square layers can be a cube array");
    check(!vkx_cube_compatible(0, 64, 64, 7, 1), "a partial cube is not made cube-compatible");
    check(!vkx_cube_compatible(0, 256, 128, 6, 1), "non-square layers cannot be a cube");
    check(!vkx_cube_compatible(0, 256, 256, 5, 1), "fewer than six layers cannot be a cube");
    check(!vkx_cube_compatible(0, 256, 256, 1, 1), "a single 2D layer is not cube-compatible");
    check(!vkx_cube_compatible(1, 256, 256, 6, 1), "a 3D image is never cube-compatible");
    check(!vkx_cube_compatible(0, 256, 256, 6, 4), "a multisampled image is never cube-compatible");
    check(!vkx_cube_compatible(0, 0, 0, 6, 1), "an empty extent is not cube-compatible");
    check(!vkx_cube_compatible(0, 256, 256, 6, 0), "a sample count of zero is not cube-compatible");

    check(vkx_cube_view_ok(1, 6), "a cube view of a six-layer cube-compatible image");
    check(vkx_cube_view_ok(1, 18), "a cube-array view of three cubes");
    check(!vkx_cube_view_ok(0, 6), "no cube view of an image created without CUBE_COMPATIBLE (the assert this fixes)");
    check(!vkx_cube_view_ok(1, 7), "no cube view of a partial cube");
    check(!vkx_cube_view_ok(1, 0), "no cube view of no layers");

    printf("t_vk_cube: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
