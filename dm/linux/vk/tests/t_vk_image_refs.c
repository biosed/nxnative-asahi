#include <stdio.h>
#include <sys/mman.h>
#include "vk_image_refs.h"
#include "vk_tex_lifetime.h"

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
    int refs = 0, serial_refs = 0;

    vkx_image_serial_ref_add(&refs, &serial_refs, 1);
    vkx_image_serial_ref_add(&refs, &serial_refs, 1);
    vkx_image_hold_add(&refs);
    check(refs == 3 && serial_refs == 2,
          "serial mappings and the presentation hold are counted separately");

    vkx_image_serial_refs_drop_all(&refs, &serial_refs);
    check(refs == 1 && serial_refs == 0,
          "pool finalization drops serial mappings but preserves presentation");
    check(vkx_image_hold_drop_retires(&refs, NULL),
          "the last presentation hold makes the image retireable exactly once");

    refs = 1;
    check(!vkx_image_hold_drop_retires(&refs, &refs) && refs == 0,
          "a stored image remains reusable after its last external hold");

    check(!vkx_image_serial_ref_add(&refs, &serial_refs, 0) && !refs,
          "a snapshot finalized before replay does not pin its image");
    vkx_image_serial_ref_add(&refs, &serial_refs, 1);
    vkx_image_serial_ref_drop(&refs, &serial_refs);
    check(refs == 0 && serial_refs == 0,
          "ordinary texture finalization drops one serial mapping");

    tex_t owner = {.magic = MAGIC('T', 'E', 'X', 0), .serial = 71};
    tex_t copy = owner;
    check(vkx_tex_release_is_copy(&owner, &copy),
          "finalizing a copy does not end the initialized texture object");
    check(!vkx_tex_release_is_copy(&owner, &owner),
          "finalizing the initialized object ends its serial");
    owner.serial++;
    check(!vkx_tex_release_is_copy(&owner, &copy),
          "a reused owner slot cannot keep an older serial live");

    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    tex_t *gone = mmap(NULL, pg, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(gone != MAP_FAILED, "temporary texture owner page maps");
    if (gone != MAP_FAILED) {
        *gone = copy;
        munmap(gone, pg);
        check(!vkx_tex_release_is_copy(gone, &copy),
              "an unmapped owner is treated as the last release without faulting");
    }

    printf("t_vk_image_refs: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
