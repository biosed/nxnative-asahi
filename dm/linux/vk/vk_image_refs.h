#pragma once

/* A Vulkan image has two kinds of references:
 *
 * - serial references, one for each live/snapshotted NVN texture serial in
 *   vk_replay.c's lookup table;
 * - external holds, currently the presentation thread while it copies an
 *   image to the swapchain.
 *
 * Finalizing a memory pool invalidates every serial backed by that pool, but
 * must not erase an external hold.  Keeping the counters separate prevents a
 * pool drop from scheduling an image twice when presentation releases it. */
static inline int vkx_image_serial_ref_add(int *refs, int *serial_refs,
                                            int serial_is_live)
{
    if (!serial_is_live) return 0;
    (*refs)++;
    (*serial_refs)++;
    return 1;
}

static inline void vkx_image_serial_ref_drop(int *refs, int *serial_refs)
{
    if (*serial_refs > 0) {
        (*serial_refs)--;
        (*refs)--;
    }
}

static inline void vkx_image_serial_refs_drop_all(int *refs, int *serial_refs)
{
    *refs -= *serial_refs;
    *serial_refs = 0;
}

static inline void vkx_image_hold_add(int *refs)
{
    (*refs)++;
}

static inline int vkx_image_hold_drop(int *refs)
{
    if (*refs > 0) (*refs)--;
    return *refs;
}

/* A stored image with no references remains reusable by a later texture over
 * the same guest bytes.  A detached image has no such future owner, so its
 * last external hold must retire it. */
static inline int vkx_image_hold_drop_retires(int *refs, const void *storage)
{
    return vkx_image_hold_drop(refs) == 0 && storage == NULL;
}
