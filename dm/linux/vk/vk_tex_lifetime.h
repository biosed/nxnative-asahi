#pragma once

#include <sys/uio.h>
#include <unistd.h>
#include "nvn_obj.h"

/* NVNtexture is plain data, so engines may finalize a copied snapshot while
 * the object passed to TextureInitialize remains live.  Read the owner through
 * the kernel: it can have been freed or unmapped by the time the copy reaches
 * Finalize, and a direct dereference would fault. */
static inline int vkx_tex_release_is_copy(const tex_t *owner,
                                          const tex_t *released)
{
    if (!owner || owner == released) return 0;
    tex_t seen;
    struct iovec local = {&seen, sizeof seen};
    struct iovec remote = {(void *)owner, sizeof seen};
    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return n == (ssize_t)sizeof seen &&
           seen.magic == MAGIC('T', 'E', 'X', 0) &&
           seen.serial == released->serial;
}
