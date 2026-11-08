// Darwin's xattr calls on Linux (see ../README.md): the position / options arguments go, XATTR_NOFOLLOW picks the l* variant; ENOATTR is ENODATA.
// Darwin application attributes have unqualified names (for example org.d2rmobile.guest-name). Linux rejects those with EOPNOTSUPP: ordinary application
// attributes live in the user.* namespace. Add that namespace at this platform boundary, while leaving an explicitly Linux-namespaced name unchanged.
#pragma once
#include_next <sys/xattr.h>
#include <errno.h>
#include <string.h>
#define XATTR_NOFOLLOW 1
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
static inline const char *dm_linux_xattr_name(const char *name, char out[256]) {
    if (!strncmp(name, "user.", 5) || !strncmp(name, "trusted.", 8) ||
        !strncmp(name, "security.", 9) || !strncmp(name, "system.", 7)) return name;
    size_t n = strlen(name);
    if (n + sizeof "user." > 256) { errno = ERANGE; return NULL; }
    memcpy(out, "user.", 5); memcpy(out + 5, name, n + 1); return out;
}
static inline ssize_t dm_linux_getxattr(const char *p, const char *name, void *v, size_t s, int opt) {
    char b[256]; name = dm_linux_xattr_name(name, b); if (!name) return -1;
    return opt & XATTR_NOFOLLOW ? lgetxattr(p, name, v, s) : getxattr(p, name, v, s);
}
static inline int dm_linux_setxattr(const char *p, const char *name, const void *v, size_t s, int opt) {
    char b[256]; name = dm_linux_xattr_name(name, b); if (!name) return -1;
    return opt & XATTR_NOFOLLOW ? lsetxattr(p, name, v, s, 0) : setxattr(p, name, v, s, 0);
}
static inline int dm_linux_removexattr(const char *p, const char *name, int opt) {
    char b[256]; name = dm_linux_xattr_name(name, b); if (!name) return -1;
    return opt & XATTR_NOFOLLOW ? lremovexattr(p, name) : removexattr(p, name);
}
#define getxattr(p, n, v, s, pos, opt) dm_linux_getxattr(p, n, v, s, opt)
#define setxattr(p, n, v, s, pos, opt) dm_linux_setxattr(p, n, v, s, opt)
#define removexattr(p, n, opt) dm_linux_removexattr(p, n, opt)
#define listxattr(p, b, s, opt) ((opt) & XATTR_NOFOLLOW ? llistxattr(p, b, s) : listxattr(p, b, s))
