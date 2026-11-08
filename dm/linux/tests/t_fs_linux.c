// Linux save-create seam: Darwin's unqualified guest-name xattr must map to
// user.*, and repeated identical host errors must have bounded diagnostics.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>
#include "fs_error_rate.h"

static int failures;
static void check(int ok, const char *what) {
    if (!ok) { failures++; printf("FAIL: %s (errno=%d)\n", what, errno); }
}

int main(void) {
    char dir[] = "/tmp/t_fs_linux.XXXXXX", path[256], value[32] = {0};
    check(mkdtemp(dir) != NULL, "temporary directory");
    snprintf(path, sizeof path, "%s/account.sav", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    check(fd >= 0, "open(O_CREAT|O_EXCL)");
    check(fd >= 0 && !ftruncate(fd, 4096), "initial save size");
    if (fd >= 0) close(fd);

    errno = 0;
    check(lremovexattr(path, "org.d2rmobile.guest-name") < 0 &&
          errno == EOPNOTSUPP,
          "raw unnamespaced lremovexattr reproduces the save-create failure");
    errno = 0;
    check(removexattr(path, "org.d2rmobile.guest-name", XATTR_NOFOLLOW) < 0 &&
          errno == ENOATTR, "missing Darwin xattr is ENOATTR, not EOPNOTSUPP");
    check(!setxattr(path, "org.d2rmobile.guest-name", "Account.sav", 11, 0,
                    XATTR_NOFOLLOW), "set translated guest-name xattr");
    check(getxattr(path, "org.d2rmobile.guest-name", value, sizeof value, 0,
                   XATTR_NOFOLLOW) == 11 && !memcmp(value, "Account.sav", 11),
          "get translated guest-name xattr");
    memset(value, 0, sizeof value);
    check(lgetxattr(path, "user.org.d2rmobile.guest-name", value,
                    sizeof value) == 11 && !memcmp(value, "Account.sav", 11),
          "physical Linux name is in user namespace");
    check(!removexattr(path, "org.d2rmobile.guest-name", XATTR_NOFOLLOW),
          "remove translated guest-name xattr");

    fs_error_rate rate = FS_ERROR_RATE_INIT; uint64_t count = 0; int reports = 0;
    for (int i = 0; i < 1000000; i++)
        reports += fs_error_rate_hit(&rate, "create", path, EOPNOTSUPP, &count);
    check(count == 1000000 && reports == 20,
          "one million identical failures produce 20 power-of-two reports");
    check(fs_error_rate_hit(&rate, "create", path, ENOSPC, &count) && count == 1,
          "a different errno still reports its first occurrence");
    pthread_mutex_destroy(&rate.lock);

    unlink(path); rmdir(dir);
    printf("t_fs_linux: %s (%d failures)\n", failures ? "FAILED" : "passed", failures);
    return failures != 0;
}
