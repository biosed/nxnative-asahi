// Real folder-backed OpenDirectory: allocation failure must return an error,
// leave the output untouched, and close the DIR already opened by the handler.
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>

static void *directory_calloc(size_t count, size_t size);
static DIR *directory_opendir(const char *path);
static int directory_closedir(DIR *d);
#define calloc directory_calloc
#define opendir directory_opendir
#define closedir directory_closedir
#include "../hle_fs.c"
#undef closedir
#undef opendir
#undef calloc

struct dm_flags g_flag;
uint64_t g_t0_ns;
uint64_t now_ns(void) { return 0; }
void dm_guest_write_prepare(void *a, size_t n) { (void)a; (void)n; }

static int fail_handle, failed_allocations, opened_fd = -1;
static int calloc_calls;
static size_t calloc_bytes;
static void *directory_calloc(size_t count, size_t size) {
    calloc_calls++;
    calloc_bytes += count * size;
    if (fail_handle && count == 1 && size == sizeof(dirh_t)) {
        failed_allocations++;
        errno = ENOMEM;
        return NULL;
    }
    return calloc(count, size);
}
static DIR *directory_opendir(const char *path) {
    DIR *d = opendir(path);
    if (d) opened_fd = dirfd(d);
    return d;
}

static int directory_closedir(DIR *d) {
    int r = closedir(d);
    // Successful cleanup may change errno; it must not hide the allocation error.
    if (fail_handle) errno = ENOENT;
    return r;
}

static int checks, fails;
#define CHECK(c, text) do { checks++; int ok = !!(c); fails += !ok; printf("%s %s\n", ok ? "PASS" : "FAIL", text); } while (0)

int main(void) {
    char root[1024], file[1100];
    const char *tmp = getenv("TMPDIR");
    snprintf(root, sizeof root, "%s/t_fs_directory.XXXXXX", tmp ? tmp : "/private/tmp");
    if (!mkdtemp(root)) { perror("mkdtemp"); return 2; }
    snprintf(file, sizeof file, "%s/entry", root);
    FILE *f = fopen(file, "w");
    if (!f) { perror("fopen"); rmdir(root); return 2; }
    fputs("hello", f); fclose(f);
    CHECK(mount("test", root, 0) == RES_OK, "mount the synthetic folder");

    const uint64_t initial[] = {0, 0x1234};
    for (size_t i = 0; i < sizeof initial / sizeof initial[0]; i++) {
        uint64_t h = initial[i];
        opened_fd = -1; failed_allocations = 0; fail_handle = 1;
        Result r = h_fs_OpenDirectory(&h, "test:/", DIRMODE_DIR | DIRMODE_FILE);
        int error = errno;
        fail_handle = 0;
        CHECK(failed_allocations == 1, "inject failure into the directory-handle allocation");
        CHECK(r == RES_FAIL, "failed allocation returns the filesystem resource failure");
        CHECK(error == ENOMEM, "allocation error survives directory cleanup");
        CHECK(h == initial[i], "failed open leaves the caller's handle untouched");
        CHECK(opened_fd >= 0, "the handler opened a real directory before allocation failed");
        errno = 0;
        CHECK(fcntl(opened_fd, F_GETFD) == -1 && errno == EBADF, "failed open closes the real directory descriptor");
    }
    h_fs_CloseDirectory(0);

    uint64_t h = 0;
    Result r = h_fs_OpenDirectory(&h, "test:/", DIRMODE_DIR | DIRMODE_FILE);
    int fd = opened_fd;
    CHECK(r == RES_OK && h != 0, "a later successful open publishes a handle");
    if (r == RES_OK && h) {
        CHECK(fcntl(fd, F_GETFD) >= 0, "successful open keeps its directory descriptor alive");
        int64_t count = -1;
        CHECK(h_fs_DirCount(&count, h) == RES_OK && count == 1, "the successful handle counts the one synthetic file");
        DirectoryEntry entry[2] = {0};
        CHECK(h_fs_ReadDirectory(&count, entry, h, 2) == RES_OK && count == 1 &&
              !strcmp(entry[0].name, "entry") && entry[0].type == ENT_FILE && entry[0].size == 5,
              "the successful handle reads the file name, type and size");
        h_fs_CloseDirectory(h);
        errno = 0;
        CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF, "CloseDirectory releases the successful handle's descriptor");
    }
    CHECK(!unlink(file) && !rmdir(root), "remove the synthetic fixture");

    // RomFS image directory: OpenDirectory allocates only dirh_t (no per-child DirectoryEntry snapshot) and fails cleanly on OOM
    {   enum { NFILES = 64 };
        char ip[1200]; snprintf(ip, sizeof ip, "%s.img", root);
        char names[4 + NFILES * 5]; memcpy(names, "big", 3); size_t noff = 3;
        img_ent ent[2 + NFILES];
        ent[0] = (img_ent){1, 1, 0, 0, 0, 0};
        ent[1] = (img_ent){2, NFILES, 0, 3, 0, 0};
        for (int i = 0; i < NFILES; i++) {
            char nm[8]; int len = snprintf(nm, sizeof nm, "f%03d", i);
            memcpy(names + noff, nm, (size_t)len);
            ent[2 + i] = (img_ent){32, (uint64_t)(i + 1), (uint32_t)noff, (uint16_t)len, 1, 0};
            noff += (size_t)len;
        }
        uint64_t hd[3] = {32 + NFILES, (2 + NFILES) * sizeof(img_ent) + noff, 2 + NFILES};
        uint8_t data[NFILES] = {0};
        FILE *im = fopen(ip, "wb");
        fwrite("DMROMFS1", 1, 8, im); fwrite(hd, 1, sizeof hd, im); fwrite(data, 1, sizeof data, im);
        fwrite(ent, 1, sizeof ent, im); fwrite(names, 1, noff, im); fclose(im);
        CHECK(mount("img", ip, 1) == RES_OK, "mount a synthetic RomFS image");
        uint64_t ih = 0x5678; failed_allocations = 0; fail_handle = 1;
        Result ir = h_fs_OpenDirectory(&ih, "img:/big", DIRMODE_FILE);
        int ierr = errno; fail_handle = 0;
        CHECK(failed_allocations == 1 && ir == RES_FAIL && ierr == ENOMEM && ih == 0x5678, "image directory allocation failure returns RES_FAIL with ENOMEM and leaves handle untouched");
        calloc_calls = 0; calloc_bytes = 0; ih = 0;
        CHECK(h_fs_OpenDirectory(&ih, "img:/big", DIRMODE_FILE) == RES_OK && ih != 0, "image directory opens");
        CHECK(calloc_calls == 1 && calloc_bytes == sizeof(dirh_t), "image directory OpenDirectory allocates only dirh_t, not a per-child snapshot");
        DirectoryEntry one[2]; int64_t icnt = -1;
        CHECK(h_fs_ReadDirectory(&icnt, one, ih, 1) == RES_OK && icnt == 1 && !strcmp(one[0].name, "f000") && one[0].size == 1, "first 1-entry read returns f000");
        CHECK(h_fs_DirCount(&icnt, ih) == RES_OK && icnt == NFILES, "DirCount returns all 64 files without moving the cursor");
        CHECK(h_fs_ReadDirectory(&icnt, one, ih, 1) == RES_OK && icnt == 1 && !strcmp(one[0].name, "f001") && one[0].size == 2, "second 1-entry read returns f001");
        h_fs_CloseDirectory(ih);
        unlink(ip);
    }

    printf("== fs directory: %d checks, %d failed ==\n", checks, fails);
    return fails != 0;
}
