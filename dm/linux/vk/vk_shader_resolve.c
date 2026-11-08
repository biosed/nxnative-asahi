#include "vk_shader_resolve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int read_text(const char *path, char **out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return 0; }
    long z = ftell(f);
    if (z < 0 || z > 1024 * 1024 || fseek(f, 0, SEEK_SET)) { fclose(f); return 0; }
    char *p = malloc((size_t)z + 1);
    if (!p) { fclose(f); return 0; }
    size_t n = fread(p, 1, (size_t)z, f); fclose(f); p[n] = 0;
    if (n != (size_t)z) { free(p); return 0; }
    *out = p; return 1;
}

static int key_ok(const char key[17]) {
    if (!key || strlen(key) != 16) return 0;
    for (int i = 0; i < 16; i++) if (!((key[i] >= '0' && key[i] <= '9') ||
        (key[i] >= 'a' && key[i] <= 'f'))) return 0;
    return 1;
}

static int mesh_first_rows(const char *txt) {
    int rows = 0;
    for (const char *p = txt; p && *p;) {
        const char *e = strchr(p, '\n'); size_t n = e ? (size_t)(e - p) : strlen(p);
        if (*p != '#' && n) {
            char line[512]; if (n >= sizeof line) return -1;
            memcpy(line, p, n); line[n] = 0;
            char *q = line, *fb = strsep(&q, "\t"), *fd = strsep(&q, "\t"), *fk = strsep(&q, "\t");
            if (!fb || !fd || !fk) return -1;
            char *end = NULL; unsigned long b = strtoul(fb, &end, 10);
            int valid = end && !*end && b <= UINT32_MAX &&
                        vkx_mesh_first_descriptor((uint32_t)b, fd[0], fk);
            if ((!strcmp(fk, "mesh_first") || (end && !*end && b == DMSH_LINUX_VK_MESH_FIRST_BINDING)) && !valid)
                return -1;
            rows += valid;
        }
        if (!e) break; p = e + 1;
    }
    return rows;
}

static int one(const char *dir, int mesh_ext, const char key[17],
               int wanted_stage, vkx_shader_resolved *out) {
    if (!dir || !*dir) return 0;
    vkx_shader_resolved r = {0};
    if (snprintf(r.dir, sizeof r.dir, "%s", dir) >= (int)sizeof r.dir ||
        snprintf(r.spv, sizeof r.spv, "%s/%s.vk.spv", dir, key) >= (int)sizeof r.spv ||
        snprintf(r.vkb, sizeof r.vkb, "%s/%s.vkb", dir, key) >= (int)sizeof r.vkb ||
        snprintf(r.job, sizeof r.job, "%s/%s.job.json", dir, key) >= (int)sizeof r.job)
        return 0;
    if (access(r.spv, R_OK) || access(r.vkb, R_OK)) return 0;
    char *vkb = NULL; if (!read_text(r.vkb, &vkb)) return 0;
    char got_key[17] = {0}, name[8] = {0}, tail = 0, header[128]; int set = -1;
    size_t hl = strcspn(vkb, "\r\n"); if (hl >= sizeof header) hl = sizeof header - 1;
    memcpy(header, vkb, hl); header[hl] = 0;
    int nf = sscanf(header, "# %16s %7s set %d %c", got_key, name, &set, &tail);
    int first_rows = mesh_ext ? mesh_first_rows(vkb) : 0;
    int st = vkx_stage_of_name(name), native = st == VKX_STAGE_TS || st == VKX_STAGE_MS;
    if (nf != 3 || strcmp(got_key, key) || st < 0 || set != vkx_stage_set(st) ||
        native != mesh_ext || (wanted_stage >= 0 && st != wanted_stage) || first_rows < 0 || first_rows > 1) {
        free(vkb); return 0;
    }
    r.stage = st; r.descriptor_set = set; r.mesh_ext = mesh_ext;
    if (native) {
        char *job = NULL;
        if (!read_text(r.job, &job) || !vkx_mesh_job_meta(job, st, &r.mesh_meta)) { free(job); free(vkb); return 0; }
        free(job);
        r.mesh_meta.mesh_first = first_rows == 1;
        if (st == VKX_STAGE_MS && r.mesh_meta.mesh_has_task && r.mesh_meta.mesh_first) { free(vkb); return 0; }
    } else r.job[0] = 0;
    free(vkb);
    *out = r; return 1;
}

int vkx_shader_resolve(const char *standard_dir, const char *mesh_dir,
                       const char key[17], int wanted_stage,
                       vkx_shader_resolved *out) {
    if (!out || !key_ok(key) || wanted_stage >= VKX_META_STAGE_COUNT) return 0;
    if (wanted_stage == VKX_STAGE_TS || wanted_stage == VKX_STAGE_MS)
        return one(mesh_dir, 1, key, wanted_stage, out);
    if (wanted_stage >= 0) return one(standard_dir, 0, key, wanted_stage, out);
    // A native artifact wins a duplicate only because the two ABIs are kept in
    // different roots and its sidecars prove stage/set/job agreement.
    return one(mesh_dir, 1, key, -1, out) || one(standard_dir, 0, key, -1, out);
}
