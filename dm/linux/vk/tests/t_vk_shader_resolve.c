#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "vk_shader_resolve.h"
#include "spv_reflect.h"
#include "nvn_shader_stage.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }
static void put(const char *p, const char *s) { FILE *f = fopen(p, "wb"); check(f != NULL, "create fixture"); if (f) { fwrite(s, 1, strlen(s), f); fclose(f); } }
static void paths(char *p, size_t n, const char *d, const char *k, const char *x) { snprintf(p, n, "%s/%s.%s", d, k, x); }

static void fixture(void) {
    uint8_t ctl[0x57] = {0}; uint32_t magic = 0x98761234, ver = 0xc9, sph;
    memcpy(ctl, &magic, 4); memcpy(ctl + 4, &ver, 4);
    ctl[0x56] = 6; check(nvn2_shader_stage(ctl, NULL) == VKX_STAGE_TS, "NVN2 control identifies task despite SPH");
    ctl[0x56] = 7; check(nvn2_shader_stage(ctl, NULL) == VKX_STAGE_MS, "NVN2 control identifies mesh despite SPH");
    ctl[0x56] = 0; sph = 5u << 10; check(nvn2_shader_stage(ctl, &sph) == VKX_STAGE_PS, "NVN2 SPH identifies fragment stage");
    ver = 0; memcpy(ctl + 4, &ver, 4); check(nvn2_shader_stage(ctl, &sph) < 0, "non-NVN2 stage is not guessed");
    char top[] = "/tmp/t-vkx-resolve-XXXXXX"; check(mkdtemp(top) != NULL, "mkdtemp");
    char sd[256], md[256], p[512]; snprintf(sd, sizeof sd, "%s/std", top); snprintf(md, sizeof md, "%s/mesh", top);
    check(!mkdir(sd, 0700) && !mkdir(md, 0700), "fixture directories");
    const char *vs = "0011223344556677", *ts = "efaa4fa6a80ba719", *ms = "554cf8c05126a133";
    paths(p, sizeof p, sd, vs, "vk.spv"); put(p, "spv"); paths(p, sizeof p, sd, vs, "vkb"); put(p, "# 0011223344556677 vs set 0\n");
    paths(p, sizeof p, md, ts, "vk.spv"); put(p, "spv"); paths(p, sizeof p, md, ts, "vkb"); put(p, "# efaa4fa6a80ba719 ts set 3\n512\tu\tmesh_first\t\t\t\t\t\t0\t0\t0\t0\t1\n");
    paths(p, sizeof p, md, ts, "job.json"); put(p, "{\"stage\":\"ts\",\"local\":[32,1,1],\"shared\":0,\"mesh\":[0,0,256],\"mesh_has_task\":false}");
    paths(p, sizeof p, md, ms, "vk.spv"); put(p, "spv"); paths(p, sizeof p, md, ms, "vkb"); put(p, "# 554cf8c05126a133 ms set 4\n");
    paths(p, sizeof p, md, ms, "job.json"); put(p, "{\"stage\":\"ms\",\"local\":[32,1,1],\"shared\":0,\"mesh\":[128,64,256],\"mesh_has_task\":true}");
    vkx_shader_resolved r;
    check(vkx_shader_resolve(sd, md, vs, VKX_STAGE_VS, &r) && !r.mesh_ext && r.stage == VKX_STAGE_VS && r.descriptor_set == 0 && !r.job[0], "ordinary shader resolves only from standard root");
    check(vkx_shader_resolve(sd, md, ts, VKX_STAGE_TS, &r) && r.mesh_ext && r.stage == VKX_STAGE_TS && r.descriptor_set == 3 && r.mesh_meta.payload == 256 && r.mesh_meta.mesh_first, "task shader resolves with set-3 mesh_first job");
    check(vkx_shader_resolve(sd, md, ms, VKX_STAGE_MS, &r) && r.mesh_ext && r.stage == VKX_STAGE_MS && r.descriptor_set == 4 && r.mesh_meta.max_vertices == 128 && r.mesh_meta.mesh_has_task_known && r.mesh_meta.mesh_has_task && !r.mesh_meta.mesh_first, "paired mesh resolves without a second mesh_first binding");
    paths(p, sizeof p, md, ms, "vkb"); put(p, "# 554cf8c05126a133 ms set 4\n512\tu\tmesh_first\t\t\t\t\t\t0\t0\t0\t0\t1\n");
    check(!vkx_shader_resolve(sd, md, ms, VKX_STAGE_MS, &r), "paired mesh with a second mesh_first binding fails closed");
    paths(p, sizeof p, md, ms, "vkb"); put(p, "# 554cf8c05126a133 ms set 4\n");
    paths(p, sizeof p, md, ms, "job.json"); put(p, "{\"stage\":\"ts\",\"local\":[32,1,1],\"shared\":0,\"mesh\":[0,0,256]}");
    check(!vkx_shader_resolve(sd, md, ms, VKX_STAGE_MS, &r), "job stage mismatch fails closed");
    paths(p, sizeof p, md, ms, "job.json"); unlink(p);
    paths(p, sizeof p, md, ms, "vkb"); unlink(p); paths(p, sizeof p, md, ms, "vk.spv"); unlink(p);
    paths(p, sizeof p, md, ts, "job.json"); unlink(p); paths(p, sizeof p, md, ts, "vkb"); unlink(p); paths(p, sizeof p, md, ts, "vk.spv"); unlink(p);
    paths(p, sizeof p, sd, vs, "vkb"); unlink(p); paths(p, sizeof p, sd, vs, "vk.spv"); unlink(p);
    rmdir(sd); rmdir(md); rmdir(top);
}

static int count_suffix(const char *dir, const char *suf) {
    DIR *d = opendir(dir); if (!d) return -1; int n = 0; size_t z = strlen(suf);
    for (struct dirent *e; (e = readdir(d));) { size_t l = strlen(e->d_name); if (l >= z && !strcmp(e->d_name + l - z, suf)) n++; }
    closedir(d); return n;
}

static void title(const char *sd, const char *md) {
    check(count_suffix(sd, ".vkb") == 876, "standard title directory remains the 876-module set");
    check(count_suffix(md, ".vkb") == 506, "segregated Cycle29 directory has 506 modules");
    static const char *const key[] = {"efaa4fa6a80ba719", "554cf8c05126a133"};
    static const int stage[] = {VKX_STAGE_TS, VKX_STAGE_MS};
    for (int i = 0; i < 2; i++) {
        vkx_shader_resolved r; check(vkx_shader_resolve(sd, md, key[i], stage[i], &r), "exact title key resolves from segregated directory");
        if (!vkx_shader_resolve(sd, md, key[i], stage[i], &r)) continue;
        check(i ? (r.mesh_meta.mesh_has_task && !r.mesh_meta.mesh_first) : r.mesh_meta.mesh_first,
              "exact first task/mesh pair owns mesh_first only in task");
        FILE *f = fopen(r.spv, "rb"); check(f != NULL, "open canonical title SPIR-V"); if (!f) continue;
        fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = malloc((size_t)z); size_t got = fread(w, 1, (size_t)z, f); fclose(f);
        spvr_info ri; char err[160]; int ok = z > 0 && !(z & 3) && got == (size_t)z && spvr_reflect(w, (size_t)z / 4, &ri, err, sizeof err) == 0;
        check(ok, "canonical title SPIR-V reflects");
        check(ok && ri.stage == vkx_stage_vk_bit(stage[i]) && !memcmp(ri.local_size, r.mesh_meta.local, sizeof ri.local_size) && ri.shared_size >= r.mesh_meta.shared,
              "SPIR-V stage/local/shared agrees with resolver job metadata");
        free(w);
    }
}

int main(int argc, char **argv) {
    fixture();
    if (argc == 4 && !strcmp(argv[1], "--title")) title(argv[2], argv[3]);
    else check(argc == 1, "arguments are empty or --title <standard> <mesh>");
    printf("t_vk_shader_resolve: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
