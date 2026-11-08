#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spv_reflect.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

static unsigned char *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    if (z < 0) { fclose(f); return NULL; }
    unsigned char *p = malloc((size_t)z); if (!p) { fclose(f); return NULL; }
    *n = fread(p, 1, (size_t)z, f); fclose(f); return p;
}

static void selftest(void) {
    uint32_t w[] = {
        0x07230203, 0x00010600, 0, 32, 0,
        (5u << 16) | 15u, 5365, 1, 0x6e69616d, 0,
        (6u << 16) | 16u, 1, 17, 32, 1, 1,
        (4u << 16) | 16u, 1, 26, 8,
        (4u << 16) | 16u, 1, 5270, 4,
        (3u << 16) | 16u, 1, 5298,
        (4u << 16) | 71u, 10, 30, 5,
        (4u << 16) | 71u, 10, 31, 1,
        (3u << 16) | 71u, 10, 5271,
        (5u << 16) | 72u, 11, 0, 11, 0,
        (4u << 16) | 71u, 18, 30, 33,
        (4u << 16) | 71u, 22, 30, 7,
        (4u << 16) | 71u, 23, 11, 1,
        (4u << 16) | 71u, 24, 11, 3,
        (4u << 16) | 71u, 25, 11, 4,
        (4u << 16) | 71u, 26, 11, 7,
        (4u << 16) | 71u, 27, 11, 9,
        (4u << 16) | 71u, 28, 11, 10,
        (4u << 16) | 71u, 29, 11, 5299,
        (4u << 16) | 71u, 30, 11, 4432,
        (4u << 16) | 71u, 31, 11, 5296,
        (3u << 16) | 22u, 2, 32,
        (4u << 16) | 23u, 3, 2, 4,
        (4u << 16) | 21u, 4, 32, 0,
        (4u << 16) | 43u, 4, 5, 8,
        (4u << 16) | 28u, 6, 3, 5,
        (4u << 16) | 32u, 7, 3, 6,
        (4u << 16) | 59u, 7, 10, 3,
        (3u << 16) | 30u, 11, 3,
        (4u << 16) | 28u, 12, 11, 5,
        (4u << 16) | 32u, 13, 3, 12,
        (4u << 16) | 59u, 13, 14, 3,
        (3u << 16) | 22u, 15, 64,
        (4u << 16) | 28u, 16, 15, 5,
        (4u << 16) | 32u, 17, 3, 16,
        (4u << 16) | 59u, 17, 18, 3,
        (3u << 16) | 22u, 19, 16,
        (4u << 16) | 28u, 20, 19, 5,
        (4u << 16) | 32u, 21, 3, 20,
        (4u << 16) | 59u, 21, 22, 3,
        (4u << 16) | 59u, 7, 23, 3,
        (4u << 16) | 59u, 7, 24, 3,
        (4u << 16) | 59u, 7, 25, 3,
        (4u << 16) | 59u, 7, 26, 3,
        (4u << 16) | 59u, 7, 27, 3,
        (4u << 16) | 59u, 7, 28, 3,
        (4u << 16) | 59u, 7, 29, 3,
        (4u << 16) | 59u, 7, 30, 3,
        (4u << 16) | 59u, 7, 31, 3,
    };
    spvr_info r; char err[160];
    check(spvr_reflect(w, sizeof w / sizeof *w, &r, err, sizeof err) == 0, "synthetic MeshEXT reflection");
    check(r.stage == VK_SHADER_STAGE_MESH_BIT_EXT && r.output_topology == SPVR_TOPOLOGY_TRIANGLES &&
          r.max_vertices == 8 && r.max_primitives == 4 && r.local_size[0] == 32,
          "topology, limits and local size reflected");
    check(r.output_builtin_mask == (SPVR_BUILTIN_POSITION | SPVR_BUILTIN_POINT_SIZE |
          SPVR_BUILTIN_CLIP_DISTANCE | SPVR_BUILTIN_CULL_DISTANCE | SPVR_BUILTIN_PRIMITIVE_ID |
          SPVR_BUILTIN_LAYER | SPVR_BUILTIN_VIEWPORT_INDEX | SPVR_BUILTIN_CULL_PRIMITIVE |
          SPVR_BUILTIN_PRIMITIVE_SHADING_RATE) && r.noutput == 3 && r.output_components == 6,
          "all ABI built-ins classified, primitive indices excluded, and user outputs counted");
    check(r.output[0].location == 5 && r.output[0].component == 1 && r.output[0].components == 4 &&
          r.output[0].bits == 32 && r.output[0].kind == 'f' && r.output[0].per_primitive,
          "32-bit per-primitive output reflected");
    check(r.output[1].location == 33 && r.output[1].bits == 64 && r.output[1].components == 1,
          "64-bit out-of-range output reflected");
    check(r.output[2].location == 7 && r.output[2].bits == 16 && r.output[2].components == 1,
          "16-bit output reflected");
    check(r.output_size == 1488, "complete output byte footprint reflected");
}

enum {
    C_TOPOLOGY = 1ull << 0, C_BITS16 = 1ull << 1, C_BITS64 = 1ull << 2,
    C_BITS_OTHER = 1ull << 3, C_LOCATION = 1ull << 4, C_KIND = 1ull << 5,
    C_POINT_SIZE = 1ull << 6, C_CLIP = 1ull << 7, C_CULL = 1ull << 8,
    C_PRIMITIVE_ID = 1ull << 9, C_LAYER = 1ull << 10, C_VIEWPORT = 1ull << 11,
    C_CULL_PRIMITIVE = 1ull << 12, C_SHADING_RATE = 1ull << 13, C_BUILTIN_OTHER = 1ull << 14,
    C_OUTPUT_32K = 1ull << 15, C_OUTPUT_8K = 1ull << 16, C_COMPONENTS4 = 1ull << 17,
    C_PAYLOAD_OUTPUT_8K = 1ull << 18, C_SHARED_24K = 1ull << 19,
    C_PAYLOAD_SHARED_24K = 1ull << 20, C_NO_USER = 1ull << 21,
};

typedef struct {
    char key[17]; uint64_t c; spvr_info r;
    uint32_t pv_components, pp_components, max_location;
} rec;

static void key_of(const char *path, char key[17]) {
    const char *b = strrchr(path, '/'); b = b ? b + 1 : path;
    snprintf(key, 17, "%.16s", b);
}

static void class_line(const char *name, const rec *r, int n, uint64_t bit) {
    int count = 0; for (int i = 0; i < n; i++) if (r[i].c & bit) count++;
    printf("CLASS\t%s\t%d", name, count);
    for (int i = 0; i < n; i++) if (r[i].c & bit) printf("\t%s", r[i].key);
    putchar('\n');
}

static void class_mask_line(const char *name, const rec *r, int n, uint64_t mask) {
    int count = 0; for (int i = 0; i < n; i++) if (r[i].c & mask) count++;
    printf("CLASS\t%s\t%d", name, count);
    for (int i = 0; i < n; i++) if (r[i].c & mask) printf("\t%s", r[i].key);
    putchar('\n');
}

static void builtin_line(const char *name, const rec *r, int n, uint32_t bit, int keys) {
    int count = 0; for (int i = 0; i < n; i++) if (r[i].r.output_builtin_mask & bit) count++;
    printf("BUILTIN\t%s\t%d", name, count);
    if (keys) for (int i = 0; i < n; i++) if (r[i].r.output_builtin_mask & bit) printf("\t%s", r[i].key);
    putchar('\n');
}

static const char *topology_name(unsigned t) {
    return t == SPVR_TOPOLOGY_POINTS ? "points" : t == SPVR_TOPOLOGY_LINES ? "lines" :
           t == SPVR_TOPOLOGY_TRIANGLES ? "triangles" : "none";
}

static void detail(const char *role, const char *key, const spvr_info *r) {
    printf("FOCUS\t%s\t%s\tstage=%#x\ttopology=%s\tlocal=%u,%u,%u\tvertices=%u\tprimitives=%u\tshared=%u\tpayload=%u\toutput=%u\tcomponents=%u\tbuiltins=%#x",
        role, key, r->stage, topology_name(r->output_topology), r->local_size[0], r->local_size[1], r->local_size[2],
        r->max_vertices, r->max_primitives, r->shared_size, r->task_payload_size, r->output_size,
        r->output_components, r->output_builtin_mask);
    for (int i = 0; i < r->noutput; i++) printf("\t%s:loc%u.c%u=%c%ux%u", r->output[i].per_primitive ? "pp" : "pv",
        r->output[i].location, r->output[i].component, r->output[i].kind, r->output[i].bits, r->output[i].components);
    putchar('\n');
}

static void coverage(int argc, char **argv) {
    if (argc == 1) return;
    check(argc >= 5 && !strcmp(argv[1], "--coverage"), "coverage arguments");
    if (argc < 5 || strcmp(argv[1], "--coverage")) return;
    int want_ts = atoi(argv[2]), want_ms = atoi(argv[3]), nts = 0, nms = 0, nr = 0;
    rec *rr = calloc((size_t)(argc - 4), sizeof *rr); check(rr != NULL, "coverage allocation"); if (!rr) return;
    spvr_info focus_ts = {0}, focus_ms = {0}; int have_ts = 0, have_ms = 0;
    uint32_t max_output = 0; char max_output_key[17] = "";
    uint32_t max_shared = 0, max_payload = 0, max_components = 0, max_location = 0;
    uint64_t total_pv = 0, total_pp = 0; int modules_pv = 0, modules_pp = 0;
    uint32_t pv_decl[128] = {0}, pp_decl[128] = {0}, pv_comp[128] = {0}, pp_comp[128] = {0};
    uint32_t kind_f = 0, kind_i = 0, kind_u = 0, kind_other = 0;
    int top[4] = {0}, widths16 = 0, widths32 = 0, widths64 = 0;
    for (int a = 4; a < argc; a++) {
        size_t nb = 0; unsigned char *b = read_all(argv[a], &nb); spvr_info s; char err[160];
        int ok = b && !(nb & 3) && spvr_reflect((const uint32_t *)b, nb / 4, &s, err, sizeof err) == 0;
        check(ok, "coverage SPIR-V reflection"); if (!ok) { free(b); continue; }
        char key[17]; key_of(argv[a], key);
        if (s.stage == VK_SHADER_STAGE_TASK_BIT_EXT) { nts++; if (!strcmp(key, "efaa4fa6a80ba719")) { focus_ts = s; have_ts = 1; } free(b); continue; }
        check(s.stage == VK_SHADER_STAGE_MESH_BIT_EXT, "coverage stage is task or mesh"); if (s.stage != VK_SHADER_STAGE_MESH_BIT_EXT) { free(b); continue; }
        nms++; rec *x = &rr[nr++]; memcpy(x->key, key, 17); x->r = s;
        if (s.output_topology < 4) top[s.output_topology]++;
        if (s.output_topology != SPVR_TOPOLOGY_TRIANGLES) x->c |= C_TOPOLOGY;
        if (!s.noutput) x->c |= C_NO_USER;
        for (int i = 0; i < s.noutput; i++) {
            const spvr_output *o = &s.output[i]; uint32_t last = o->location + (o->component + o->components - 1) / 4;
            if (last > x->max_location) x->max_location = last;
            if (last > 31) x->c |= C_LOCATION;
            if (o->bits == 16) { x->c |= C_BITS16; widths16++; }
            else if (o->bits == 32) widths32++;
            else if (o->bits == 64) { x->c |= C_BITS64; widths64++; }
            else x->c |= C_BITS_OTHER;
            if (o->kind != 'f' && o->kind != 'i' && o->kind != 'u') x->c |= C_KIND;
            if (o->kind == 'f') kind_f++; else if (o->kind == 'i') kind_i++;
            else if (o->kind == 'u') kind_u++; else kind_other++;
            if (o->location < 128) {
                uint32_t *decl = o->per_primitive ? pp_decl : pv_decl;
                uint32_t *comp = o->per_primitive ? pp_comp : pv_comp;
                decl[o->location]++; comp[o->location] += o->components;
            }
            if (o->per_primitive) x->pp_components += o->components; else x->pv_components += o->components;
        }
        if (x->pv_components) { modules_pv++; total_pv += x->pv_components; }
        if (x->pp_components) { modules_pp++; total_pp += x->pp_components; }
        if (x->max_location > max_location) max_location = x->max_location;
        uint32_t bi = s.output_builtin_mask;
        if (bi & SPVR_BUILTIN_POINT_SIZE) x->c |= C_POINT_SIZE;
        if (bi & SPVR_BUILTIN_CLIP_DISTANCE) x->c |= C_CLIP;
        if (bi & SPVR_BUILTIN_CULL_DISTANCE) x->c |= C_CULL;
        if (bi & SPVR_BUILTIN_PRIMITIVE_ID) x->c |= C_PRIMITIVE_ID;
        if (bi & SPVR_BUILTIN_LAYER) x->c |= C_LAYER;
        if (bi & SPVR_BUILTIN_VIEWPORT_INDEX) x->c |= C_VIEWPORT;
        if (bi & SPVR_BUILTIN_CULL_PRIMITIVE) x->c |= C_CULL_PRIMITIVE;
        if (bi & SPVR_BUILTIN_PRIMITIVE_SHADING_RATE) x->c |= C_SHADING_RATE;
        if (bi & SPVR_BUILTIN_OTHER) x->c |= C_BUILTIN_OTHER;
        if (s.output_size > 32 * 1024) x->c |= C_OUTPUT_32K;
        if (s.output_size > 8 * 1024) x->c |= C_OUTPUT_8K;
        if (s.output_components > 4) x->c |= C_COMPONENTS4;
        if ((uint64_t)s.task_payload_size + s.output_size > 8 * 1024) x->c |= C_PAYLOAD_OUTPUT_8K;
        if (s.shared_size > 24 * 1024) x->c |= C_SHARED_24K;
        if ((uint64_t)s.task_payload_size + s.shared_size > 24 * 1024) x->c |= C_PAYLOAD_SHARED_24K;
        if (s.output_size > max_output) { max_output = s.output_size; memcpy(max_output_key, key, 17); }
        if (s.shared_size > max_shared) max_shared = s.shared_size;
        if (s.task_payload_size > max_payload) max_payload = s.task_payload_size;
        if (s.output_components > max_components) max_components = s.output_components;
        if (!strcmp(key, "554cf8c05126a133")) { focus_ms = s; have_ms = 1; }
        free(b);
    }
    check(nts == want_ts && nms == want_ms, "coverage stage counts");
    printf("SUMMARY\ttask=%d\tmesh=%d\ttopology_none=%d\tpoints=%d\tlines=%d\ttriangles=%d\tmodules_pv=%d\tmodules_pp=%d\tpv_components=%llu\tpp_components=%llu\toutputs_16=%d\toutputs_32=%d\toutputs_64=%d\tmax_location=%u\tmax_shared=%u\tmax_payload=%u\tmax_output=%u:%s\tmax_components=%u\n",
        nts, nms, top[0], top[1], top[2], top[3], modules_pv, modules_pp,
        (unsigned long long)total_pv, (unsigned long long)total_pp, widths16, widths32, widths64,
        max_location, max_shared, max_payload, max_output, max_output_key, max_components);
    printf("KINDS\tf=%u\ti=%u\tu=%u\tother=%u\n", kind_f, kind_i, kind_u, kind_other);
    for (unsigned i = 0; i < 128; i++) if (pv_decl[i] || pp_decl[i])
        printf("LOCATION\t%u\tpv_decl=%u\tpv_components=%u\tpp_decl=%u\tpp_components=%u\n",
            i, pv_decl[i], pv_comp[i], pp_decl[i], pp_comp[i]);
    class_mask_line("abi_v3_shape_unsupported", rr, nr,
        C_TOPOLOGY | C_BITS16 | C_BITS64 | C_BITS_OTHER | C_LOCATION | C_KIND |
        C_POINT_SIZE | C_CLIP | C_CULL | C_PRIMITIVE_ID | C_LAYER | C_VIEWPORT |
        C_CULL_PRIMITIVE | C_SHADING_RATE | C_BUILTIN_OTHER);
    class_line("abi_non_triangle", rr, nr, C_TOPOLOGY);
    class_line("abi_output_16bit", rr, nr, C_BITS16);
    class_line("abi_output_64bit", rr, nr, C_BITS64);
    class_line("abi_output_other_width", rr, nr, C_BITS_OTHER);
    class_line("abi_location_over_31", rr, nr, C_LOCATION);
    class_line("abi_bool_or_unknown_kind", rr, nr, C_KIND);
    class_line("abi_PointSize", rr, nr, C_POINT_SIZE);
    class_line("abi_ClipDistance", rr, nr, C_CLIP);
    class_line("abi_CullDistance", rr, nr, C_CULL);
    class_line("abi_PrimitiveId", rr, nr, C_PRIMITIVE_ID);
    class_line("abi_Layer", rr, nr, C_LAYER);
    class_line("abi_ViewportIndex", rr, nr, C_VIEWPORT);
    class_line("abi_CullPrimitiveEXT", rr, nr, C_CULL_PRIMITIVE);
    class_line("abi_PrimitiveShadingRate", rr, nr, C_SHADING_RATE);
    class_line("abi_other_builtin", rr, nr, C_BUILTIN_OTHER);
    class_line("required_output_over_32KiB", rr, nr, C_OUTPUT_32K);
    class_line("g13_output_over_8KiB", rr, nr, C_OUTPUT_8K);
    class_line("g13_user_components_over_4", rr, nr, C_COMPONENTS4);
    class_line("g13_payload_plus_output_over_8KiB", rr, nr, C_PAYLOAD_OUTPUT_8K);
    class_line("g13_shared_over_24KiB", rr, nr, C_SHARED_24K);
    class_line("g13_payload_plus_shared_over_24KiB", rr, nr, C_PAYLOAD_SHARED_24K);
    class_line("experimental_zero_user_output_metadata", rr, nr, C_NO_USER);
    builtin_line("Position", rr, nr, SPVR_BUILTIN_POSITION, 0);
    builtin_line("PointSize", rr, nr, SPVR_BUILTIN_POINT_SIZE, 1);
    builtin_line("ClipDistance", rr, nr, SPVR_BUILTIN_CLIP_DISTANCE, 1);
    builtin_line("CullDistance", rr, nr, SPVR_BUILTIN_CULL_DISTANCE, 1);
    builtin_line("PrimitiveId", rr, nr, SPVR_BUILTIN_PRIMITIVE_ID, 1);
    builtin_line("Layer", rr, nr, SPVR_BUILTIN_LAYER, 1);
    builtin_line("ViewportIndex", rr, nr, SPVR_BUILTIN_VIEWPORT_INDEX, 1);
    builtin_line("CullPrimitiveEXT", rr, nr, SPVR_BUILTIN_CULL_PRIMITIVE, 1);
    builtin_line("PrimitiveShadingRate", rr, nr, SPVR_BUILTIN_PRIMITIVE_SHADING_RATE, 1);
    printf("PER_PRIMITIVE\t%d", modules_pp);
    for (int i = 0; i < nr; i++) if (rr[i].pp_components) printf("\t%s", rr[i].key);
    putchar('\n');
    for (int i = 0; i < nr; i++) if (rr[i].pp_components) detail("per_primitive", rr[i].key, &rr[i].r);
    check(have_ts && have_ms, "focused title task/mesh pair present");
    if (have_ts) detail("task", "efaa4fa6a80ba719", &focus_ts);
    if (have_ms) detail("mesh", "554cf8c05126a133", &focus_ms);
    free(rr);
}

int main(int argc, char **argv) {
    selftest(); coverage(argc, argv);
    printf("t_vk_mesh_coverage: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
