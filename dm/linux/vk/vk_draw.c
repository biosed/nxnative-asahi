// Linux build: the draw path of the Vulkan replay (step 3.4 of the backend, LINUX.md). Step 3.4b: shaders and their resources.
//   - the bound state, per NVN stage, as the replay's walker records it: uniform and storage buffers, texture / image / sampler handles, the descriptor pools' bases, the program
//   - a program's stages as shader objects (VK_EXT_shader_object), made the first time from <key>.vk.spv in DM_VK_SPV (dm/gfx/tools/dmsh_spv.py), fitted to the device
//     (spvr_fit), with one set layout for every set: dmsh_spv.py moved every shader's resources to fixed bindings, a set per stage
//   - per draw, a descriptor set for each of the program's stages, written from the shader's binding list (<key>.vkb): the support buffer, the shader's c1, the driver
//     banks c0 / c2 (filled from the bound state), uniform and storage buffers (vkm_lookup), textures with views of the shader's dimension and their samplers
//     (cached by their parameters). Whatever is missing gets a dummy, so a shader never reads an unwritten descriptor; the misses are counted (DM_VK_DRAWLOG=1)
// The draws themselves (vertex input, dynamic state, render passes) are step 3.4c
#include "dm.h"
#include "vk_dev.h"
#include "vk_fmt.h"
#include "vk_mem.h"
#include "vk_replay.h"
#include "spv_reflect.h"
#include "vk_draw.h"
#include "vk_skipcs.h"
#include "vk_movie.h"
#include "vk_gmem_trace.h"
#include "vk_missing_log.h"
#include "vk_stage.h"
#include "vk_shader_resolve.h"
#include "vk_mesh_trace.h"
#include "vk_omap.h"
#include "vk_compile_watch.h"
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/resource.h>
#include <dirent.h>
#include <sched.h>

#define NST VKX_STATE_STAGE_COUNT
typedef struct { uint64_t addr, size; } bbuf;
// The bound state is per NVN queue (a queue keeps its bindings and raster state across submissions; Ocarina of Time submits on several queues): one set per queue
// serial, selected at each submission by vkx_queue_select (mtl_queue_activate); slot 0 until a queue is named. Reading S / D anywhere reads the selected queue's
#define VKX_NQ 8
typedef struct {
    bbuf ubo[NST][16], ssbo[NST][16];
    uint64_t tex[NST][256], samp[NST][128], img[NST][32];
    int tex_base, samp_base;
    const prog_t *prog[NST];
    // SPH fragment output map for the program prepared by the current draw:
    // four RGBA bits per colour target.  Maxwell preserves target components
    // outside this map even when the NVN channel mask enables them.
    uint32_t frag_out_map;
} vkx_bind_st;   // prog: the program bound to each stage (vkx_bind_program)
static vkx_bind_st g_bind[VKX_NQ]; static vkx_bind_st *g_bindp = &g_bind[0];
#define S (*g_bindp)

// ---- shaders ----
#define NBIND_MAX 96
// a binding's kind and sampler kind (the .vkb's names), as numbers for the per-draw descriptor writes
enum { BK_OTHER, BK_SUPPORT, BK_C1, BK_CBUF, BK_DRVCB, BK_SBUF, BK_UBO_PTR, BK_TEX, BK_TEX_UBO, BK_GMEM_MAP, BK_GMEM_LOG, BK_MESH_FIRST };
enum { BS_OTHER, BS_SAMP_TEX, BS_SAMP, BS_SAMP_ID, BS_SAMP_UBO };
typedef struct { uint16_t b; char desc; char kind[12]; int reg, off; char skind[12]; int sreg, soff; uint8_t dim, arr, ms, dep; uint32_t cnt; uint8_t k, sk; } vkbind;
typedef struct vks {
    char key[17]; uint64_t k64; struct vks *next; int stage, ok;
    VkShaderEXT sh; VkShaderStageFlagBits vstage;
    VkShaderEXT sh_gl; uint32_t *spv; size_t nspv; int gl_none;   // the GL-depth variant (spvr_gl_depth), made the first time it is needed from the fitted SPIR-V kept here
    int nb; vkbind bd[NBIND_MAX];
    VkBuffer c1; VkDeviceMemory c1m; uint32_t c1_len;
    uint32_t frag_out_map;       // pixel SPH word 18; 4 RGBA bits per target
    vkx_mesh_stage_meta mesh_meta;   // actual stage and native task/mesh job limits; host slot remains separate
} vks;
typedef struct {
    char key[17];
    int kind;
    vkbind input, output;
} vkx_movie_shader;
#define VKX_MOVIE_SHADERS_MAX 64
static vkx_movie_shader g_movie_shader[VKX_MOVIE_SHADERS_MAX];
static int g_nmovie_shader;
static vks *g_sh[1024]; static pthread_mutex_t g_shmu = PTHREAD_MUTEX_INITIALIZER;
enum { VKX_BINDLESS_SET = 5, VKX_FEEDBACK_SET = 6, VKX_BINDLESS_2D = 0, VKX_BINDLESS_2DA = 1, VKX_BINDLESS_DEPTH = 2, VKX_BINDLESS_SAMP = 3 };
static VkDescriptorSetLayout g_dsl, g_bindless_dsl; static VkDescriptorPool g_bindless_pool; static VkDescriptorSet g_bindless_set;
static VkDescriptorSetLayout g_feedback_dsl; static VkDescriptorPool g_feedback_pool; static VkDescriptorSet g_feedback_set[CB_RING];
static VkBuffer g_feedback_buf[CB_RING]; static VkDeviceMemory g_feedback_mem[CB_RING]; static uint32_t *g_feedback_map[CB_RING];
static uint8_t g_bindless_real[VKR_BINDLESS_TEXTURES];
static uint8_t g_bindless_seed[VKR_BINDLESS_TEXTURES];
static int g_feedback_enabled;
static VkPipelineLayout g_pl_gfx, g_pl_cs;
static int g_trace;   // the replay's pass log is on (DM_VK_PASSLOG): each draw is logged with its shaders and textures
static int g_trace_pre_stage, g_trace_pre_gl, g_trace_pre_fallback;
void vkx_trace(int on) { g_trace = on; }
static int g_depth_zero;   // NVN depth mode: 1 near = 0 (as Vulkan), 0 near = -w (GL, NVN's default): the last stage before rasterization then takes its GL-depth variant
static int g_init = -1, g_gmem_trace = -1; static char g_spv[1024], g_mesh_spv[1024]; static uint32_t g_per_set[5];   // descriptors of a set by type: uniform, storage, combined, storage image, texel buffer
static uint64_t g_st_load, g_st_fail, g_st_nokey, g_st_draws, g_st_sets, g_miss[16];
enum { M_UBO, M_TEX, M_TEXREG, M_SAMP, M_SBUF, M_IMG, M_TBUF, M_UNALIGNED, M_NOSHADER, M_OTHER };
static const char *const MISS[] = {"uniform", "texture", "texture ID", "sampler", "storage", "image", "texel buffer", "unaligned", "shader", "other"};

static void key_hex(const uint8_t key[8], char out[17]) { for (int i = 0; i < 8; i++) snprintf(out + 2 * i, 3, "%02x", key[i]); }

// Prefer stage identity retained with the program at ProgramSetShaders.  A
// legacy capture has zero stage bytes, so fall back to sidecar agreement; that
// fallback intentionally cannot guess a missing translation.
static int bound_stage_key(const prog_t *p, int wanted, char out[17]) {
    if (!p) return 0;
    int stage[6], n = 0; char key_text[6][17];
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {
        uint8_t key[8]; int actual = -1;
        if (!prog_key_info(p, i, key, &actual)) continue;
        key_hex(key, key_text[n]); stage[n++] = actual;
    }
    if (vkx_mesh_trace_pick(stage, key_text, (unsigned)n, wanted, out)) return 1;
    for (int i = 0; i < n; i++) if (stage[i] < 0) {
        vkx_shader_resolved r; if (vkx_shader_resolve(g_spv, g_mesh_spv, key_text[i], wanted, &r)) { memcpy(out, key_text[i], 17); return 1; }
    }
    return 0;
}

void vkx_mesh_draw_telemetry(const p_draw *d) {
    static int enabled = -1; static _Atomic unsigned seen;
    if (enabled < 0) { const char *e = getenv("DM_VK_MESH_LOG"); enabled = e && atoi(e) != 0; }
    if (!enabled || !d || !vkx_mesh_trace_take(&seen, d->ind != 0)) return;
    char ts[17] = "-", ms[17] = "-", ps[17] = "-", line[320];
    bound_stage_key(S.prog[3], VKX_STAGE_TS, ts);
    bound_stage_key(S.prog[4], VKX_STAGE_MS, ms);
    bound_stage_key(S.prog[1], VKX_STAGE_PS, ps);
    vkx_mesh_trace_event e = {.task_key = ts, .mesh_key = ms, .fragment_key = ps,
        .indirect = d->ind != 0, .first = (uint32_t)d->first, .count = (uint32_t)d->count, .address = d->ind};
    if (vkx_mesh_trace_format(line, sizeof line, &e) >= 0) LOG("%s", line);
}

static uint32_t hexbyte(const char *p) { unsigned v = 0; sscanf(p, "%2x", &v); return v; }
static int read_file(const char *path, uint8_t **out, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    *out = malloc((size_t)len + 1); *n = fread(*out, 1, (size_t)len, f); (*out)[*n] = 0; fclose(f); return 1;
}
static void buffer_with(const void *data, size_t n, VkBuffer *b, VkDeviceMemory *m) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n ? n : 16, .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
    vkCreateBuffer(g_vkd.dev, &bi, NULL, b);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, *b, &mr);
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = vkr_pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    vkAllocateMemory(g_vkd.dev, &ma, NULL, m); vkBindBufferMemory(g_vkd.dev, *b, *m, 0);
    void *p; vkMapMemory(g_vkd.dev, *m, 0, VK_WHOLE_SIZE, 0, &p); memset(p, 0, mr.size); if (data && n) memcpy(p, data, n); vkUnmapMemory(g_vkd.dev, *m);
}
static int feedback_init(void) {
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
    if (vkCreateDescriptorSetLayout(g_vkd.dev, &li, NULL, &g_feedback_dsl) != VK_SUCCESS) return 0;
    if (!g_feedback_enabled) return 1;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, CB_RING};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = CB_RING, .poolSizeCount = 1, .pPoolSizes = &ps};
    if (vkCreateDescriptorPool(g_vkd.dev, &pi, NULL, &g_feedback_pool) != VK_SUCCESS) return 0;
    VkDescriptorSetLayout layouts[CB_RING]; for (int i = 0; i < CB_RING; i++) layouts[i] = g_feedback_dsl;
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g_feedback_pool,
        .descriptorSetCount = CB_RING, .pSetLayouts = layouts};
    if (vkAllocateDescriptorSets(g_vkd.dev, &ai, g_feedback_set) != VK_SUCCESS) return 0;
    const VkDeviceSize bytes = (VKR_BINDLESS_TEXTURES / 32u) * sizeof(uint32_t);
    for (int i = 0; i < CB_RING; i++) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        if (vkCreateBuffer(g_vkd.dev, &bi, NULL, &g_feedback_buf[i]) != VK_SUCCESS) return 0;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkd.dev, g_feedback_buf[i], &mr);
        uint32_t mt = vkr_pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt};
        if (mt == UINT32_MAX || vkAllocateMemory(g_vkd.dev, &ma, NULL, &g_feedback_mem[i]) != VK_SUCCESS ||
            vkBindBufferMemory(g_vkd.dev, g_feedback_buf[i], g_feedback_mem[i], 0) != VK_SUCCESS ||
            vkMapMemory(g_vkd.dev, g_feedback_mem[i], 0, bytes, 0, (void **)&g_feedback_map[i]) != VK_SUCCESS) return 0;
        memset(g_feedback_map[i], 0, (size_t)bytes);
        VkDescriptorBufferInfo db = {g_feedback_buf[i], 0, bytes};
        VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_feedback_set[i], .dstBinding = 0,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db};
        vkUpdateDescriptorSets(g_vkd.dev, 1, &wr, 0, NULL);
    }
    LOG("vulkan draws: bindless GPU-access feedback enabled: %d timeline-safe %llu KiB bitmaps", CB_RING,
        (unsigned long long)(bytes >> 10));
    return 1;
}
static VkShaderStageFlagBits vstage_of(int st) {
    return vkx_stage_vk_bit(st);
}
static void parse_vkb(vks *s, const char *txt);
static int bindless_vkb(const vkbind *b);
static int movie_named(const char key[17]) {
    const char *p = DM_GAME_MOVIE_NET_KEYS;
    while (*p) {
        if (!strncmp(p, key, 16) && (p[16] == ',' || !p[16])) return 1;
        p = strchr(p, ','); if (!p) break; p++;
    }
    return 0;
}
static void movie_reflect(const char key[17], const vks *s) {
    if (!movie_named(key)) return;
    int nt = 0, ni = 0, ns = 0; const vkbind *input = NULL, *output = NULL;
    for (int i = 0; i < s->nb; i++) {
        const vkbind *b = &s->bd[i];
        if (b->desc == 't') { nt++; input = b; }
        else if (b->desc == 'i') { ni++; output = b; }
        else if (b->k == BK_SBUF || b->k == BK_UBO_PTR) ns++;
    }
    int kind = vkx_movie_classify(1, nt, ni, ns);
    if (kind == VKX_MOVIE_NONE) return;
    if (g_nmovie_shader == VKX_MOVIE_SHADERS_MAX) {
        static int warned; if (!warned++) LOG("[RED] vulkan draws: movie network reflection exceeds its %d-shader bound", VKX_MOVIE_SHADERS_MAX);
        return;
    }
    vkx_movie_shader *m = &g_movie_shader[g_nmovie_shader++];
    snprintf(m->key, sizeof m->key, "%s", key); m->kind = kind;
    if (input) m->input = *input;
    if (output) m->output = *output;
}
static void shc_init(uint64_t layout_hash); static void precompile_start(void);
static char (*g_keys)[17]; static int g_nkeys;   // every standard and native-mesh shader key (draw_init)
static void cw_start(void);   // the compile watcher thread (below sh_object)
static int draw_init(void) {
    cw_start();
    if (g_init >= 0) return g_init;
    g_feedback_enabled = getenv("DM_VK_BINDLESS_FEEDBACK") && atoi(getenv("DM_VK_BINDLESS_FEEDBACK"));
    const char *seed = getenv("DM_VK_BINDLESS_SEED");
    unsigned nseed = 0;
    while (seed && *seed) {
        char *end = NULL; unsigned long id = strtoul(seed, &end, 10);
        if (end != seed) {
            if (id > 0 && id < VKR_BINDLESS_TEXTURES && !g_bindless_seed[id]) {
                g_bindless_seed[id] = 1; nseed++;
            }
            seed = end;
        } else seed++;
        while (*seed == ',' || *seed == ';' || *seed == ' ' || *seed == '\t') seed++;
    }
    if (nseed) LOG("vulkan draws: %u bindless texture IDs seeded for first-use-safe materialisation", nseed);
    g_gmem_trace = getenv("DM_VK_GMEM_TRACE") && atoi(getenv("DM_VK_GMEM_TRACE"));
    const char *d = getenv("DM_VK_SPV");
    if (!vkd_ok() || !g_vkd.shader_object || !d || !*d) {
        LOG("%svulkan draws: %s", d && *d ? "[RED] " : "", !vkd_ok() ? "no device" : !g_vkd.shader_object ? "no VK_EXT_shader_object" : "no shaders (set DM_VK_SPV to dmsh_spv.py's spv-<chain> directory)");
        return g_init = 0;
    }
    snprintf(g_spv, sizeof g_spv, "%s", d);
    const char *md = getenv("DMSH_VK_MESH_EXT_DIR");
    if (md && *md) snprintf(g_mesh_spv, sizeof g_mesh_spv, "%s", md);
    if (g_gmem_trace) LOG("vulkan draws: DM_VK_GMEM_TRACE=1: exact raw-global guest pages will invalidate only overlapping uploaded images");
    // One set layout for every stage's set: the union of the bindings the game's shaders use (dmsh_spv.py's canonical bindings, listed in each <key>.vkb), each partially
    // bound. A set takes pool space for every binding of its layout, so only the used ones are in it
    static uint32_t cnt[1024]; static char typ[1024];
    const char *dirs[2] = {g_spv, g_mesh_spv}; int nfiles = 0, nmesh = 0;
    for (int di = 0; di < 2; di++) {
        if (!dirs[di][0]) continue;
        DIR *dir = opendir(dirs[di]);
        if (!dir) {
            LOG("[RED] vulkan draws: %s=%s is not a directory", di ? "DMSH_VK_MESH_EXT_DIR" : "DM_VK_SPV", dirs[di]);
            if (!di) return g_init = 0;
            continue;
        }
        for (struct dirent *de; (de = readdir(dir));) {
            size_t l = strlen(de->d_name); if (l != 20 || strcmp(de->d_name + 16, ".vkb")) continue;
            char key[17]; memcpy(key, de->d_name, 16); key[16] = 0;
            vkx_shader_resolved rr;
            if (!vkx_shader_resolve(g_spv, g_mesh_spv, key, -1, &rr) || rr.mesh_ext != di) continue;
            uint8_t *txt; size_t nt; if (!read_file(rr.vkb, &txt, &nt)) continue;
            static vks tmp; tmp.nb = 0; parse_vkb(&tmp, (const char *)txt); free(txt); nfiles++; if (di) nmesh++;
            if (!di) movie_reflect(key, &tmp);
            if (!(g_nkeys & 1023)) g_keys = realloc(g_keys, (g_nkeys + 1024) * sizeof *g_keys);
            memcpy(g_keys[g_nkeys], key, 17); g_nkeys++;
            for (int i = 0; i < tmp.nb; i++) { const vkbind *b = &tmp.bd[i]; if (b->b >= 1024 || bindless_vkb(b)) continue;
                if (typ[b->b] && typ[b->b] != b->desc) LOG("[RED] vulkan draws: binding %u is %c in one shader and %c in another", b->b, typ[b->b], b->desc);
                typ[b->b] = b->desc; uint32_t c = b->cnt ? b->cnt : 1; if (c > cnt[b->b]) cnt[b->b] = c; }
        }
        closedir(dir);
    }
    VkDescriptorSetLayoutBinding b[1024]; VkDescriptorBindingFlags f[1024]; uint32_t n = 0;
    for (uint32_t i = 0; i < 1024; i++) {
        if (!typ[i]) continue;
        VkDescriptorType t = typ[i] == 'u' ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : typ[i] == 's' ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : typ[i] == 't' ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                             typ[i] == 'i' ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : typ[i] == 'b' ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_SAMPLER;
        b[n] = (VkDescriptorSetLayoutBinding){i, t, cnt[i], VK_SHADER_STAGE_ALL, NULL}; f[n++] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        for (int k = 0; k < 5; k++) if ((int)t == (int[]){VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER}[k]) g_per_set[k] += cnt[i];
    }
    if (nfiles == nmesh) { LOG("[RED] vulkan draws: no validated standard <key>.vkb in %s (run dm/gfx/tools/dmsh_spv.py)", g_spv); return g_init = 0; }
    { uint64_t lh = 1469598103934665603ull; for (uint32_t i = 0; i < n; i++) { lh = (lh ^ b[i].binding) * 1099511628211ull; lh = (lh ^ (uint64_t)b[i].descriptorType) * 1099511628211ull; lh = (lh ^ b[i].descriptorCount) * 1099511628211ull; }
      lh = (lh ^ VKX_BINDLESS_SET) * 1099511628211ull; lh = (lh ^ VKR_BINDLESS_TEXTURES) * 1099511628211ull; lh = (lh ^ VKR_BINDLESS_SAMPLERS) * 1099511628211ull;
      /* Keep the pre-feedback six-set ABI and its shader-cache identity when
       * feedback is off.  Merely appending an unused seventh set changes the
       * shader-object binary contract on some drivers. */
      if (g_feedback_enabled) lh = (lh ^ 1ull) * 1099511628211ull;
      shc_init(lh); }   // shader binaries depend on the layout they were made with
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .bindingCount = n, .pBindingFlags = f};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi, .bindingCount = n, .pBindings = b};
    if (vkCreateDescriptorSetLayout(g_vkd.dev, &li, NULL, &g_dsl)) { LOG("[RED] vulkan draws: no descriptor set layout"); return g_init = 0; }
    VkDescriptorSetLayoutBinding bb[4] = {
        {VKX_BINDLESS_2D, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKR_BINDLESS_TEXTURES, VK_SHADER_STAGE_ALL, NULL},
        {VKX_BINDLESS_2DA, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKR_BINDLESS_TEXTURES, VK_SHADER_STAGE_ALL, NULL},
        {VKX_BINDLESS_DEPTH, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKR_BINDLESS_TEXTURES, VK_SHADER_STAGE_ALL, NULL},
        {VKX_BINDLESS_SAMP, VK_DESCRIPTOR_TYPE_SAMPLER, VKR_BINDLESS_SAMPLERS, VK_SHADER_STAGE_ALL, NULL},
    };
    VkDescriptorBindingFlags bf = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    VkDescriptorBindingFlags bbf[4] = {bf, bf, bf, bf};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .bindingCount = 4, .pBindingFlags = bbf};
    VkDescriptorSetLayoutCreateInfo bli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &bfi, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, .bindingCount = 4, .pBindings = bb};
    VkDescriptorPoolSize bps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3u * VKR_BINDLESS_TEXTURES}, {VK_DESCRIPTOR_TYPE_SAMPLER, VKR_BINDLESS_SAMPLERS}};
    VkDescriptorPoolCreateInfo bpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = bps};
    VkResult br;
    if ((br = vkCreateDescriptorSetLayout(g_vkd.dev, &bli, NULL, &g_bindless_dsl)) ||
        (br = vkCreateDescriptorPool(g_vkd.dev, &bpi, NULL, &g_bindless_pool))) {
        LOG("[RED] vulkan draws: no bindless pool layout (%d)", (int)br); return g_init = 0;
    }
    VkDescriptorSetAllocateInfo bai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g_bindless_pool, .descriptorSetCount = 1, .pSetLayouts = &g_bindless_dsl};
    if ((br = vkAllocateDescriptorSets(g_vkd.dev, &bai, &g_bindless_set))) { LOG("[RED] vulkan draws: no bindless pool set (%d)", (int)br); return g_init = 0; }
    if (!feedback_init()) { LOG("[RED] vulkan draws: no bindless feedback layout/buffers"); return g_init = 0; }
    VkDescriptorSetLayout sets[7] = {g_dsl, g_dsl, g_dsl, g_dsl, g_dsl, g_bindless_dsl, g_feedback_dsl};
    uint32_t nsets = g_feedback_enabled ? 7u : 6u;
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = nsets, .pSetLayouts = sets};
    VkPipelineLayoutCreateInfo pc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = nsets, .pSetLayouts = sets};
    if (vkCreatePipelineLayout(g_vkd.dev, &pi, NULL, &g_pl_gfx) || vkCreatePipelineLayout(g_vkd.dev, &pc, NULL, &g_pl_cs)) { LOG("[RED] vulkan draws: no pipeline layouts"); return g_init = 0; }
    LOG("vulkan draws: shaders from %s (%d standard)%s%s%s, one set layout of %u bindings for every stage (%u uniform, %u storage, %u textures, %u images, %u texel buffers per set); bindless set %d has 3x%u sampled images + %u samplers", g_spv, nfiles - nmesh,
        nmesh ? " and " : "", nmesh ? g_mesh_spv : "", nmesh ? " (native task/mesh)" : "", n,
        g_per_set[0], g_per_set[1], g_per_set[2], g_per_set[3], g_per_set[4], VKX_BINDLESS_SET, VKR_BINDLESS_TEXTURES, VKR_BINDLESS_SAMPLERS);
    precompile_start();   // after the layouts: a shader is compiled against g_dsl (started before it, a compiler thread passed a null layout and Mesa crashed in lookup_ycbcr_conversion)
    return g_init = 1;
}
static void parse_vkb(vks *s, const char *txt) {
    for (const char *p = txt; *p;) {
        const char *e = strchr(p, '\n'); size_t len = e ? (size_t)(e - p) : strlen(p);
        if (*p != '#' && len && s->nb < NBIND_MAX) {
            char line[512]; snprintf(line, sizeof line, "%.*s", (int)(len < 511 ? len : 511), p);
            char *f[13] = {0}; int nf = 0; for (char *q = line, *t; nf < 13 && (t = strsep(&q, "\t")); ) f[nf++] = t;
            if (nf == 13) {
                vkbind *b = &s->bd[s->nb++];
                b->b = (uint16_t)atoi(f[0]); b->desc = f[1][0]; snprintf(b->kind, sizeof b->kind, "%s", f[2]); b->reg = *f[3] ? atoi(f[3]) : -1; b->off = *f[4] ? atoi(f[4]) : 0;
                snprintf(b->skind, sizeof b->skind, "%s", f[5]); b->sreg = *f[6] ? atoi(f[6]) : -1; b->soff = *f[7] ? atoi(f[7]) : 0;
                b->dim = (uint8_t)atoi(f[8]); b->arr = (uint8_t)atoi(f[9]); b->ms = (uint8_t)atoi(f[10]); b->dep = (uint8_t)atoi(f[11]); b->cnt = (uint32_t)strtoul(f[12], NULL, 10);
                static const char *const kn[] = {"", "support", "c1", "cbuf", "drvcb", "sbuf", "ubo_ptr", "tex", "tex_ubo", "gmem_map", "gmem_log"}, *const sn[] = {"", "samp_tex", "samp", "samp_id", "samp_ubo"};
                b->k = BK_OTHER; for (int i = 1; i < 11; i++) if (!strcmp(b->kind, kn[i])) b->k = (uint8_t)i;
                if (vkx_mesh_first_descriptor(b->b, b->desc, b->kind)) b->k = BK_MESH_FIRST;
                b->sk = BS_OTHER; for (int i = 1; i < 5; i++) if (!strcmp(b->skind, sn[i])) b->sk = (uint8_t)i;
            }
        }
        if (!e) break; p = e + 1;
    }
}
static int bindless_vkb(const vkbind *b) {
    return (b->desc == 't' && b->cnt == 1048576 && b->b >= 564 && b->b <= 566) ||
           (b->desc == 'p' && b->cnt == 4096 && b->b == 660);
}
// The i-th shader of a program, made the first time (its key from prog_key, which takes the index in the program's list, not a stage: a compute program is one
// shader at index 0; the shader's stage is the one its translation names, in its .vkb header). NULL when there is none or it cannot be made, logged once per key
// Shader objects made and the time they took (the long-frame log's "new pipelines + shader libraries": shaders as libraries, GL-depth variants as pipelines)
static _Atomic uint64_t g_mk_sh, g_mk_sh_ns, g_mk_var, g_mk_var_ns;
static uint64_t mk_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
// ---- compile visibility (vk_compile_watch.h) ----
// A first-use compile of a large kernel holds the replay thread for the driver's whole compile (Ocarina's NDVE d15370a382630d4e: 57 s on Honeykrisp), which the
// log showed only as a GPU heartbeat with nothing submitted. The table below holds every SPIR-V compile in flight; the watcher thread prints those older than 2 s
// every 5 s, the heartbeat appends them (vkx_compile_text), and a finished compile of 1 s or more is logged by default. DM_VK_COMPILE_WATCH=0 leaves the thread out.
static vkx_cw_table g_cw;
enum { CW_REPORT_NS = 2000000000, CW_PERIOD_NS = 5000000000 };
int vkx_compile_text(char *out, size_t n) { return vkx_cw_text(&g_cw, mk_now(), CW_REPORT_NS, out, n); }
static void *cw_thread(void *arg) {
    (void)arg; uint64_t last = 0; char line[160];
    for (;;) {
        usleep(1000000);
        uint64_t now = mk_now();
        if (now - last < CW_PERIOD_NS) continue;
        if (vkx_cw_text(&g_cw, now, CW_REPORT_NS, line, sizeof line)) { last = now; LOG("vulkan shader compile: %s", line); }
    }
    return NULL;
}
static void cw_start(void) {   // from draw_init, once, before any thread compiles
    static _Atomic int started;
    if (atomic_exchange(&started, 1)) return;
    const char *e = getenv("DM_VK_COMPILE_WATCH"); if (e && atoi(e) == 0) return;
    pthread_t t; if (!pthread_create(&t, NULL, cw_thread, NULL)) pthread_detach(t); else LOG("[RED] vulkan draws: no compile watcher thread (long compiles are logged only when they finish)");
}
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = atomic_load(&g_mk_sh); *pso = atomic_load(&g_mk_var); }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = atomic_load(&g_mk_sh_ns); *pso_ns = atomic_load(&g_mk_var_ns); }
static int drawlog(void) { static int v = -1; if (v < 0) v = getenv("DM_VK_DRAWLOG") != NULL; return v; }   // DM_VK_DRAWLOG: draw counters at powers of two
// DM_VK_MISSING_LOG=<file>: append each absent translation key and its program
// index once ("<16 hex>\t<index>\n"). The fixed key set prevents a bad guest or
// repeated draws from turning the focused harvest into an unbounded log.
static vkx_missing_log_set g_missing_log_set;
static int g_missing_log_fd = -2, g_missing_log_full;
static void missing_log(const char key[17], uint64_t k64, int idx) {
    const char *path = getenv("DM_VK_MISSING_LOG");
    if (!path || !*path) return;
    int fresh = vkx_missing_log_once(&g_missing_log_set, k64);
    if (!fresh) return;
    if (fresh < 0) {
        if (!g_missing_log_full++) LOG("[RED] vulkan draws: DM_VK_MISSING_LOG reached its %u-key bound; no more keys will be appended", VKX_MISSING_LOG_MAX);
        return;
    }
    if (g_missing_log_fd == -2) {
        g_missing_log_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (g_missing_log_fd < 0) LOG("[RED] vulkan draws: cannot append DM_VK_MISSING_LOG=%s: %s", path, strerror(errno));
        else LOG("vulkan draws: unique missing shader keys append to %s (bounded at %u)", path, VKX_MISSING_LOG_MAX);
    }
    if (g_missing_log_fd < 0) return;
    char line[40]; int n = vkx_missing_log_line(line, sizeof line, key, idx);
    if (n < 0) return;
    for (int off = 0; off < n;) {
        ssize_t w = write(g_missing_log_fd, line + off, (size_t)(n - off));
        if (w > 0) off += (int)w;
        else if (w < 0 && errno == EINTR) continue;
        else { LOG("[RED] vulkan draws: DM_VK_MISSING_LOG write failed: %s", strerror(errno)); close(g_missing_log_fd); g_missing_log_fd = -1; break; }
    }
}
// ---- shader objects and their disk cache ----
// A shader's SPIR-V as the device takes it (fitted, outputs initialized), and its stage from the .vkb; with s, the binding table and c1 go into s. NULL when it is not there
static uint32_t *sh_code(const char *key, int *st_out, size_t *nw, vks *s, unsigned *ch_out) {
    char path[1200]; uint8_t *code = NULL, *vkb = NULL, *job = NULL; size_t ncode = 0, nvkb = 0, njob = 0;
    vkx_shader_resolved rr; int have = vkx_shader_resolve(g_spv, g_mesh_spv, key, -1, &rr);
    if (have) have = read_file(rr.spv, &code, &ncode) && read_file(rr.vkb, &vkb, &nvkb);
    { const char *ov = getenv("DM_VK_SHOVERRIDE"), *m = ov ? strstr(ov, key) : NULL;   // diagnostic: DM_VK_SHOVERRIDE=<key>=<spv file>[,...] replaces that shader's code
      if (have && m && m[16] == '=') { char f[1024]; size_t n = strcspn(m + 17, ","); if (n >= sizeof f) n = sizeof f - 1; memcpy(f, m + 17, n); f[n] = 0;
          uint8_t *oc = NULL; size_t on = 0; if (read_file(f, &oc, &on)) { free(code); code = oc; ncode = on; if (s) LOG("vulkan draws: shader %s replaced by %s", key, f); } } }
    int st = have ? rr.stage : -1;
    if (!have || st < 0) { free(code); free(vkb); free(job); return NULL; }
    if (s) {
        // Older packages have no Vulkan reflection sidecar.  Preserve their
        // historical all-components behaviour; a present pixel reflection
        // may legitimately contain zero for a depth-only shader.
        s->frag_out_map = 0xffffffffu;
        if (st == VKX_STAGE_PS) {
            uint8_t *refl = NULL; size_t nrefl = 0;
            snprintf(path, sizeof path, "%s/%s.vk.json", rr.dir, key);
            if (read_file(path, &refl, &nrefl)) {
                s->frag_out_map = vkx_json_u32_or((const char *)refl, "\"frag_out_map\"", 0xffffffffu);
                free(refl);
            }
        }
    }
    if (s || st == VKX_STAGE_TS || st == VKX_STAGE_MS) {
        if (rr.job[0]) read_file(rr.job, &job, &njob);
        else { snprintf(path, sizeof path, "%s/%s.job.json", rr.dir, key); read_file(path, &job, &njob); }
    }
    vkx_mesh_stage_meta stage_meta = rr.mesh_ext ? rr.mesh_meta :
        (vkx_mesh_stage_meta){.present = 1, .stage = st, .host_slot = vkx_stage_set(st), .descriptor_set = vkx_stage_set(st)};
    if ((st == VKX_STAGE_TS || st == VKX_STAGE_MS) && (!job || !stage_meta.present)) {
        static int warned_meta;
        if (!warned_meta++) LOG("[RED] vulkan draws: native task/mesh shader %s has no matching job metadata: stage refused", key);
        free(code); free(vkb); free(job); return NULL;
    }
    if (!vkx_stage_runtime_enabled_on(st, vkd_mesh_active() != NULL)) {
        static int warned[VKX_META_STAGE_COUNT];
        if (st < VKX_META_STAGE_COUNT && !warned[st]++) LOG("[RED] vulkan draws: native %s metadata is present, but the renderer physical-proof gate is closed: stage refused", st == VKX_STAGE_TS ? "task" : "mesh");
        free(code); free(vkb); free(job); return NULL;
    }
    vks meta = {0}; vks *bindings = s ? s : &meta;
    if (s || g_gmem_trace) parse_vkb(bindings, (const char *)vkb);
    if (s) {
        s->mesh_meta = stage_meta;
        const char *c1 = job ? strstr((const char *)job, "\"c1\": \"") : NULL;   // c1: the shader's own constants, from its translation job
        uint8_t *c1d = NULL; size_t c1n = 0;
        if (c1) { c1 += 7; const char *e = strchr(c1, '"'); c1n = e ? (size_t)(e - c1) / 2 : 0; c1d = calloc(1, c1n ? c1n : 256); for (size_t i = 0; i < c1n; i++) c1d[i] = (uint8_t)hexbyte(c1 + 2 * i); }
        if (!c1d) c1d = calloc(1, 256);   // no c1 in the job: 256 zero bytes. The buffer below copied 256 bytes from a 1-byte block (or from NULL), a read past the
                                          // heap block that faulted when it ended a mapped page (SIGSEGV in sh_code at the first shader of a new scene)
        buffer_with(c1d, c1n > 65536 ? 65536 : (c1n ? c1n : 256), &s->c1, &s->c1m); s->c1_len = (uint32_t)c1n; free(c1d);
    }
    static int nanp = -1;   // the driver keeps NaN and infinities when the module asks (SignedZeroInfNanPreserve for 32-bit floats)
    if (nanp < 0) { VkPhysicalDeviceFloatControlsProperties fc = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
                    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &fc}; vkGetPhysicalDeviceProperties2(g_vkd.pd, &p2); nanp = fc.shaderSignedZeroInfNanPreserveFloat32 != 0; }
    spvr_device dev = {0, 0, 0, nanp}; size_t fw; unsigned ch = 0;
    uint32_t *fit = spvr_fit((const uint32_t *)code, ncode / 4, &dev, &fw, &ch);
    if (fit && s && (st == VKX_STAGE_TS || st == VKX_STAGE_MS)) {
        spvr_info ri; char err[160];
        int reflected = spvr_reflect(fit, fw, &ri, err, sizeof err) == 0;
        int agree = reflected && ri.stage == vstage_of(st) &&
                    !memcmp(ri.local_size, stage_meta.local, sizeof ri.local_size) &&
                    ri.shared_size >= stage_meta.shared;
        if (!agree) {
            LOG("[RED] vulkan draws: native shader %s emitted-memory metadata disagrees with its job%s%s", key,
                reflected ? "" : ": ", reflected ? "" : err);
            free(fit); fit = NULL;
        } else {
            s->mesh_meta.payload = ri.task_payload_size;
            s->mesh_meta.shared = ri.shared_size;
            s->mesh_meta.output_memory = ri.output_size;
            s->mesh_meta.output_components = ri.output_components;
        }
    }
    if (fit && s && !vkx_spv_per_primitive_locations(fit, fw, &s->mesh_meta.per_primitive_locations)) { free(fit); fit = NULL; }
    if (fit && g_gmem_trace) {
        const vkbind *gb = NULL;
        for (int i = 0; i < bindings->nb; i++) if (bindings->bd[i].k == BK_GMEM_LOG) { gb = &bindings->bd[i]; break; }
        if (gb) {
            size_t tw = 0; unsigned np = 0; uint32_t *tr = spvr_gmem_trace(fit, fw, gb->b, &tw, &np);
            if (tr && np) { free(fit); fit = tr; fw = tw; }
            else { free(tr); LOG("[RED] vulkan draws: shader %s raw-global page trace did not match its store-log ABI; original bitmap retained", key); }
        }
    }
    if (fit && (st == 0 || st == 4)) { size_t iw; uint32_t *io = spvr_init_outputs(fit, fw, &iw); if (io) { free(fit); fit = io; fw = iw; } }   // unwritten output components read 0, as on Maxwell
    if (fit) { size_t nw2; unsigned nc; uint32_t *nn = spvr_nan(fit, fw, &nw2, &nc); if (nn) { free(fit); fit = nn; fw = nw2; } }   // Maxwell's sin / cos and stores of NaN
    if (fit) { size_t bw; unsigned bp; uint32_t *bl = spvr_bindless(fit, fw, VKX_BINDLESS_SET, VKR_BINDLESS_TEXTURES, &bw, &bp);
        if (bl) { free(fit); fit = bl; fw = bw; }
    }
    if (fit && g_feedback_enabled) { size_t bw; unsigned bp; uint32_t *bl = spvr_bindless_feedback(fit, fw, VKX_BINDLESS_SET, VKX_FEEDBACK_SET, &bw, &bp);
        if (bl) { free(fit); fit = bl; fw = bw; }
    }
    free(code); free(vkb); free(job);
    *st_out = st; *nw = fw; if (ch_out) *ch_out = ch;
    return fit;
}
// Shader binaries on disk (VK_EXT_shader_object's binary form): the driver does not cache shader objects, so each one compiled again at every run and every new one in
// play stopped the frame for 5-15 ms (a 1.3 s stop when a level's 120 came at once). Found by key and variant under a directory named by the shader set, the descriptor
// layout and the driver's binary UUID and version (a driver update starts a new one). DM_VK_SHCACHE=<dir> moves it, DM_VK_SHCACHE=0 turns it off
static char g_shc[1400]; static int g_shc_on;
typedef struct { uint32_t magic, version; uint8_t uuid[16]; uint64_t size; } shc_hdr;
static VkPhysicalDeviceShaderObjectPropertiesEXT g_sop = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT};
static _Atomic uint64_t g_shc_hit, g_shc_saved, g_shc_hit_ns;
static void shc_init(uint64_t layout_hash) {
    const char *e = getenv("DM_VK_SHCACHE"); if (e && !strcmp(e, "0")) return;
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &g_sop}; vkGetPhysicalDeviceProperties2(g_vkd.pd, &p2);
    const char *base = strrchr(g_spv, '/'); base = base ? base + 1 : g_spv;
    char root[1024];
    if (e && *e) snprintf(root, sizeof root, "%s", e);
    else { const char *x = getenv("XDG_CACHE_HOME"), *h = getenv("HOME"); if (x && *x) snprintf(root, sizeof root, "%s/nxasahi/vkshaders", x); else if (h) snprintf(root, sizeof root, "%s/.cache/nxasahi/vkshaders", h); else return; }
    char u[33]; for (int i = 0; i < 16; i++) snprintf(u + 2 * i, 3, "%02x", g_sop.shaderBinaryUUID[i]);
    snprintf(g_shc, sizeof g_shc, "%s/%s-%016llx-%s-%u-f%d-g%d", root, base, (unsigned long long)layout_hash, u, g_sop.shaderBinaryVersion, SPVR_FIT_VERSION, g_gmem_trace != 0);   // f: fit rewrites; g: page-trace instrumentation
    char mk[1400]; snprintf(mk, sizeof mk, "%s", g_shc);   // mkdir -p
    for (char *q = mk + 1; *q; q++) if (*q == '/') { *q = 0; mkdir(mk, 0755); *q = '/'; }
    if (mkdir(mk, 0755) && errno != EEXIST) { LOG("[RED] vulkan draws: no shader cache directory %s", g_shc); return; }
    g_shc_on = 1; LOG("vulkan draws: shader binaries cached in %s", g_shc);
}
static void shc_path(char *out, size_t n, const char *key, int st, int gl) { snprintf(out, n, "%s/%s.%d%d.bin", g_shc, key, st, gl); }
static int shc_has(const char *key, int st, int gl) { char p[1500]; shc_path(p, sizeof p, key, st, gl); return access(p, R_OK) == 0; }
static VkShaderCreateInfoEXT sh_ci(int st, VkShaderCodeTypeEXT type, const void *code, size_t size, const VkDescriptorSetLayout *sets) {
    return (VkShaderCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = vstage_of(st),
        .nextStage = vkx_stage_next(st),
        .codeType = type, .codeSize = size, .pCode = code, .pName = "main",
        .setLayoutCount = g_feedback_enabled ? 7u : 6u, .pSetLayouts = sets};
}
// The shader object of a key's stage and variant (gl: the GL-depth form of the last stage before rasterization), from the disk cache or compiled (and then cached).
// spv: the fitted code. *compiled: 1 when it had to be compiled
static VkShaderEXT sh_object(const char *key, int st, int gl, const uint32_t *spv, size_t nw, int *compiled) {
    VkDescriptorSetLayout sets[7] = {g_dsl, g_dsl, g_dsl, g_dsl, g_dsl, g_bindless_dsl, g_feedback_dsl}; VkShaderEXT o = VK_NULL_HANDLE; char p[1500];
    if (compiled) *compiled = 0;
    if (g_shc_on) {
        shc_path(p, sizeof p, key, st, gl); uint8_t *b; size_t n; uint64_t t0 = mk_now();
        if (read_file(p, &b, &n)) {
            shc_hdr h; int ok = n >= sizeof h && (memcpy(&h, b, sizeof h), h.magic == 0x31636873u && h.version == g_sop.shaderBinaryVersion && !memcmp(h.uuid, g_sop.shaderBinaryUUID, 16) && h.size == n - sizeof h);
            if (ok) { VkShaderCreateInfoEXT ci = sh_ci(st, VK_SHADER_CODE_TYPE_BINARY_EXT, b + sizeof h, (size_t)h.size, sets); if (g_vkd.create_shaders(g_vkd.dev, 1, &ci, NULL, &o)) o = VK_NULL_HANDLE; }
            free(b);
            if (o) { atomic_fetch_add(&g_shc_hit, 1); atomic_fetch_add(&g_shc_hit_ns, mk_now() - t0);
                uint64_t hn = atomic_load(&g_shc_hit); if (!(hn & (hn - 1))) LOG("vulkan draws: %llu shaders loaded from the cache, %.2f ms each", (unsigned long long)hn, atomic_load(&g_shc_hit_ns) / 1e6 / hn);
                return o; }
        }
    }
    size_t gw = 0; uint32_t *g = gl ? spvr_gl_depth(spv, nw, &gw) : NULL;
    if (gl && !g) return VK_NULL_HANDLE;
    VkShaderCreateInfoEXT ci = sh_ci(st, VK_SHADER_CODE_TYPE_SPIRV_EXT, g ? g : spv, (g ? gw : nw) * 4, sets);
    static int compile_log = -1;
    if (compile_log < 0) {
        const char *e = getenv("DM_VK_SHADER_COMPILE_LOG");
        compile_log = e && atoi(e) != 0;
    }
    uint64_t compile_t0 = mk_now();
    if (compile_log)
        LOG("vulkan shader compile: key=%s stage=%d gl=%d fence-next=%llu begin",
            key, st, gl, (unsigned long long)vkr_fence_next());
    int cw = vkx_cw_begin(&g_cw, key, st, gl, ci.codeSize, compile_t0);
    VkResult r = g_vkd.create_shaders(g_vkd.dev, 1, &ci, NULL, &o); free(g);
    vkx_cw_end(&g_cw, cw);
    uint64_t compile_ns = mk_now() - compile_t0;
    if (compile_log)
        LOG("vulkan shader compile: key=%s stage=%d gl=%d fence-next=%llu result=%d elapsed=%.3f s",
            key, st, gl, (unsigned long long)vkr_fence_next(), (int)r, compile_ns / 1e9);
    else if (compile_ns >= 1000000000ull)   // a compile that held a frame: always in the log, with what reproduces it
        LOG("vulkan shader compile: key=%s stage=%d gl=%d %zu bytes SPIR-V result=%d took %.1f s", key, st, gl, ci.codeSize, (int)r, compile_ns / 1e9);
    if (r) return VK_NULL_HANDLE;
    if (compiled) *compiled = 1;
    if (g_shc_on && g_vkd.shader_binary) {   // into the cache: written to a temporary name and renamed, so a reader never sees half a file
        size_t n = 0;
        if (!g_vkd.shader_binary(g_vkd.dev, o, &n, NULL) && n) {
            uint8_t *b = malloc(sizeof(shc_hdr) + n); shc_hdr h = {0x31636873u, g_sop.shaderBinaryVersion, {0}, n}; memcpy(h.uuid, g_sop.shaderBinaryUUID, 16); memcpy(b, &h, sizeof h);
            if (b && !g_vkd.shader_binary(g_vkd.dev, o, &n, b + sizeof h)) {
                char t[1600]; snprintf(t, sizeof t, "%s.%ld.tmp", p, (long)syscall(SYS_gettid)); shc_path(p, sizeof p, key, st, gl);
                FILE *f = fopen(t, "wb"); if (f) { int w = fwrite(b, 1, sizeof h + n, f) == sizeof h + n; w = !fclose(f) && w; if (w && !rename(t, p)) atomic_fetch_add(&g_shc_saved, 1); else unlink(t); }
            }
            free(b);
        }
    }
    return o;
}
static vks *shader_get(const prog_t *p, int idx) {
    uint8_t k8[8]; if (!prog_key(p, (unsigned)idx, k8)) return NULL;
    uint64_t k64; memcpy(&k64, k8, 8);   // found by the key's bytes; its hex form only for a shader made now
    unsigned hsh = (k8[0] | k8[1] << 8) & 1023;
    pthread_mutex_lock(&g_shmu);
    for (vks *s = g_sh[hsh]; s; s = s->next) if (s->k64 == k64) { pthread_mutex_unlock(&g_shmu); return s->ok ? s : NULL; }
    char key[17]; for (int i = 0; i < 8; i++) snprintf(key + 2 * i, 3, "%02x", k8[i]);
    uint64_t mk0 = mk_now();
    vks *s = calloc(1, sizeof *s); snprintf(s->key, sizeof s->key, "%s", key); s->k64 = k64;
    s->next = g_sh[hsh]; g_sh[hsh] = s;
    int st = -1; size_t fw = 0; unsigned ch = 0;
    uint32_t *fit = sh_code(key, &st, &fw, s, &ch);
    if (!fit) { missing_log(key, k64, idx); g_st_nokey++; if (drawlog() || g_st_nokey <= 8 || !(g_st_nokey & (g_st_nokey - 1))) LOG("[RED] vulkan draws: no translated shader %s (index %d) in %s (%llu keys without one so far)", key, idx, g_spv, (unsigned long long)g_st_nokey); pthread_mutex_unlock(&g_shmu); return NULL; }
    s->stage = st; s->vstage = vstage_of(st);
    // Stages that can feed the rasterizer are made when a draw asks for a variant (sh_plain / gl_variant): the GL-depth form is the one drawn in a GL-depth game, and
    // making the plain one as well compiled every vertex shader twice
    if (st == 0 || st == 2 || st == 4 || st == VKX_STAGE_MS) { s->spv = fit; s->nspv = fw; s->ok = 1; }
    else { int comp; s->sh = sh_object(key, st, 0, fit, fw, &comp); free(fit); s->ok = s->sh != VK_NULL_HANDLE;
        if (comp || !s->ok) { atomic_fetch_add(&g_mk_sh, 1); atomic_fetch_add(&g_mk_sh_ns, mk_now() - mk0); } }
    if (s->ok) g_st_load++; else { g_st_fail++; LOG("[RED] vulkan draws: shader %s (stage %d) not made", key, st); }
    if (s->ok && !(g_st_load & (g_st_load - 1))) LOG("vulkan draws: %llu shaders loaded (latest %s, stage %d, %d bindings%s)", (unsigned long long)g_st_load, key, st, s->nb,
                                                     ch & SPVR_FIT_STORES ? ", [RED] writes a vertex-stage buffer the device cannot" : "");
    pthread_mutex_unlock(&g_shmu);
    return s->ok ? s : NULL;
}

// ---- resources ----
static VkBuffer g_zero, g_sink, g_support; static VkDeviceMemory g_zero_m, g_sink_m, g_support_m;
static VkImage g_dimg[6]; static VkImageView g_dview[6]; static VkDeviceMemory g_dmem[6]; static VkSampler g_dsamp;
static int g_res_ready;
// Dummies: a zero buffer, and a 1x1 zero image of each view type (1D/2D as 2D, 2D array, 3D, cube, cube array), cleared and in GENERAL layout before the first use
static void res_init(VkCommandBuffer c) {
    if (g_res_ready) return; g_res_ready = 1;
    buffer_with(NULL, 65536, &g_zero, &g_zero_m);
    buffer_with(NULL, 65536, &g_sink, &g_sink_m);
    float sup[16384] = {0}; for (int i = 0; i < 73; i++) sup[(192 + 16 * i) / 4] = 1.0f;   // render_scale = 1 (Ryujinx SupportBuffer, as nvn_mtl.m support_buf)
    buffer_with(sup, sizeof sup, &g_support, &g_support_m);
    for (int k = 0; k < 6; k++) {
        int cube = k == 3 || k == 4, three = k == 2;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, .imageType = three ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = cube ? 6 : 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        vkCreateImage(g_vkd.dev, &ci, NULL, &g_dimg[k]);
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vkd.dev, g_dimg[k], &mr);
        VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = vkr_pick_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        vkAllocateMemory(g_vkd.dev, &ma, NULL, &g_dmem[k]); vkBindImageMemory(g_vkd.dev, g_dimg[k], g_dmem[k], 0);
        static const VkImageViewType vt[6] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_IMAGE_VIEW_TYPE_2D};
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_dimg[k], .viewType = vt[k], .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, cube ? 6 : 1}};
        vkCreateImageView(g_vkd.dev, &vi, NULL, &g_dview[k]);
        VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_dimg[k], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, cube ? 6 : 1}};
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkClearColorValue z = {{0, 0, 0, 0}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, cube ? 6 : 1};
        vkCmdClearColorImage(c, g_dimg[k], VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
    }
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .maxLod = VK_LOD_CLAMP_NONE,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    vkCreateSampler(g_vkd.dev, &si, NULL, &g_dsamp);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static int dummy_index(const vkbind *b) { return b->dim == 3 ? 2 : b->dim == 4 ? (b->arr ? 4 : 3) : b->arr ? 1 : 0; }

// Samplers, cached by their parameters (the device allows a few thousand)
typedef struct samp_ent { samp_par p; VkSampler s; struct samp_ent *next; } samp_ent;
static samp_ent *g_samps[256]; static int g_nsamp;
static VkSamplerAddressMode wrap(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 5: case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                 case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static VkSampler sampler_get(const samp_par *p) {
    uint32_t h = 2166136261u; for (size_t i = 0; i < sizeof *p; i++) h = (h ^ ((const uint8_t *)p)[i]) * 16777619u;
    for (samp_ent *e = g_samps[h & 255]; e; e = e->next) if (!memcmp(&e->p, p, sizeof *p)) return e->s;
    if (g_nsamp >= 3900) return g_dsamp;
    int minf = p->minf, mip = minf >= 2;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = (minf == 1 || minf == 3 || minf == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = (minf == 4 || minf == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = wrap(p->ws), .addressModeV = wrap(p->wt), .addressModeW = wrap(p->wr), .mipLodBias = p->lod_bias,
        .anisotropyEnable = p->aniso > 1, .maxAnisotropy = p->aniso > 16 ? 16 : p->aniso < 1 ? 1 : p->aniso,
        .compareEnable = p->cmp_mode == 1, .compareOp = p->cmp_func >= 1 && p->cmp_func <= 8 ? (VkCompareOp)(p->cmp_func - 1) : VK_COMPARE_OP_ALWAYS,
        .minLod = p->lod_min, .maxLod = mip ? p->lod_max : 0.25f, .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
    samp_ent *e = calloc(1, sizeof *e); e->p = *p;
    if (vkCreateSampler(g_vkd.dev, &si, NULL, &e->s)) { free(e); return g_dsamp; }
    e->next = g_samps[h & 255]; g_samps[h & 255] = e; g_nsamp++;
    return e->s;
}
// The components a sampled texture reads: the format's own (formats without alpha read a as 1 and so on, vk_fmt.c), then the texture's swizzle from its builder
// (TextureBuilderSetSwizzle: 3 bits per component, 0 zero 1 one 2 R 3 G 4 B 5 A), then the view's (TextureView SetSwizzle), each picking from what the one before gives
static VkComponentSwizzle swz_pick(int sel, const VkComponentSwizzle base[4]) {
    switch (sel) { case 0: return VK_COMPONENT_SWIZZLE_ZERO; case 1: return VK_COMPONENT_SWIZZLE_ONE; case 2: case 3: case 4: case 5: return base[sel - 2]; default: return VK_COMPONENT_SWIZZLE_IDENTITY; }
}
static VkComponentMapping tex_swizzle(const tex_t *o, const view_t *v, int has_v) {
    VkComponentMapping m; vkf_from_nvn(o->d.format, NULL, &m);
    VkComponentSwizzle c[4] = {m.r == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : m.r, m.g == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : m.g,
                               m.b == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : m.b, m.a == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : m.a};
    if (o->set_swz) { VkComponentSwizzle n[4]; for (int i = 0; i < 4; i++) n[i] = swz_pick((o->swz >> (3 * i)) & 7, c); memcpy(c, n, sizeof c); }
    if (has_v && v->set_swizzle) { VkComponentSwizzle n[4]; for (int i = 0; i < 4; i++) n[i] = swz_pick(v->swz[i], c); memcpy(c, n, sizeof c); }
    return (VkComponentMapping){c[0], c[1], c[2], c[3]};
}
// A view of an image as the shader declares it (dim / arrayed from its binding), cached on the image (under g_tmu)
static VkImageView view_get(vtex *t, const vkbind *b, const tex_t *o, const view_t *ov, int has_v) {
    VkImageViewType vt = b->dim == 3 ? VK_IMAGE_VIEW_TYPE_3D : b->dim == 4 ? (b->arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE)
                       : b->arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if ((vt == VK_IMAGE_VIEW_TYPE_3D) != (t->d > 1)) return VK_NULL_HANDLE;   // the image is not of this kind
    if ((vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && t->layers % 6) return VK_NULL_HANDLE;
    VkComponentMapping sw = tex_swizzle(o, ov, has_v); VkFormat vf = vkr_view_format(t->fmt, ov, has_v);
    uint64_t key = (uint32_t)vt | (uint32_t)sw.r << 4 | (uint32_t)sw.g << 8 | (uint32_t)sw.b << 12 | (uint32_t)sw.a << 16 | (uint64_t)(uint32_t)vf << 32;   // the view type, components and compatible format (below 0x80000000: attachment views)
    for (vkview *v = t->views; v; v = v->next) if (v->key == key) return v->view;
    VkImageAspectFlags asp = t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : t->aspect;   // a depth-stencil image samples its depth
    uint32_t layers = vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_3D ? 1 : vt == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? t->layers - t->layers % 6 : t->layers;
    VkImageViewUsageCreateInfo vu = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = vf != t->fmt ? &vu : NULL, .image = t->img, .viewType = vt, .format = vf,
        .components = sw, .subresourceRange = {asp, 0, t->levels, 0, layers}};
    vkview *v = calloc(1, sizeof *v); v->key = key;
    if (vkCreateImageView(g_vkd.dev, &vi, NULL, &v->view)) { free(v); return VK_NULL_HANDLE; }
    v->next = t->views; t->views = v;
    return v->view;
}

// The translator indexes NVN's registered texture and sampler IDs directly.
// Registrations dirty only their affected array element; at the next command
// buffer we materialise all three image shapes used by this title.  SPIR-V is
// canonicalised to the matching binding by spvr_bindless(), independent of
// the translator's per-module 564..566 discovery order.
enum { BINDLESS_RETRY_CAP = 16384, BINDLESS_RETRY_BATCH = 32 };
static int g_bindless_retry[BINDLESS_RETRY_CAP];
static unsigned g_bindless_retry_n, g_bindless_retry_at;

static void bindless_retry_add(int id) {
    for (unsigned i = 0; i < g_bindless_retry_n; i++) if (g_bindless_retry[i] == id) return;
    if (g_bindless_retry_n < BINDLESS_RETRY_CAP) g_bindless_retry[g_bindless_retry_n++] = id;
}

static void bindless_feedback_scan(int slot) {
    if (!g_feedback_enabled || slot < 0 || slot >= CB_RING || !g_feedback_map[slot]) return;
    enum { WORDS = VKR_BINDLESS_TEXTURES / 32 };
    unsigned fresh = 0;
    for (unsigned i = 0; i < WORDS; i++) {
        uint32_t bits = g_feedback_map[slot][i];
        while (bits) {
            unsigned b = (unsigned)__builtin_ctz(bits), id = i * 32u + b;
            if (id && !g_bindless_real[id]) { bindless_retry_add((int)id); fresh++; }
            bits &= bits - 1;
        }
    }
    memset(g_feedback_map[slot], 0, WORDS * sizeof(uint32_t));
    if (fresh) {
        static uint64_t total; total += fresh;
        if (total <= 64 || !(total & (total - 1)))
            LOG("vulkan draws: bindless feedback observed %u unresolved texture IDs (%llu observations, %u queued)",
                fresh, (unsigned long long)total, g_bindless_retry_n);
    }
}

static void bindless_write_dummy(const int *ids, int n) {
    if (n <= 0) return;
    VkWriteDescriptorSet wr[32 * 3]; VkDescriptorImageInfo ii[32 * 3]; int nw = 0;
    for (int k = 0; k < n; k++) for (int q = 0; q < 3; q++) {
        ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, g_dview[q == 1], VK_IMAGE_LAYOUT_GENERAL};
        wr[nw] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = g_bindless_set, .dstBinding = (uint32_t)q,
            .dstArrayElement = (uint32_t)ids[k], .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
        nw++;
    }
    vkUpdateDescriptorSets(g_vkd.dev, (uint32_t)nw, wr, 0, NULL);
}

// 1: a real image was found/made, 0: keep retrying, -1: ID is no longer registered.
static int bindless_texture_views(int id, const vkbind shape[3], uint64_t budget,
                                  uint64_t single_max, VkImageView iv[3]) {
    tex_t o; view_t v; int hv;
    if (!vkr_treg(id, &o, &v, &hv)) return -1;
    pthread_mutex_lock(&g_tmu);
    vtex *t = vkr_tex_sampled_budgeted(&o, budget, single_max);
    if (t) for (int q = 0; q < 3; q++) { VkImageView x = view_get(t, &shape[q], &o, &v, hv); if (x) iv[q] = x; }
    pthread_mutex_unlock(&g_tmu);
    return t ? 1 : 0;
}

static void bindless_update(VkCommandBuffer c) {
    (void)c;
    enum { BATCH = 32 };
    int ids[BATCH], n;
    static int base_done;
    static uint64_t tex_writes, samp_writes, tex_deferred, tex_retried;
    static int budget_init; static uint64_t budget, single_max;
    if (!budget_init) {
        budget_init = 1; const char *e = getenv("DM_VK_BINDLESS_IMAGE_MB");
        if (e && *e) { char *end = NULL; unsigned long long mb = strtoull(e, &end, 10); if (end != e && !*end && mb) budget = mb << 20; }
        e = getenv("DM_VK_BINDLESS_IMAGE_MAX_MB");
        if (e && *e) { char *end = NULL; unsigned long long mb = strtoull(e, &end, 10); if (end != e && !*end && mb) single_max = mb << 20; }
        if (budget || single_max) LOG("vulkan draws: speculative bindless images limited to %llu MiB total / %llu MiB each; directly bound images and render targets remain unrestricted",
                                      (unsigned long long)(budget >> 20), (unsigned long long)(single_max >> 20));
    }
    uint64_t tex_before = tex_writes, samp_before = samp_writes,
             retry_before = tex_retried;
    static const vkbind shape[3] = {{.dim = 2}, {.dim = 2, .arr = 1}, {.dim = 2, .dep = 1}};
    if (!base_done) {
        VkDescriptorImageInfo ii[4] = {{VK_NULL_HANDLE, g_dview[0], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, g_dview[1], VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, g_dview[0], VK_IMAGE_LAYOUT_GENERAL}, {g_dsamp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}};
        VkWriteDescriptorSet wr[4];
        for (int q = 0; q < 4; q++) wr[q] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_bindless_set,
            .dstBinding = (uint32_t)q, .dstArrayElement = 0, .descriptorCount = 1,
            .descriptorType = q == VKX_BINDLESS_SAMP ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[q]};
        vkUpdateDescriptorSets(g_vkd.dev, 4, wr, 0, NULL); base_done = 1;
    }
    /* The first genuine mesh scene is also this title's large bindless
     * generation boundary.  Roll only images sampled before that boundary;
     * replacements created by the retry queue have a newer last_sample and
     * cannot be selected by later batches.  This preserves the strict byte
     * budget instead of letting title and scene generations accumulate. */
    static uint64_t mesh_generation, recent_generation, rolled_bytes;
    static int roll_done, roll_enabled = -1;
    if (roll_enabled < 0) { const char *e = getenv("DM_VK_BINDLESS_ROLLOVER"); roll_enabled = e && atoi(e) != 0; }
    if (roll_enabled && budget && !roll_done && vkr_mesh_issue_count()) {
        if (!mesh_generation) {
            mesh_generation = vkr_submission_count();
            recent_generation = mesh_generation > 2048 ? mesh_generation - 2048 : 1;
            g_bindless_retry_n = (unsigned)vkr_bindless_recent_tex_ids(recent_generation,
                g_bindless_retry, BINDLESS_RETRY_CAP);
            g_bindless_retry_at = 0;
            LOG("vulkan draws: first mesh bindless generation rollover begins at submission %llu; prioritising %u IDs registered since %llu",
                (unsigned long long)mesh_generation, g_bindless_retry_n,
                (unsigned long long)recent_generation);
        }
        int evicted[32]; uint64_t retired = 0;
        int ne = vkr_bindless_evict_old_speculative(recent_generation, evicted, 32, &retired);
        bindless_write_dummy(evicted, ne);
        rolled_bytes += retired;
        if (!retired) {
            roll_done = 1;
            LOG("vulkan draws: first mesh bindless generation rollover retired %.0f MiB; %u IDs await normal retry",
                rolled_bytes / 1048576.0, g_bindless_retry_n);
        }
    }
    // A registration may precede the first render/storage use of its image.
    // If speculative creation was budget-rejected, retry a bounded number of
    // those IDs each command buffer so the slot is replaced as soon as the
    // unrestricted attachment/dispatch path creates the image.
    {
        VkWriteDescriptorSet wr[BINDLESS_RETRY_BATCH * 3];
        VkDescriptorImageInfo ii[BINDLESS_RETRY_BATCH * 3];
        int nw = 0;
        unsigned attempts = g_bindless_retry_n < BINDLESS_RETRY_BATCH ?
            g_bindless_retry_n : BINDLESS_RETRY_BATCH;
        for (unsigned k = 0; k < attempts && g_bindless_retry_n; k++) {
            if (g_bindless_retry_at >= g_bindless_retry_n) g_bindless_retry_at = 0;
            int id = g_bindless_retry[g_bindless_retry_at];
            VkImageView iv[3] = {g_dview[0], g_dview[1], g_dview[0]};
            uint64_t retry_max = g_feedback_enabled && single_max < (128ull << 20) ? (128ull << 20) : single_max;
            int state = bindless_texture_views(id, shape, budget, retry_max, iv);
            if (state <= 0) {
                if (state == 0) { g_bindless_retry_at++; continue; }
                if (id > 0 && id < VKR_BINDLESS_TEXTURES) g_bindless_real[id] = 0;
                g_bindless_retry[g_bindless_retry_at] = g_bindless_retry[--g_bindless_retry_n];
                continue;
            }
            for (int q = 0; q < 3; q++) {
                ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, iv[q], VK_IMAGE_LAYOUT_GENERAL};
                wr[nw] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = g_bindless_set, .dstBinding = (uint32_t)q,
                    .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
                nw++;
            }
            if (id > 0 && id < VKR_BINDLESS_TEXTURES) g_bindless_real[id] = 1;
            g_bindless_retry[g_bindless_retry_at] = g_bindless_retry[--g_bindless_retry_n];
            tex_retried++;
        }
        if (nw) vkUpdateDescriptorSets(g_vkd.dev, (uint32_t)nw, wr, 0, NULL);
    }
    while ((n = vkr_bindless_take_tex_dirty(ids, BATCH)) > 0) {
        VkWriteDescriptorSet wr[BATCH * 3]; VkDescriptorImageInfo ii[BATCH * 3]; int nw = 0;
        for (int k = 0; k < n; k++) {
            int id = ids[k]; VkImageView iv[3] = {g_dview[0], g_dview[1], g_dview[0]};
            int state = -1;
            if (id > 0 && id < VKR_BINDLESS_TEXTURES) g_bindless_real[id] = 0;
            if (!g_feedback_enabled || (id > 0 && id < VKR_BINDLESS_TEXTURES && g_bindless_seed[id])) {
                uint64_t initial_max = g_feedback_enabled && single_max < (128ull << 20) ?
                    (128ull << 20) : single_max;
                state = bindless_texture_views(id, shape, budget, initial_max, iv);
                if (state == 0) { tex_deferred++; bindless_retry_add(id); }
                else if (state > 0 && id > 0 && id < VKR_BINDLESS_TEXTURES) g_bindless_real[id] = 1;
            } else tex_deferred++;
            for (int q = 0; q < 3; q++) {
                ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, iv[q], VK_IMAGE_LAYOUT_GENERAL};
                wr[nw] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_bindless_set,
                    .dstBinding = (uint32_t)q, .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
                nw++;
            }
        }
        vkUpdateDescriptorSets(g_vkd.dev, (uint32_t)nw, wr, 0, NULL); tex_writes += (uint64_t)n;
    }
    while ((n = vkr_bindless_take_samp_dirty(ids, BATCH)) > 0) {
        VkWriteDescriptorSet wr[BATCH]; VkDescriptorImageInfo ii[BATCH]; int nw = 0;
        for (int k = 0; k < n; k++) {
            samp_par p; VkSampler s = vkr_sreg(ids[k], &p) ? sampler_get(&p) : g_dsamp;
            ii[nw] = (VkDescriptorImageInfo){s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            wr[nw] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_bindless_set,
                .dstBinding = VKX_BINDLESS_SAMP, .dstArrayElement = (uint32_t)ids[k], .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &ii[nw]};
            nw++;
        }
        vkUpdateDescriptorSets(g_vkd.dev, (uint32_t)nw, wr, 0, NULL); samp_writes += (uint64_t)n;
    }
    {   // The pool's standing state, every 4096 submissions and loud while texture IDs draws may sample still hold the blank placeholder: a run with
        // thousands of them pending drew black models and empty text with nothing in the log (Ocarina of Time: 2520 of 8041 after seven minutes)
        static uint64_t calls; calls++;
        if (!(calls & 4095)) { unsigned real = 0; for (int i = 1; i < VKR_BINDLESS_TEXTURES; i++) real += g_bindless_real[i];
            LOG("%svulkan draws: bindless pool after %llu submissions: %llu texture IDs written, %u hold a real image, %u pending retry behind the %.0f MiB speculative budget (%llu placeholder writes, %llu repaired)",
                g_bindless_retry_n ? "[RED] " : "", (unsigned long long)calls, (unsigned long long)tex_writes, real, g_bindless_retry_n, budget / 1048576.0,
                (unsigned long long)tex_deferred, (unsigned long long)tex_retried); }
    }
    static int shown;
    if ((tex_writes != tex_before || samp_writes != samp_before || tex_retried != retry_before) &&
        (!shown++ || tex_writes + samp_writes <= 64 || !((tex_writes + samp_writes) & (tex_writes + samp_writes - 1))))
        LOG("vulkan draws: bindless pool populated: %llu texture IDs, %llu sampler IDs; %llu initial dummy writes, %llu repaired, %u pending retry",
            (unsigned long long)tex_writes, (unsigned long long)samp_writes,
            (unsigned long long)tex_deferred, (unsigned long long)tex_retried,
            g_bindless_retry_n);
}

// ---- descriptor sets: a pool per ring slot, reset when its submission is reused ----
#define POOL_SETS 256   // sets per pool; a submission takes as many pools as it needs, all reset when its ring slot is reused
typedef struct { VkDescriptorPool p[256]; uint16_t used[256]; int n, cur; uint8_t limit_logged; } pool_chain;
// Sets made in this ring slot, by a hash of everything written into them: a draw whose stage would get the same descriptors takes the set already made (allocating and
// writing a set per stage per draw was the host's largest cost after the dirty checks). Emptied with the slot's pools.  Keep the cache comfortably above the
// submission high-water below: with the old 4K table, an eight-entry collision window could fill, after which repeated states were allocated but never cached.
#define SETC 32768
#define SETC_PROBES 8
typedef struct { uint64_t h; VkDescriptorSet set; } setc_ent;
static setc_ent g_setc[CB_RING][SETC]; static uint64_t g_setc_hit, g_setc_miss;
static pool_chain g_pools[CB_RING]; static int g_pool_slot = -1; static uint32_t g_sub_sets;
static void gmem_log_poll(uint64_t done);
static VkDescriptorPool pool_new(void) {
    VkDescriptorPoolSize ps[5]; uint32_t np = 0;
    static const VkDescriptorType T[5] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER};
    for (int k = 0; k < 5; k++) if (g_per_set[k]) ps[np++] = (VkDescriptorPoolSize){T[k], g_per_set[k] * POOL_SETS};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = POOL_SETS, .poolSizeCount = np, .pPoolSizes = ps};
    VkDescriptorPool p = VK_NULL_HANDLE;
    VkResult r = vkCreateDescriptorPool(g_vkd.dev, &pi, NULL, &p);
    if (r != VK_SUCCESS) LOG("[RED] vulkan draws: vkCreateDescriptorPool failed: VkResult %d", (int)r);
    return p;
}
static void arena_reset(int slot);
void vkx_sub_begin(int slot, VkCommandBuffer c, uint64_t done) {
    if (!draw_init()) return;
    gmem_log_poll(done);   // before a completed slot's mapped arena is reused
    res_init(c);
    bindless_feedback_scan(slot);
    vkr_cpu_written_refresh();
    bindless_update(c);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, VKX_BINDLESS_SET, 1, &g_bindless_set, 0, NULL);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl_cs, VKX_BINDLESS_SET, 1, &g_bindless_set, 0, NULL);
    if (g_feedback_enabled) {
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, VKX_FEEDBACK_SET, 1, &g_feedback_set[slot], 0, NULL);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl_cs, VKX_FEEDBACK_SET, 1, &g_feedback_set[slot], 0, NULL);
    }
    pool_chain *pc = &g_pools[slot];   // the slot's last submission is done (vk_replay.c waited for it)
    for (int i = 0; i < pc->n; i++) {
        // A failed reset leaves the pool unusable.  Replace it rather than
        // carrying unknown driver state into the next submission.  A failed
        // replacement remains null and stage_set fails closed at that pool.
        if (!pc->p[i]) {
            pc->p[i] = pool_new(); pc->used[i] = 0;
            continue;
        }
        VkResult r = vkResetDescriptorPool(g_vkd.dev, pc->p[i], 0);
        if (r == VK_SUCCESS) { pc->used[i] = 0; continue; }
        LOG("[RED] vulkan draws: vkResetDescriptorPool failed: VkResult %d pool %d used %u; replacing pool",
            (int)r, i, (unsigned)pc->used[i]);
        vkDestroyDescriptorPool(g_vkd.dev, pc->p[i], NULL);
        pc->p[i] = pool_new(); pc->used[i] = 0;
    }
    pc->cur = 0; pc->limit_logged = 0; g_sub_sets = 0; memset(g_setc[slot], 0, sizeof g_setc[slot]);
    arena_reset(slot);
    g_pool_slot = slot;
}
int vkx_descriptor_pressure(void) {
    return g_pool_slot >= 0 &&
           vkx_descriptor_pressure_values(g_sub_sets,
                                          (uint32_t)g_pools[g_pool_slot].cur);
}
// ---- the driver banks c0 / c2 (nvn_mtl.m's layout): filled per draw / dispatch from the bound state, for shaders that read them ----
// c0: per stage a storage buffer table (16 slots of address 64 + size 32 bits at base + 16 r: vertex 0x110, compute 0x310, geometry 0x410, pixel 0x510); compute also
// the dispatch size at 0x200 and the uniform buffer table at 0x210 + 16 r; graphics stages the largest point size at 0x610 and the base instance at 0x644.
// c2: the handle tables: combined texture slots from 0x20 (slots 32-39 are the image slots, 0x120), separate textures from 0x168, samplers from 0x568
enum { DRV_SIZE = 0xA00, ARENA = 8 << 20 };
typedef struct { VkBuffer b[64]; VkDeviceMemory m[64]; uint8_t *p[64]; int n, cur; size_t off; } arena_t;
static arena_t g_arena[CB_RING]; static uint32_t g_dispatch[3]; static uint32_t g_binst;
static VkDescriptorBufferInfo g_mesh_first_desc[NST]; static uint8_t g_mesh_first_valid[NST];
void vkx_mesh_abort(void) {
    memset(g_mesh_first_desc, 0, sizeof g_mesh_first_desc);
    memset(g_mesh_first_valid, 0, sizeof g_mesh_first_valid);
}
static int arena_get(size_t n, VkBuffer *b, VkDeviceSize *off, uint8_t **p) {
    arena_t *a = &g_arena[g_pool_slot];
    for (;;) {
        if (a->cur < a->n && a->off + n <= ARENA) { *b = a->b[a->cur]; *off = a->off; *p = a->p[a->cur] + a->off; a->off += (n + 255) & ~(size_t)255; return 1; }
        if (a->cur < a->n - 1) { a->cur++; a->off = 0; continue; }
        if (a->n == 64) return 0;
        buffer_with(NULL, ARENA, &a->b[a->n], &a->m[a->n]);
        void *mp; vkMapMemory(g_vkd.dev, a->m[a->n], 0, VK_WHOLE_SIZE, 0, &mp); a->p[a->n] = mp;
        a->cur = a->n++; a->off = 0;
    }
}
static uint64_t g_arena_gen[CB_RING];   // +1 when the slot's arena is reset (drv_fill's reuse is for the arena's current contents only)
static void arena_reset(int slot) { g_arena[slot].cur = 0; g_arena[slot].off = 0; g_arena_gen[slot]++; vkx_mesh_abort(); }
// The last fill of each stage's bank in this ring slot: a fill with the same bytes reuses it (most draws change none of what the banks hold), which also lets their
// descriptor sets be reused (stage_set)
static struct { int slot; uint64_t gen; VkDescriptorBufferInfo at; uint8_t d[DRV_SIZE]; } g_drv_last[NST][3];
// NVN2 push constants (c[24], DMSH_PUSH_BANK; dmsh_spv.py binds cbuf reg 21 as binding 19): the block as the record stream left it (vkr_push), in the arena, reused while unchanged
static struct { int slot; uint64_t gen; VkDescriptorBufferInfo at; uint8_t d[DMSH_PUSH_SIZE]; } g_push_last[NST];
static int push_fill(int st, VkDescriptorBufferInfo *out) {
    const uint8_t *d = vkr_push(st == 5);
    if (g_push_last[st].slot == g_pool_slot && g_push_last[st].gen == g_arena_gen[g_pool_slot] && !memcmp(g_push_last[st].d, d, DMSH_PUSH_SIZE)) { *out = g_push_last[st].at; return 1; }
    VkBuffer b; VkDeviceSize off; uint8_t *ad;
    if (!arena_get(DMSH_PUSH_SIZE, &b, &off, &ad)) return 0;
    memcpy(ad, d, DMSH_PUSH_SIZE);
    *out = (VkDescriptorBufferInfo){b, off, DMSH_PUSH_SIZE};
    g_push_last[st].slot = g_pool_slot; g_push_last[st].gen = g_arena_gen[g_pool_slot]; g_push_last[st].at = *out; memcpy(g_push_last[st].d, d, DMSH_PUSH_SIZE);
    return 1;
}
static int drv_fill(int st, int actual_st, int bank, VkDescriptorBufferInfo *out) {
    VkBuffer b; VkDeviceSize off; uint8_t *ad, d[DRV_SIZE];
    memset(d, 0, DRV_SIZE);
    if (bank == 0) {
        int base = vkx_stage_c0_ssbo_base(actual_st);
        if (base >= 0) for (int r = 0; r < 16; r++) { memcpy(d + base + 16 * r, &S.ssbo[st][r].addr, 8); uint32_t n = (uint32_t)S.ssbo[st][r].size; memcpy(d + base + 16 * r + 8, &n, 4); }
        if (st == 5) { memcpy(d + 0x200, g_dispatch, 12); if (dm_sdk_ounce()) memcpy(d + DMSH_C0_DISPATCH_NVN2, g_dispatch, 12); for (int r = 0; r < 16; r++) { memcpy(d + 0x210 + 16 * r, &S.ubo[st][r].addr, 8); uint32_t n = (uint32_t)S.ubo[st][r].size; memcpy(d + 0x210 + 16 * r + 8, &n, 4); } }
        else { const float maxp = 2047.0f; memcpy(d + 0x610, &maxp, 4); memcpy(d + 0x644, &g_binst, 4); }
    } else if (bank == 2) {
        for (int k = 0; k < 41; k++) { uint64_t h = k >= 32 && k < 40 ? S.img[st][k - 32] : S.tex[st][k]; memcpy(d + 0x20 + 8 * k, &h, 8); }
        for (int k = 0; k < 128; k++) { memcpy(d + 0x168 + 8 * k, &S.tex[st][128 + k], 8); memcpy(d + 0x568 + 8 * k, &S.samp[st][k], 8); }
    }
    int bi = bank == 2 ? 2 : bank == 0 ? 0 : 1;
    if (g_drv_last[st][bi].slot == g_pool_slot && g_drv_last[st][bi].gen == g_arena_gen[g_pool_slot] && !memcmp(g_drv_last[st][bi].d, d, DRV_SIZE)) { *out = g_drv_last[st][bi].at; return 1; }
    if (!arena_get(DRV_SIZE, &b, &off, &ad)) return 0;
    memcpy(ad, d, DRV_SIZE);
    *out = (VkDescriptorBufferInfo){b, off, DRV_SIZE};
    g_drv_last[st][bi].slot = g_pool_slot; g_drv_last[st][bi].gen = g_arena_gen[g_pool_slot]; g_drv_last[st][bi].at = *out; memcpy(g_drv_last[st][bi].d, d, DRV_SIZE);
    return 1;
}
// Raw global memory uses dmsh_abi.h's sorted table. The sparse region buffer
// has one stable BDA; only contiguous backed runs are exported, so a shader
// cannot address an uncommitted chunk. Keep the table for this command-ring
// slot while vk_mem's generation stands, as the driver banks are kept above.
static struct { int slot; uint64_t arena_gen, mem_gen; VkDescriptorBufferInfo at; } g_gmem_last;
static int gmem_map_fill(VkDescriptorBufferInfo *out) {
    const size_t bytes = (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    uint64_t gen = vkm_generation();
    if (g_gmem_last.slot == g_pool_slot && g_gmem_last.arena_gen == g_arena_gen[g_pool_slot] && g_gmem_last.mem_gen == gen) {
        *out = g_gmem_last.at; return 1;
    }
    VkBuffer b; VkDeviceSize off; uint8_t *d;
    if (!arena_get(bytes, &b, &off, &d)) return 0;
    int n = vkm_gmem_map(d, bytes, &gen);
    if (n < 0) return 0;
    *out = (VkDescriptorBufferInfo){b, off, bytes};
    g_gmem_last = (typeof(g_gmem_last)){g_pool_slot, g_arena_gen[g_pool_slot], gen, *out};
    static uint64_t shown_gen;
    if (shown_gen != g_gmem_last.mem_gen) { shown_gen = g_gmem_last.mem_gen; LOG("vulkan draws: raw-global map exports %d backed guest-memory runs", n); }
    return 1;
}

typedef struct { uint32_t *w; uint64_t fence; char key[17]; } gmem_log;
static gmem_log *g_gmem_log; static size_t g_ngmem_log, g_cgmem_log;
static int page_cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void gmem_log_poll(uint64_t done) {
    if (!g_gmem_trace || !g_ngmem_log) return;
    size_t keep = 0;
    for (size_t i = 0; i < g_ngmem_log; i++) {
        gmem_log *l = &g_gmem_log[i];
        if (l->fence > done) { if (keep != i) g_gmem_log[keep] = *l; keep++; continue; }
        uint64_t pages[VKX_GMEM_TRACE_SLOTS]; int overflow = 0; uint32_t stores = 0;
        size_t n = vkx_gmem_trace_pages(l->w, VKX_GMEM_TRACE_WORDS, pages, VKX_GMEM_TRACE_SLOTS, &overflow, &stores);
        if (n) {
            qsort(pages, n, sizeof *pages, page_cmp);
            size_t u = 0; for (size_t j = 0; j < n; j++) if (!u || pages[j] != pages[u - 1]) pages[u++] = pages[j]; n = u;
        }
        vkr_gmem_invalidate_pages(l->key, pages, n, overflow, stores);
    }
    g_ngmem_log = keep;
}

// Default: the translated shader's original per-map-entry bitmap, unchanged.
// Opt in: the first atomic winner also appends a full guest page. Its private
// log is consumed only after this submission's timeline fence completes.
static int gmem_log_fill(const vks *s, VkDescriptorBufferInfo *out) {
    const size_t bytes = g_gmem_trace ? VKX_GMEM_TRACE_BYTES : (DMSH_GMEM_MAP_ENTRIES + 7u) / 8u;
    VkBuffer b; VkDeviceSize off; uint8_t *d;
    if (!arena_get(bytes, &b, &off, &d)) return 0;
    memset(d, 0, bytes); *out = (VkDescriptorBufferInfo){b, off, bytes};
    if (g_gmem_trace) {
        if (g_ngmem_log == g_cgmem_log) { g_cgmem_log = g_cgmem_log ? 2 * g_cgmem_log : 256; g_gmem_log = realloc(g_gmem_log, g_cgmem_log * sizeof *g_gmem_log); }
        gmem_log *l = &g_gmem_log[g_ngmem_log++]; l->w = (uint32_t *)d; l->fence = vkr_fence_next(); snprintf(l->key, sizeof l->key, "%s", s->key);
    } else { static _Atomic int warned; if (!atomic_exchange(&warned, 1)) LOG("[RED] vulkan draws: raw-global store logs are bound but not consumed yet; separately uploaded textures may miss those GPU writes"); }
    return 1;
}
// A storage image view of the texture an image handle names, as the shader declares it (under g_tmu). NVN permits
// compatible 32-bit colour images to be addressed as R32 integer words; Ocarina uses that for both packed scene data
// and NDSR image atomics. Using the base colour view would numerically convert those words instead of preserving bits.
static VkImageView image_view(vtex *t, const vkbind *b, const view_t *v, int has_v) {
    if (!(t->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    VkImageViewType vt = b->dim == 3 ? VK_IMAGE_VIEW_TYPE_3D : (b->arr || b->dim == 4) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;   // a cube is written as its 6 layers
    if ((vt == VK_IMAGE_VIEW_TYPE_3D) != (t->d > 1)) return VK_NULL_HANDLE;
    VkFormat vf = vkr_view_format(t->fmt, v, has_v);
    int word = vkr_word_view_compatible(t->fmt, vf) && vf != t->fmt;
    uint64_t key = 0x40000000u | (uint32_t)vt | (uint64_t)(uint32_t)vf << 32;
    for (vkview *v = t->views; v; v = v->next) if (v->key == key) return v->view;
    VkImageViewUsageCreateInfo vu = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = VK_IMAGE_USAGE_STORAGE_BIT};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = vf != t->fmt ? &vu : NULL,
        .image = t->img, .viewType = vt, .format = vf,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_3D ? 1 : t->layers}};
    vkview *iv = calloc(1, sizeof *iv); iv->key = key;
    if (vkCreateImageView(g_vkd.dev, &vi, NULL, &iv->view)) { free(iv); return VK_NULL_HANDLE; }
    iv->next = t->views; t->views = iv;
    if (word) { static int logged; if (!logged++) LOG("vulkan draws: compatible 32-bit colour storage uses its guest R32 integer word view"); }
    return iv->view;
}
// The 32-bit word at byte off of stage st's uniform slot reg (a bindless handle the game wrote there): 0 when unbound or out of range
static uint32_t ubo_word(int st, int reg, int off) {
    if (reg < 0 || reg >= 16 || off < 0) return 0;
    const bbuf *u = &S.ubo[st][reg]; if (!u->addr || (uint64_t)off + 4 > u->size) return 0;
    uint32_t w; memcpy(&w, (const void *)(uintptr_t)(u->addr + (uint64_t)off), 4); return w;
}
static int buf_desc(uint64_t addr, uint64_t size, uint64_t align, uint64_t maxr, VkDescriptorBufferInfo *out) {
    VkBuffer b; VkDeviceSize off; VkDeviceAddress g;
    if (!addr || !vkm_lookup((const void *)(uintptr_t)addr, size ? size : 1, &b, &off, &g)) return 0;
    if (off % align) { g_miss[M_UNALIGNED]++; return 0; }
    *out = (VkDescriptorBufferInfo){b, off, size && size < maxr ? size : maxr};
    return 1;
}
// The descriptor set of one stage's shader for the draw now: 0 when nothing could be allocated
static uint64_t g_prof_ns[8];   // DM_VK_PROF: shaders + sets, state, vertex input, the draw call; set writes listed, sets allocated + written, set cache hits, misses (counts)
void vkx_prof_take(uint64_t out[8]) { memcpy(out, g_prof_ns, sizeof g_prof_ns); memset(g_prof_ns, 0, sizeof g_prof_ns); }
static int g_dprof = -1;
static uint64_t mix64(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 0xff51afd7ed558ccdull; return h ^ (h >> 33); }
static VkDescriptorSet stage_set(int st, vks *s) {
    VkDescriptorSet set = VK_NULL_HANDLE; pool_chain *pc = &g_pools[g_pool_slot];
    VkWriteDescriptorSet wr[NBIND_MAX]; VkDescriptorBufferInfo bi[NBIND_MAX]; VkDescriptorImageInfo ii[NBIND_MAX]; int nw = 0;
    if (g_dprof < 0) g_dprof = getenv("DM_VK_PROF") != NULL;
    uint64_t pt0 = g_dprof ? mk_now() : 0;
    pthread_mutex_lock(&g_tmu);
    for (int i = 0; i < s->nb; i++) {
        const vkbind *b = &s->bd[i]; VkWriteDescriptorSet *w = &wr[nw];
        if (bindless_vkb(b)) continue;
        *w = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = b->b, .descriptorCount = 1};
        if (b->desc == 'u') {
            w->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w->pBufferInfo = &bi[nw];
            bi[nw] = (VkDescriptorBufferInfo){g_zero, 0, 65536};
            if ((b->k == BK_SUPPORT)) bi[nw].buffer = g_support;
            else if (b->k == BK_MESH_FIRST) {
                // The active proof-gated mesh bridge supplies one ring-owned
                // 256-byte/aligned UBO only to the owner stage.  With the
                // gate closed this path is unreachable and zero stays safe.
                bi[nw] = g_mesh_first_valid[st] ? g_mesh_first_desc[st] :
                         (VkDescriptorBufferInfo){g_zero, 0, 256};
            }
            else if ((b->k == BK_C1)) bi[nw] = (VkDescriptorBufferInfo){s->c1, 0, VK_WHOLE_SIZE};
            else if (b->k == BK_CBUF && b->reg == DMSH_PUSH_BANK - 3) { if (!push_fill(st, &bi[nw])) g_miss[M_OTHER]++; }   // c[24]: the NVN2 push-constant block
            else if ((b->k == BK_CBUF)) {
                VkBuffer shb; VkDeviceSize sho; uint64_t usz = b->reg >= 0 && b->reg < 16 ? (S.ubo[st][b->reg].size > 65536 ? 65536 : S.ubo[st][b->reg].size) : 0;
                int ok = b->reg >= 0 && b->reg < 16 && vkr_ub_shadow(S.ubo[st][b->reg].addr, usz, &shb, &sho) ? (bi[nw] = (VkDescriptorBufferInfo){shb, sho, usz}, 1)   // deferred updates over it (vk_replay.c)
                       : b->reg >= 0 && b->reg < 16 && buf_desc(S.ubo[st][b->reg].addr, S.ubo[st][b->reg].size, 64, 65536, &bi[nw]); if (!ok) g_miss[M_UBO]++;
                if (g_trace && b->reg >= 0 && b->reg < 16) { const float *f = (const float *)(uintptr_t)S.ubo[st][b->reg].addr;
                    LOG("[pass]     stage %d cbuf binding %u c%d: addr %#llx size %llu%s; first %g %g %g %g", st, b->b, b->reg, (unsigned long long)S.ubo[st][b->reg].addr, (unsigned long long)S.ubo[st][b->reg].size,
                        ok ? "" : " [NOT BOUND]", f ? f[0] : 0.f, f ? f[1] : 0.f, f ? f[2] : 0.f, f ? f[3] : 0.f);
                    const char *dd = getenv("DM_VK_SBUFDUMP");   // constant buffers too, as c<address>.bin
                    if (dd && f && S.ubo[st][b->reg].size <= 65536) { char path[600]; snprintf(path, sizeof path, "%s/c%llx.bin", dd, (unsigned long long)S.ubo[st][b->reg].addr); FILE *o = fopen(path, "wb"); if (o) { fwrite(f, 1, S.ubo[st][b->reg].size, o); fclose(o); } } } }
            else if ((b->k == BK_DRVCB)) {
                // Ounce's c[24] is the NVN2 push block.  Some shaders expose it
                // as a driver bank rather than an ordinary cbuf, so dmsh_spv.py
                // gives it binding 18; binding an empty synthetic driver bank
                // zeroed the movie YUV matrix and made every converted frame
                // black.  c0/c2 remain the host-built driver tables.
                if (g_trace && b->b == 18) {
                    const float *f = (const float *)vkr_push(st == 5);
                    LOG("[pass]     stage %d driver c24 binding %u: "
                        "%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g",
                        st, b->b,
                        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                        f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                }
                int ok = b->b == 18 ? push_fill(st, &bi[nw]) :
                         drv_fill(st, s->stage, b->b == 17 ? 2 : b->b == 16 ? 0 : -1, &bi[nw]);
                if (!ok) g_miss[M_OTHER]++;
            }
            else g_miss[M_OTHER]++;
        } else if (b->desc == 's') {
            w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w->pBufferInfo = &bi[nw];
            bi[nw] = (VkDescriptorBufferInfo){g_sink, 0, 65536};
            const bbuf *u = (b->k == BK_SBUF) && b->reg >= 0 && b->reg < 16 ? &S.ssbo[st][b->reg] : (b->k == BK_UBO_PTR) && b->reg >= 0 && b->reg < 16 ? &S.ubo[st][b->reg] : NULL;
            int ok = b->k == BK_GMEM_MAP ? gmem_map_fill(&bi[nw]) : b->k == BK_GMEM_LOG ? gmem_log_fill(s, &bi[nw])
                   : u && buf_desc(u->addr, u->size, 16, 0x7fffffff, &bi[nw]); if (!ok) g_miss[M_SBUF]++;
            if (g_trace) { const float *f = u && u->size >= 24 ? (const float *)(uintptr_t)u->addr : NULL;   // what the buffer holds now (a buffer the GPU writes: as last written)
                LOG("[pass]     stage %d sbuf binding %u s%d: addr %#llx size %llu%s; first %g %g %g %g %g %g", st, b->b, b->reg, u ? (unsigned long long)u->addr : 0ull, u ? (unsigned long long)u->size : 0ull,
                    ok ? "" : " [NOT BOUND]", f ? f[0] : 0.f, f ? f[1] : 0.f, f ? f[2] : 0.f, f ? f[3] : 0.f, f ? f[4] : 0.f, f ? f[5] : 0.f);
                const char *dd = getenv("DM_VK_SBUFDUMP");   // DM_VK_SBUFDUMP=<dir> (with DM_VK_PASSLOG): each storage buffer of the traced frame saved as <dir>/<address>.bin, as it is at bind time
                if (dd && f) { char path[600]; snprintf(path, sizeof path, "%s/%llx.bin", dd, (unsigned long long)u->addr); FILE *o = fopen(path, "wb"); if (o) { fwrite(f, 1, u->size, o); fclose(o); } } }
        } else if (b->desc == 't') {
            w->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w->pImageInfo = &ii[nw];
            ii[nw] = (VkDescriptorImageInfo){g_dsamp, g_dview[dummy_index(b)], VK_IMAGE_LAYOUT_GENERAL};
            if (b->cnt > 1) { w->descriptorCount = 1; }   // arrays (tex_c2a): element 0 only for now (open)
            uint64_t h = 0; int tid = 0, sid = 0;
            if ((b->k == BK_TEX) && b->reg >= 0 && b->reg < 256) { h = S.tex[st][b->reg]; tid = (int)(h & 0xfffff); }
            else if ((b->k == BK_TEX_UBO)) { uint32_t hw = ubo_word(st, b->reg, b->off); tid = (int)(hw & 0xfffff); if (tid) tid += S.tex_base; h = hw; }
            if ((b->sk == BS_SAMP_TEX)) sid = (int)((h >> 20) & 0xfff);
            else if ((b->sk == BS_SAMP) && b->sreg >= 0 && b->sreg < 128) sid = (int)(S.samp[st][b->sreg] >> 20);
            else if ((b->sk == BS_SAMP_ID)) sid = b->sreg;
            else if ((b->sk == BS_SAMP_UBO)) { uint32_t hw = ubo_word(st, b->sreg, b->soff); sid = (int)(hw >> 20); }
            if (sid && (b->sk != BS_SAMP_ID) && (b->k == BK_TEX_UBO)) sid += S.samp_base;
            tex_t o; view_t v; int hv;
            vtex *tt = NULL;
            if (!tid) g_miss[M_TEX]++;
            else if (!vkr_treg(tid, &o, &v, &hv)) g_miss[M_TEXREG]++;
            else { vtex *t = tt = vkr_tex_sampled(&o, VK_NULL_HANDLE); VkImageView iv = t ? view_get(t, b, &o, &v, hv) : VK_NULL_HANDLE; if (iv) ii[nw].imageView = iv; else g_miss[M_TEX]++; }
            if (g_trace) LOG("[pass]     stage %d %s binding %u slot %d: texture ID %d serial %llu %ux%u fmt %#x target %d, sampler ID %d; image of serial %llu, %s, storage %p", st, b->kind, b->b, b->reg, tid,
                             tid && vkr_treg(tid, &o, &v, &hv) ? (unsigned long long)o.serial : 0ull, tid ? o.d.width : 0, tid ? o.d.height : 0, tid ? o.d.format : 0, tid ? o.d.target : -1, sid,
                             tt ? (unsigned long long)tt->serial : 0ull, !tt ? "none" : tt->gpu_owned ? "GPU-written" : tt->uploaded ? "loaded" : "never written", tt ? (const void *)tt->kmem : NULL);
            samp_par sp; if (sid && vkr_sreg(sid, &sp)) ii[nw].sampler = sampler_get(&sp); else g_miss[M_SAMP]++;
        } else if (b->desc == 'i') {
            w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w->pImageInfo = &ii[nw];
            ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, g_dview[dummy_index(b)], VK_IMAGE_LAYOUT_GENERAL};
            int tid = b->reg >= 0 && b->reg < 32 ? (int)(S.img[st][b->reg] & 0xfffff) : 0; tex_t o; view_t v; int hv;
            if (tid && vkr_treg(tid, &o, &v, &hv)) { vtex *t = vkr_tex_sampled(&o, VK_NULL_HANDLE); VkImageView iv = t ? image_view(t, b, &v, hv) : VK_NULL_HANDLE;
                if (iv) { ii[nw].imageView = iv; vkr_tex_gpu_write(t); } else g_miss[M_IMG]++;
                if (g_trace) LOG("[pass]     stage %d image binding %u slot %d: texture ID %d serial %llu %ux%u fmt %#x target %d; image of serial %llu", st, b->b, b->reg, tid, (unsigned long long)o.serial, o.d.width, o.d.height, o.d.format, o.d.target, t ? (unsigned long long)t->serial : 0ull);
            } else g_miss[M_IMG]++;
        } else if (b->desc == 'b') { g_miss[M_TBUF]++; continue; }   // open: texel buffers (left unwritten: partially bound)
        else { g_miss[M_OTHER]++; continue; }
        nw++;
    }
    pthread_mutex_unlock(&g_tmu);
    uint64_t h = mix64(0x51ed270b27ull, (uint64_t)(uintptr_t)s);
    for (int i = 0; i < nw; i++) {
        h = mix64(h, (uint64_t)wr[i].dstBinding << 32 | (uint32_t)wr[i].descriptorType);
        if (wr[i].pBufferInfo) { h = mix64(h, (uint64_t)(uintptr_t)wr[i].pBufferInfo->buffer); h = mix64(h, wr[i].pBufferInfo->offset); h = mix64(h, wr[i].pBufferInfo->range); }
        if (wr[i].pImageInfo) { h = mix64(h, (uint64_t)(uintptr_t)wr[i].pImageInfo->sampler); h = mix64(h, (uint64_t)(uintptr_t)wr[i].pImageInfo->imageView); }
    }
    if (!h) h = 1;
    setc_ent *tab = g_setc[g_pool_slot], *e = NULL;
    uint64_t pt1 = g_dprof ? mk_now() : 0; if (g_dprof) g_prof_ns[4] += pt1 - pt0;
    for (int k = 0; k < SETC_PROBES; k++) { setc_ent *x = &tab[(h + (uint64_t)k) % SETC]; if (x->h == h) { g_setc_hit++; g_prof_ns[6]++; return x->set; } if (!x->h) { e = x; break; } }   // e: where a new one goes
    for (;;) {
        if (pc->cur == pc->n) {
            if (pc->n == 256) {
                if (!pc->limit_logged) {
                    pc->limit_logged = 1;
                    LOG("[RED] vulkan draws: descriptor pool chain limit reached (%d pools; sets %llu, cache hits %llu, misses %llu)",
                        pc->n, (unsigned long long)g_st_sets, (unsigned long long)g_setc_hit,
                        (unsigned long long)g_setc_miss);
                }
                return VK_NULL_HANDLE;
            }
            pc->p[pc->n] = pool_new(); pc->used[pc->n] = 0;
            if (!pc->p[pc->n]) return VK_NULL_HANDLE;
            pc->n++;
        }
        if (!pc->p[pc->cur]) {
            LOG("[RED] vulkan draws: descriptor pool %d is unavailable after reset/recreate failure", pc->cur);
            return VK_NULL_HANDLE;
        }
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pc->p[pc->cur], .descriptorSetCount = 1, .pSetLayouts = &g_dsl};
        VkResult r = vkAllocateDescriptorSets(g_vkd.dev, &ai, &set);
        vkx_pool_alloc_action action = vkx_pool_alloc_result(r, pc->used[pc->cur]);
        if (action == VKX_POOL_ALLOC_OK) { pc->used[pc->cur]++; break; }
        if (action == VKX_POOL_ALLOC_ADVANCE) { pc->cur++; continue; }
        LOG("[RED] vulkan draws: vkAllocateDescriptorSets failed: VkResult %d pool %d used %u%s",
            (int)r, pc->cur, (unsigned)pc->used[pc->cur],
            (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) ? " (fresh-pool capacity failure)" : "");
        return VK_NULL_HANDLE;
    }
    for (int i = 0; i < nw; i++) wr[i].dstSet = set;
    if (nw) vkUpdateDescriptorSets(g_vkd.dev, (uint32_t)nw, wr, 0, NULL);
    g_st_sets++; g_sub_sets++; g_setc_miss++; if (g_dprof) { g_prof_ns[5] += mk_now() - pt1; g_prof_ns[7]++; }
    if (e) *e = (setc_ent){h, set};
    return set;
}
// Background precompiler: every shader of the set into the disk cache, in the form a draw will ask for (the GL-depth form for vertex, geometry and tessellation
// evaluation shaders in a GL-depth game), on low-priority threads while the game runs. The first run fills the cache (about a minute on the M1's large cores for
// Mario Wonder's 12,811); after it no shader is compiled in play. DM_VK_PRECOMPILE=<threads> (2 by default), 0 turns it off. In play the threads run on the cores
// of less than the highest capacity when there are any (the M1's Icestorm cores): the process keeps the game on the large ones (darwin_compat.c prefer_big_cores),
// and two compiler threads there took half of Super Smash Bros. Ultimate's four, about half of its frames over 16.7 ms.
// DM_VK_PRECOMPILE_ONLY=1 (vkx_precompile_tool): the precompiler alone, no game, on every core by default, then the process exits; an explicit
// DM_VK_PRECOMPILE=<threads> also bounds tool mode for driver bring-up and bisecting, while keeping the ordinary all-core default.
// (scripts/build_ipa_from_nsp.py --platform linux), so the first run starts with a full cache.
// The depth mode decides a pre-rasterization shader's form, and only the game sets it (NVN's depth mode; Smash sets near = 0 at start, so its vertex shaders are
// plain). The first draw of a run records it in the cache as depth_mode; the tool takes it from there, and without it makes both forms
static _Atomic int g_pc_next, g_pc_done, g_pc_threads; static _Atomic uint64_t g_pc_compiled; static uint64_t g_pc_t0; static int g_pc_tool, g_pc_both, g_pc_depth = -1;
// Tool-mode bring-up can warm an exact comma-separated key set without first
// compiling unrelated known-bad shaders.  The full directory still participates
// in draw_init(), so the resulting binaries retain the production layout hash.
static int pc_key_selected(const char *key) {
    const char *p = getenv("DM_VK_PRECOMPILE_KEYS");
    if (!g_pc_tool || !p || !*p) return 1;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        size_t n = strcspn(p, ", \t");
        if (n == 16 && !memcmp(p, key, 16)) return 1;
        p += n;
    }
    return 0;
}
static void shc_note_depth(void) {   // the first draw: the game's depth mode into the cache, for the tool
    static int noted; if (noted || !g_shc_on) return; noted = 1;
    if (g_pc_depth == g_depth_zero) return;
    char p[1500], t[1600]; snprintf(p, sizeof p, "%s/depth_mode", g_shc); snprintf(t, sizeof t, "%s.%d.tmp", p, (int)getpid());
    FILE *f = fopen(t, "w"); if (!f) return;
    int w = fprintf(f, "%d\n", g_depth_zero) > 0; w = !fclose(f) && w; if (!w || rename(t, p)) unlink(t);
}
static void pc_cores(void) {   // a compiler thread in play: onto the cores of less than the highest capacity, when the CPU has cores of different capacity
    int cap[CPU_SETSIZE], n = 0, best = -1;
    for (int c = 0; c < CPU_SETSIZE; c++) { char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpu_capacity", c);
        FILE *f = fopen(p, "r"); cap[c] = -1; if (f) { if (fscanf(f, "%d", &cap[c]) != 1) cap[c] = -1; fclose(f); }
        if (cap[c] < 0) break; n = c + 1; if (cap[c] > best) best = cap[c]; }
    cpu_set_t set; CPU_ZERO(&set); int k = 0;
    for (int c = 0; c < n; c++) if (cap[c] < best) { CPU_SET(c, &set); k++; }
    if (k) sched_setaffinity(0, sizeof set, &set);
}
static void *precompile_main(void *arg) {
    (void)arg;
    if (!g_pc_tool) { setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10); pc_cores(); }
    for (int i; (i = atomic_fetch_add(&g_pc_next, 1)) < g_nkeys;) {
        const char *key = g_keys[i];
        if (!pc_key_selected(key)) continue;
        vkx_shader_resolved rr;
        if (!vkx_shader_resolve(g_spv, g_mesh_spv, key, -1, &rr)) continue;
        int st = rr.stage;
        if (!vkx_stage_runtime_enabled_on(st, vkd_mesh_active() != NULL)) continue;
        int pre = st == 0 || st == 2 || st == 4 || st == VKX_STAGE_MS;
        int zero = g_pc_tool ? (g_pc_both ? -1 : g_pc_depth) : g_depth_zero;   // zero < 0: explicitly warm both forms
        int forms = !pre ? 1 : zero < 0 ? 3 : zero ? 1 : 2;   // bit 0 the plain form, bit 1 the GL-depth form
        for (int gl = 0; gl < 2; gl++) {
            if (!(forms >> gl & 1) || shc_has(key, st, gl)) continue;
            size_t nw; int st2; uint32_t *spv = sh_code(key, &st2, &nw, NULL, NULL); if (!spv) break;
            int comp = 0; VkShaderEXT o = sh_object(key, st, gl, spv, nw, &comp); free(spv);
            if (o) g_vkd.destroy_shader(g_vkd.dev, o, NULL);
            if (comp) atomic_fetch_add(&g_pc_compiled, 1);
        }
    }
    if (atomic_fetch_add(&g_pc_done, 1) + 1 == atomic_load(&g_pc_threads)) LOG("vulkan draws: shader precompile done: %llu compiled into the cache in %.1f s (%llu shaders in the set)",
        (unsigned long long)atomic_load(&g_pc_compiled), (mk_now() - g_pc_t0) / 1e9, (unsigned long long)g_nkeys);
    return NULL;
}
static void precompile_start(void) {
    const char *e = getenv("DM_VK_PRECOMPILE"); int n = e ? atoi(e) : g_pc_tool ? (int)sysconf(_SC_NPROCESSORS_ONLN) : 2;
    e = getenv("DM_VK_PRECOMPILE_BOTH_FORMS"); g_pc_both = g_pc_tool && e && atoi(e) != 0;
    if (!g_shc_on || n <= 0 || !g_nkeys) return;
    if (n > 8 && !g_pc_tool) n = 8;
    char p[1500]; snprintf(p, sizeof p, "%s/depth_mode", g_shc); FILE *f = fopen(p, "r");
    if (f) { int v; if (fscanf(f, "%d", &v) == 1 && (v == 0 || v == 1)) g_pc_depth = v; fclose(f); }
    if (g_pc_tool) {   // every core: the process limit to the large ones (prefer_big_cores) is lifted for its threads
        cpu_set_t all; CPU_ZERO(&all); for (int c = 0; c < n && c < CPU_SETSIZE; c++) CPU_SET(c, &all); sched_setaffinity(0, sizeof all, &all);
    }
    g_pc_t0 = mk_now(); atomic_store(&g_pc_threads, n);
    for (int i = 0; i < n; i++) { pthread_t t; if (pthread_create(&t, NULL, precompile_main, NULL)) atomic_fetch_sub(&g_pc_threads, 1); else pthread_detach(t); }
    LOG("vulkan draws: precompiling %d shaders into the cache on %d %sthreads%s", g_nkeys, n, g_pc_tool ? "" : "background ",
        g_pc_tool ? (g_pc_both ? " (explicitly both forms of the pre-rasterization shaders)" : g_pc_depth < 0 ? " (depth mode not recorded yet: both forms of the pre-rasterization shaders)" : g_pc_depth ? " (depth mode near = 0)" : " (depth mode GL)") : "");
    if (g_pc_tool && getenv("DM_VK_PRECOMPILE_KEYS")) LOG("vulkan draws: precompile exact-key filter: %s", getenv("DM_VK_PRECOMPILE_KEYS"));
}
// DM_VK_PRECOMPILE_ONLY=1: the shader cache filled with no game (the precompiler above on every core), its progress on stdout as "precompile: <done> / <total>"
// every second, then exit (0 when the set was gone through). Called from main before anything loads (headless.c mtl_pso_prebuild_tool)
int vkx_precompile_tool(void) {
    const char *e = getenv("DM_VK_PRECOMPILE_ONLY"); if (!e || !atoi(e)) return 0;
    g_pc_tool = 1;
    if (!vkd_init(0) || !draw_init()) { LOG("[RED] shader precompile: no Vulkan device or no shader set (see the lines above)"); exit(1); }
    if (!g_shc_on) { LOG("[RED] shader precompile: no shader cache (DM_VK_SHCACHE=0?)"); exit(1); }
    if (!atomic_load(&g_pc_threads)) { LOG("[RED] shader precompile: no threads started"); exit(1); }
    for (;;) {
        int d = atomic_load(&g_pc_next); if (d > g_nkeys) d = g_nkeys;
        printf("precompile: %d / %d\n", d, g_nkeys); fflush(stdout);
        if (atomic_load(&g_pc_done) >= atomic_load(&g_pc_threads)) break;
        usleep(1000000);
    }
    exit(0);
}
// The plain and GL-depth forms of a pre-rasterization shader, made the first time a draw asks (under no lock: the submit thread alone makes them)
static VkShaderEXT sh_plain(vks *s) {
    if (s->sh || !s->spv) return s->sh;
    uint64_t mk0 = mk_now(); int comp;
    s->sh = sh_object(s->key, s->stage, 0, s->spv, s->nspv, &comp);
    if (comp) { atomic_fetch_add(&g_mk_sh, 1); atomic_fetch_add(&g_mk_sh_ns, mk_now() - mk0); }
    if (!s->sh) LOG("[RED] vulkan draws: shader %s (stage %d) not made", s->key, s->stage);
    return s->sh;
}
static VkShaderEXT gl_variant(vks *s) {
    if (s->sh_gl) return s->sh_gl;
    if (s->gl_none) return sh_plain(s);
    uint64_t mk0 = mk_now(); int comp;
    s->sh_gl = s->spv ? sh_object(s->key, s->stage, 1, s->spv, s->nspv, &comp) : VK_NULL_HANDLE;
    if (!s->sh_gl) { s->gl_none = 1; static int w; if (!w++) LOG("[RED] vulkan draws: no GL-depth form of shader %s (no gl_Position this pass can find, or not made): drawn with GL depth uncorrected", s->key); return sh_plain(s); }
    if (comp) { atomic_fetch_add(&g_mk_var, 1); atomic_fetch_add(&g_mk_var_ns, mk_now() - mk0); }
    return s->sh_gl;
}
static vks *program_stage(const prog_t *p, int wanted);

// A draw: the program's shaders and their descriptor sets (bound in c). 0 when the draw cannot be made (no program, a shader missing)
int vkx_prepare_draw(VkCommandBuffer c) {
    if (!draw_init() || g_pool_slot < 0 || !S.prog[0]) return 0;
    shc_note_depth();
    g_st_draws++;
    vks *sh[5] = {0};
    for (int st = 0; st < 5; st++) if (S.prog[st])
        sh[st] = program_stage(S.prog[st], st);
    if (!sh[0]) { g_miss[M_NOSHADER]++; return 0; }
    VkDescriptorSet sets[5] = {0};
    for (int st = 0; st < 5; st++) if (sh[st]) { sets[st] = stage_set(st, sh[st]); if (!sets[st]) return 0; }
    for (int st = 0; st < 5; st++) if (sets[st]) vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, (uint32_t)st, 1, &sets[st], 0, NULL);
    S.frag_out_map = sh[1] ? sh[1]->frag_out_map : 0xffffffffu;
    // Shader-object bindings persist in a command buffer.  Explicitly clear
    // the mutually exclusive task/mesh path when returning to a vertex draw;
    // otherwise Honeykrisp sees the stale mesh raster shader on the following
    // vkCmdDraw* and has no linked ordinary vertex program.  A command-buffer
    // split used to hide this by starting with empty bindings.
    VkShaderStageFlagBits stages[7] = {
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT,
    };
    VkShaderEXT objs[7] = {0}; int last = sh[2] ? 2 : sh[4] ? 4 : 0;
    for (int st = 0; st < 5; st++) objs[st] = !sh[st] ? VK_NULL_HANDLE : st == 0 || st == 2 || st == 4 ? (st == last && !g_depth_zero ? gl_variant(sh[st]) : sh_plain(sh[st])) : sh[st]->sh;   // GL clip depth: the stage that feeds the rasterizer maps it to Vulkan's
    for (int st = 0; st < 5; st++) if (sh[st] && !objs[st]) { g_miss[M_NOSHADER]++; return 0; }
    if (g_trace) {
        g_trace_pre_stage = last;
        g_trace_pre_gl = !g_depth_zero && sh[last]->sh_gl && objs[last] == sh[last]->sh_gl;
        g_trace_pre_fallback = !g_depth_zero && !g_trace_pre_gl;
    }
    g_vkd.bind_shaders(c, sizeof stages / sizeof *stages, stages, objs);
    if (drawlog() && !(g_st_draws & (g_st_draws - 1))) {
        char m[400]; int q = 0; for (int i = 0; i < 10; i++) if (g_miss[i]) q += snprintf(m + q, sizeof m - q, " %s %llu", MISS[i], (unsigned long long)g_miss[i]);
        LOG("vulkan draws: %llu prepared, %llu descriptor sets, %llu shader objects (%llu failed, %llu keys without a translation); missing:%s", (unsigned long long)g_st_draws,
            (unsigned long long)g_st_sets, (unsigned long long)g_st_load, (unsigned long long)g_st_fail, (unsigned long long)g_st_nokey, q ? m : " nothing");
    }
    return 1;
}

static vks *program_stage(const prog_t *p, int wanted) {
    if (!p) return NULL;
    uint8_t k8[8];
    for (int i = 0; i < NST; i++) {
        int actual = -1;
        if (!prog_key_info(p, (unsigned)i, k8, &actual)) break;
        // NVN2 retains authoritative stage identity next to the key. Do not
        // reject a split-bound stage because an unrelated earlier stage of the
        // same program has no translation.
        if (actual >= 0 && actual != wanted) continue;
        vks *s = shader_get(p, i);
        if (s && s->stage == wanted) return s;
    }
    return NULL;
}

// Build the submit-owned first/indirect records before dynamic rendering. The
// arena belongs to the current command-ring slot and is reset only after that
// slot's timeline value completed, so descriptors and indirect commands keep
// their storage for the whole submit without per-draw allocation objects.
static int mesh_bridge_failure(const char *reason, const p_draw *d)
{
    static _Atomic unsigned reported;
    const char *e = getenv("DM_VK_MESH_LOG");
    if (e && atoi(e) && atomic_fetch_add(&reported, 1) < 16)
        LOG("[RED] vulkan mesh bridge: reject=%s indirect=%d first=%d count=%d address=%#llx",
            reason, d && d->ind != 0, d ? d->first : 0, d ? d->count : 0,
            (unsigned long long)(d ? d->ind : 0));
    return 0;
}

// ---- indirect mesh records prepared before their pass ----
// An indirect mesh draw's NVN record {count, first} reaches the Vulkan record {count, 1, 1} and the first word through transfer copies, which cannot be recorded
// inside dynamic rendering. Ending the pass for each one split Ocarina of Time's 683x384 G-buffer into about 20 extra passes a frame (10 ms of tile stores and
// reloads on Honeykrisp G13, DM_VK_PASS_GPU). The replay looks ahead when a pass is about to begin and copies the records of its indirect mesh draws first: up to
// such a draw only draws and state changes run, so nothing on the GPU writes those guest words in between. A draw finds its copy by guest address, in order
#define PB_MAX 128
static struct { uint64_t ind; VkBuffer b; VkDeviceSize base; int used; } g_pb[PB_MAX];
static int g_npb, g_npb_fenced, g_pb_slot = -1; static uint64_t g_pb_gen;
static int pb_valid(void) { return g_npb && g_pb_slot == g_pool_slot && g_pb_gen == g_arena_gen[g_pool_slot]; }
void vkx_mesh_prebridge_reset(void) { g_npb = g_npb_fenced = 0; }
int vkx_mesh_prebridge(VkCommandBuffer c, uint64_t ind) {
    if (g_pool_slot < 0 || !ind || ind > UINT64_MAX - 8) return 0;
    if (g_npb && !pb_valid()) vkx_mesh_prebridge_reset();
    if (g_npb >= PB_MAX) return 0;
    VkBuffer source, scratch; VkDeviceSize source_off, base; VkDeviceAddress gpu; uint8_t *host;
    if (!vkm_lookup((const void *)(uintptr_t)ind, 8, &source, &source_off, &gpu) || source_off > UINT64_MAX - 8) return 0;
    if (!arena_get(512, &scratch, &base, &host)) return 0;
    const uint32_t yz[2] = {1, 1}; memcpy(host + 256 + 4, yz, sizeof yz);   // Y and Z by the host (visible at submission, as a direct draw's record); X is copied
    VkBufferCopy count = {source_off + 0, base + 256, 4}, first = {source_off + 4, base, 4};   // vkx_mesh_draw_indirect: count at +0, first at +4
    vkCmdCopyBuffer(c, source, scratch, 1, &count); vkCmdCopyBuffer(c, source, scratch, 1, &first);
    g_pb[g_npb++] = (typeof(g_pb[0])){ind, scratch, base, 0}; g_pb_slot = g_pool_slot; g_pb_gen = g_arena_gen[g_pool_slot];
    return 1;
}
void vkx_mesh_prebridge_barrier(VkCommandBuffer c) {
    if (g_npb_fenced == g_npb) return;
    g_npb_fenced = g_npb;
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT |
        VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static int pb_find(uint64_t ind) {
    if (!pb_valid()) return -1;
    for (int i = 0; i < g_npb_fenced; i++) if (!g_pb[i].used && g_pb[i].ind == ind) return i;
    return -1;
}
int vkx_mesh_prebridged(uint64_t ind) { return pb_find(ind) >= 0; }

int vkx_mesh_bridge(VkCommandBuffer c, const p_draw *d, vkx_mesh_submit *out) {
    const vkx_mesh_caps *caps = vkd_mesh_active();
    // A prior attempt may have allocated scratch and then failed before issue.
    // Scratch remains submit-owned, but its descriptor lease must not escape.
    vkx_mesh_abort();
    if (!caps || !c || !d || !out || g_pool_slot < 0 || !S.prog[4] || !S.prog[1])
        return mesh_bridge_failure("prerequisite", d);
    vks *ms = program_stage(S.prog[4], VKX_STAGE_MS);
    if (!ms) return mesh_bridge_failure("mesh-stage", d);
    int have_task = ms->mesh_meta.mesh_has_task_known ? ms->mesh_meta.mesh_has_task : S.prog[3] != NULL;
    vks *ts = have_task ? program_stage(S.prog[3], VKX_STAGE_TS) : NULL;
    vks *fs = program_stage(S.prog[1], VKX_STAGE_PS);
    if (have_task && !caps->task) return mesh_bridge_failure("task-capability", d);
    if (have_task && !ts) return mesh_bridge_failure("task-stage", d);
    if (!have_task) ts = NULL; // ignore a stale host-slot-3 task binding for a taskless mesh program
    uint32_t max_groups, max_total;
    if (!vkx_mesh_dispatch_limits(caps, have_task, &max_groups, &max_total))
        return mesh_bridge_failure("dispatch-limits", d);
    vkx_mesh_draw_meta meta;
    if (d->ind) {
        if (!vkx_mesh_indirect_stage_allowed(caps, have_task))
            return mesh_bridge_failure("indirect-stage", d);
        if (!g_vkd.draw_mesh_indirect)
            return mesh_bridge_failure("indirect-entrypoint", d);
        if (!vkx_mesh_draw_indirect(d->ind, max_groups, max_total, &meta))
            return mesh_bridge_failure("indirect-meta", d);
    } else {
        if (!caps->direct_ready) return mesh_bridge_failure("direct-capability", d);
        if (!g_vkd.draw_mesh) return mesh_bridge_failure("direct-entrypoint", d);
        if (d->first < 0 || d->count < 0) return mesh_bridge_failure("direct-negative", d);
        if (!vkx_mesh_draw_direct((uint32_t)d->first, (uint32_t)d->count,
                                  max_groups, max_total, &meta))
            return mesh_bridge_failure("direct-meta", d);
    }

    VkBuffer scratch; VkDeviceSize base; uint8_t *host = NULL;
    size_t bytes = d->ind ? 512 : 256;
    int pb = d->ind ? pb_find(d->ind) : -1;   // copied before the pass began (vkx_mesh_prebridge)
    if (pb >= 0) { scratch = g_pb[pb].b; base = g_pb[pb].base; g_pb[pb].used = 1; }
    else if (!arena_get(bytes, &scratch, &base, &host)) return mesh_bridge_failure("arena", d);
    VkDeviceSize first_off = base, record_off = base + 256;
    vkx_mesh_indirect_layout layout = {0};
    VkBuffer source = VK_NULL_HANDLE; VkDeviceSize source_off = 0; VkDeviceAddress source_gpu;
    if (d->ind) {
        if (!vkm_lookup((const void *)(uintptr_t)d->ind, 8, &source, &source_off, &source_gpu))
            return mesh_bridge_failure("indirect-address", d);
        if (source_off > UINT64_MAX - 8) return mesh_bridge_failure("indirect-offset", d);
        (void)source_gpu;
        layout = (vkx_mesh_indirect_layout){.src_size = source_off + 8, .src_offset = source_off,
            .dst_size = ARENA, .record_offset = record_off, .first_offset = first_off,
            .same_buffer = source == scratch};
    } else {
        memset(host, 0, bytes);
        memcpy(host, &meta.first, 4);
    }

    vkx_mesh_command_plan command;
    if (!vkx_mesh_command_plan_make(&meta, have_task, d->ind ? &layout : NULL, &command))
        return mesh_bridge_failure("command-plan", d);
    int owner = command.owner_stage == VKX_STAGE_TS ? 3 : 4;
    g_mesh_first_desc[owner] = (VkDescriptorBufferInfo){scratch, first_off, 256};
    g_mesh_first_valid[owner] = 1;

    if (d->ind && pb < 0) {
        // Initialize only Y/Z in the transfer domain; X's sole writer is the
        // count copy below.  The writes are disjoint, so no dependency is
        // needed between them and the final barrier covers the whole record.
        const uint32_t yz[2] = {1, 1};
        vkCmdUpdateBuffer(c, scratch, record_off + 4, sizeof yz, yz);
        VkBufferCopy count = {command.bridge.count.src, command.bridge.count.dst, 4};
        VkBufferCopy first = {command.bridge.first.src, command.bridge.first.dst, 4};
        vkCmdCopyBuffer(c, source, scratch, 1, &count);
        vkCmdCopyBuffer(c, source, scratch, 1, &first);
        VkBufferMemoryBarrier barrier[2] = {
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
             .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = scratch, .offset = record_off, .size = sizeof(VkDrawMeshTasksIndirectCommandEXT)},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
             .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = scratch, .offset = first_off, .size = 4},
        };
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | command.owner_dst_stage,
            0, 0, NULL, 2, barrier, 0, NULL);
    }
    // A direct draw's record is written by the host into coherent mapped memory before the command buffer is submitted, which is visible to the GPU
    // without a barrier; recording one here would also be illegal inside the dynamic rendering the draw now stays in (vk_replay.c)
    *out = (vkx_mesh_submit){.command = command,
        .indirect_buffer = d->ind ? scratch : VK_NULL_HANDLE,
        .indirect_offset = d->ind ? record_off : 0,
        .guest_indirect = d->ind};
    if (ts) snprintf(out->task_key, sizeof out->task_key, "%s", ts->key);
    snprintf(out->mesh_key, sizeof out->mesh_key, "%s", ms->key);
    if (fs) snprintf(out->frag_key, sizeof out->frag_key, "%s", fs->key);
    return 1;
}

// Prepare a VS-free native mesh program. Replay can reach this only after the
// exact root proof token makes one validated caps profile active.
static void mesh_prepare_failure(const char *where, unsigned reject,
                                 const vks *ts, const vks *ms, const vks *fs) {
    static _Atomic unsigned reported;
    const char *e = getenv("DM_VK_MESH_LOG");
    if (!e || !atoi(e) || atomic_fetch_add(&reported, 1) >= 16) return;
    LOG("[RED] vulkan mesh prepare: %s reject=%#x ts=%s ms=%s ps=%s "
        "mesh{local=%u,%u,%u payload=%u shared=%u output=%u components=%u vertices=%u primitives=%u} "
        "frag_pp=%#x mesh_pp=%#x",
        where, reject, ts ? ts->key : "-", ms ? ms->key : "-", fs ? fs->key : "-",
        ms ? ms->mesh_meta.local[0] : 0, ms ? ms->mesh_meta.local[1] : 0,
        ms ? ms->mesh_meta.local[2] : 0, ms ? ms->mesh_meta.payload : 0,
        ms ? ms->mesh_meta.shared : 0, ms ? ms->mesh_meta.output_memory : 0,
        ms ? ms->mesh_meta.output_components : 0, ms ? ms->mesh_meta.max_vertices : 0,
        ms ? ms->mesh_meta.max_primitives : 0,
        fs ? fs->mesh_meta.per_primitive_locations : 0,
        ms ? ms->mesh_meta.per_primitive_locations : 0);
}

int vkx_prepare_mesh_program(VkCommandBuffer c, vkx_mesh_program_plan *plan_out) {
    const vkx_mesh_caps *caps = vkd_mesh_active();
    if (!caps || !draw_init() || g_pool_slot < 0 || !S.prog[4] || !S.prog[1]) {
        mesh_prepare_failure("prerequisite", 0, NULL, NULL, NULL); return 0;
    }
    vks *ms = program_stage(S.prog[4], VKX_STAGE_MS);
    int want_task = ms && ms->mesh_meta.mesh_has_task_known ? ms->mesh_meta.mesh_has_task : S.prog[3] != NULL;
    vks *ts = want_task ? program_stage(S.prog[3], VKX_STAGE_TS) : NULL;
    vks *fs = program_stage(S.prog[1], VKX_STAGE_PS);
    vkx_mesh_program_plan plan = vkx_mesh_program(caps, ts ? &ts->mesh_meta : NULL,
                                                  ms ? &ms->mesh_meta : NULL,
                                                  fs ? &fs->mesh_meta : NULL);
    if (plan_out) *plan_out = plan;
    if (plan.reject) { mesh_prepare_failure("program", plan.reject, ts, ms, fs); return 0; }

    // Resolve every shader object before consuming per-submission descriptor
    // sets.  An inadmissible mesh module can otherwise burn three unique sets
    // per attempted draw (the mesh_first owner block is draw-local) until the
    // whole chain is exhausted, even though no command can be issued.
    shc_note_depth();
    VkShaderEXT mesh_obj = g_depth_zero ? sh_plain(ms) : gl_variant(ms);
    VkShaderEXT task_obj = ts ? ts->sh : VK_NULL_HANDLE;
    if (!mesh_obj || !fs->sh || (ts && !task_obj)) {
        mesh_prepare_failure("shader-object", 0, ts, ms, fs); return 0;
    }

    VkDescriptorSet tset = ts ? stage_set(3, ts) : VK_NULL_HANDLE;
    VkDescriptorSet mset = stage_set(4, ms), fset = stage_set(1, fs);
    if ((ts && !tset) || !mset || !fset) {
        mesh_prepare_failure("descriptor-set", 0, ts, ms, fs); return 0;
    }
    if (tset) vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, 3, 1, &tset, 0, NULL);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, 4, 1, &mset, 0, NULL);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pl_gfx, 1, 1, &fset, 0, NULL);
    VkShaderStageFlagBits stages[] = {
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkShaderEXT objects[] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                             task_obj, mesh_obj, fs->sh};
    g_vkd.bind_shaders(c, sizeof stages / sizeof *stages, stages, objects);
    S.frag_out_map = fs->frag_out_map;
    return 1;
}

// ---- the bound state, from the walker ----
// NVN's BindProgram(program, stage bits) binds per stage: the stages in the bits take that program's shaders (a null program unbinds them), the others stay (as the Mac replay's
// bind_stages). Pokémon FireRed binds its vertex and fragment programs apart; replacing one bound program for every stage left its only draw without a vertex shader, and the screen black
void vkx_bind_program(const prog_t *p, int stages) {
    unsigned host = vkx_program_host_mask((unsigned)stages);
    for (int st = 0; st < NST; st++) if (host >> st & 1) S.prog[st] = p;
}
void vkx_prog_forget(const prog_t *p) { for (int st = 0; st < NST; st++) if (S.prog[st] == p) S.prog[st] = NULL; }
void vkx_bind_buf(int ssbo, const p_bind_buf *b) { if (b->stage >= 0 && b->stage < NST && b->index >= 0 && b->index < 16) (ssbo ? S.ssbo : S.ubo)[b->stage][b->index] = (bbuf){b->addr, b->size}; }
void vkx_bind_handles(int what, const p_bind_handles *b, const uint64_t *h) {   // what: 0 textures, 1 samplers, 2 images
    if (b->stage < 0 || b->stage >= NST || b->first < 0 || b->n <= 0) return;
    int lim = what == 0 ? 256 : what == 1 ? 128 : 32;
    uint64_t *dst = what == 0 ? S.tex[b->stage] : what == 1 ? S.samp[b->stage] : S.img[b->stage];
    for (int k = 0; k < b->n && b->first + k < lim; k++) dst[b->first + k] = h[k];
}
void vkx_set_pool(int tex, int base) { if (tex) S.tex_base = base; else S.samp_base = base; }

// ---- step 3.4c: the draws ----
// The fixed-function state, as the walker records it (NVN state objects are copied by value into the records: nvn_obj.h's bit fields), set as dynamic state before each draw:
// with shader objects every piece of graphics state is dynamic, so there are no pipelines to make or cache
typedef struct {
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    float vp[16][6]; int nvp; float dn, df; int has_dr; int sc[4], has_sc;
    int st_vmask[2], st_wmask[2], st_ref[2];
    float pof, pou, poc; float bc[4]; int clamp, restart, discard;
    va_st attr[16]; vs_st stream[16]; NVNbufferRange vbuf[16];
    int origin_ll, depth_zero;
} vkx_dyn_st;
#define VKX_DYN_INIT {.df = 1, .chmask = {0xffffffff}, .poly = {.cull = 0, .front = 1, .mode = 2}, .st_vmask = {0xff, 0xff}, .st_wmask = {0xff, 0xff}}
static vkx_dyn_st g_dyn[VKX_NQ] = {[0 ... VKX_NQ - 1] = VKX_DYN_INIT}; static vkx_dyn_st *g_dynp = &g_dyn[0];
#define D (*g_dynp)
// The queue whose state the replay reads from now on (mtl_queue_activate, before each submission): a slot per queue serial, taken in order of first use; past
// VKX_NQ queues the last slot is shared, logged once. A forgotten queue's slot is reused for the next new one
static struct { uint64_t serial; int used; } g_qslot[VKX_NQ]; static int g_qn;
void vkx_queue_select(uint64_t serial) {
    int i; for (i = 0; i < VKX_NQ; i++) if (g_qslot[i].used && g_qslot[i].serial == serial) break;
    if (i == VKX_NQ) {
        for (i = 0; i < VKX_NQ && g_qslot[i].used; i++) {}
        if (i == VKX_NQ) { static int w; if (!w++) LOG("[RED] vulkan draws: more than %d NVN queues live at once: queue %llu shares the last slot's state", VKX_NQ, (unsigned long long)serial); i = VKX_NQ - 1; }
        else { g_qslot[i] = (typeof(g_qslot[0])){serial, 1}; memset(&g_bind[i], 0, sizeof g_bind[i]); g_dyn[i] = (vkx_dyn_st)VKX_DYN_INIT; if (++g_qn == 2) LOG("vulkan draws: a second NVN queue (%llu): per-queue bound state from here on", (unsigned long long)serial); }
    }
    g_bindp = &g_bind[i]; g_dynp = &g_dyn[i];
}
void vkx_queue_forget(uint64_t serial) { for (int i = 0; i < VKX_NQ; i++) if (g_qslot[i].used && g_qslot[i].serial == serial) { g_qslot[i].used = 0; if (g_bindp == &g_bind[i]) { g_bindp = &g_bind[0]; g_dynp = &g_dyn[0]; } } }
static PFN_vkCmdSetVertexInputEXT p_vin; static PFN_vkCmdSetPolygonModeEXT p_poly; static PFN_vkCmdSetRasterizationSamplesEXT p_rs; static PFN_vkCmdSetSampleMaskEXT p_sm;
static PFN_vkCmdSetAlphaToCoverageEnableEXT p_a2c; static PFN_vkCmdSetColorBlendEnableEXT p_cbe; static PFN_vkCmdSetColorBlendEquationEXT p_cbq; static PFN_vkCmdSetColorWriteMaskEXT p_cwm;
static PFN_vkCmdSetLogicOpEnableEXT p_loe; static PFN_vkCmdSetDepthClampEnableEXT p_dce; static PFN_vkCmdSetAlphaToOneEnableEXT p_a21;
static VkPhysicalDeviceFeatures g_feat; static uint64_t g_st_drawn, g_st_skip[8];
enum { SK_QUADS, SK_FORMAT, SK_VBUF, SK_IBUF, SK_INDIRECT, SK_PREP, SK_U8 };
static const char *const SKIP[] = {"quads", "vertex format", "vertex buffer", "index buffer", "indirect buffer", "shaders", "8-bit indices"};
static void dyn_init(void) {
    if (p_vin) return;
    #define P(v, n) v = (void *)vkGetDeviceProcAddr(g_vkd.dev, n)
    P(p_vin, "vkCmdSetVertexInputEXT"); P(p_poly, "vkCmdSetPolygonModeEXT"); P(p_rs, "vkCmdSetRasterizationSamplesEXT"); P(p_sm, "vkCmdSetSampleMaskEXT");
    P(p_a2c, "vkCmdSetAlphaToCoverageEnableEXT"); P(p_cbe, "vkCmdSetColorBlendEnableEXT"); P(p_cbq, "vkCmdSetColorBlendEquationEXT"); P(p_cwm, "vkCmdSetColorWriteMaskEXT");
    P(p_loe, "vkCmdSetLogicOpEnableEXT"); P(p_dce, "vkCmdSetDepthClampEnableEXT"); P(p_a21, "vkCmdSetAlphaToOneEnableEXT");
    #undef P
    vkGetPhysicalDeviceFeatures(g_vkd.pd, &g_feat);
}
void vkx_bind_state(const p_bind_state *s) {
    switch (s->kind) {
    case 0: { blend_st b; memcpy(&b, s->bytes, sizeof b); if (b.target < 8) D.blend[b.target] = b; break; }
    case 1: memcpy(&D.color, s->bytes, sizeof D.color); break;
    case 2: memcpy(&D.chmask, s->bytes, sizeof D.chmask); break;
    case 3: memcpy(&D.ms, s->bytes, sizeof D.ms); break;
    case 4: memcpy(&D.poly, s->bytes, sizeof D.poly); break;
    case 5: memcpy(&D.ds, s->bytes, sizeof D.ds); break;
    }
}
void vkx_viewports(const p_viewports *v) { for (int i = 0; i < v->n && v->first + i < 16; i++) memcpy(D.vp[v->first + i], v->v[i], sizeof D.vp[0]); if (v->first + v->n > D.nvp) D.nvp = v->first + v->n > 16 ? 16 : v->first + v->n; }
int vkx_scissor_get(int r[4]) { memcpy(r, D.sc, sizeof D.sc); return D.has_sc; }   // the scissor rectangle, and whether one was set
uint32_t vkx_chmask(void) { return D.chmask.mask; }
void vkx_scissor(const p_scissor *s) { D.sc[0] = s->x; D.sc[1] = s->y; D.sc[2] = s->w; D.sc[3] = s->h; D.has_sc = 1; }
void vkx_depth_range(const p_depth_range *r) { D.dn = r->n; D.df = r->f; D.has_dr = 1; }
void vkx_stencil(const p_stencil *s) {
    for (int f = 0; f < 2; f++) if (s->face & (f ? 2 : 1)) { if (s->which == 0) D.st_vmask[f] = s->value_mask; else if (s->which == 1) D.st_wmask[f] = s->mask; else D.st_ref[f] = s->ref; }
}
void vkx_poly_offset(const p_poly_offset *p) { D.pof = p->factor; D.pou = p->units; D.poc = p->clamp; }
void vkx_blend_color(const p_blend_color *b) { memcpy(D.bc, b->rgba, sizeof D.bc); }
void vkx_depth_clamp(int on) { D.clamp = on; }
// SetRasterizerDiscard: draws made while it is on rasterize nothing. Their only output on hardware is transform feedback, which this backend does not capture, so they are skipped,
// as nvn_mtl.m does without a capture. Drawn, Mario Kart 8 Deluxe's discarded geometry showed in its race scene as a dark box around the player's kart
void vkx_rast_discard(int on) { D.discard = on; }
void vkx_prim_restart(int on) { D.restart = on; }
void vkx_vattr(const p_bind_vattr *a) { for (int i = 0; i < a->n && a->first + i < 16; i++) memcpy(&D.attr[a->first + i], &a->st[i], sizeof(va_st)); }
void vkx_vstream(const p_bind_vstream *s) { for (int i = 0; i < s->n && s->first + i < 16; i++) D.stream[s->first + i] = s->st[i]; }
void vkx_vbuf(const p_bind_vbuf *b) { for (int i = 0; i < b->n && b->first + i < 16; i++) D.vbuf[b->first + i] = b->r[i]; }
void mtl_set_origin_lower_left(int on) { D.origin_ll = on; }
int vkx_origin_ll(void) { return D.origin_ll; }   // lower-left origin (NVN's default, GL): the window image is stored bottom-up, so presenting flips it
void mtl_set_depth_zero(int on) { D.depth_zero = on; g_depth_zero = on; }

static VkBlendFactor bfac(int f) {
    switch (f) {
    case 0x1: return VK_BLEND_FACTOR_ZERO; case 0x2: return VK_BLEND_FACTOR_ONE; case 0x3: return VK_BLEND_FACTOR_SRC_COLOR; case 0x4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x5: return VK_BLEND_FACTOR_SRC_ALPHA; case 0x6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 0x7: return VK_BLEND_FACTOR_DST_ALPHA; case 0x8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x9: return VK_BLEND_FACTOR_DST_COLOR; case 0xA: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 0xB: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp bop(int e) { return e == 2 ? VK_BLEND_OP_SUBTRACT : e == 3 ? VK_BLEND_OP_REVERSE_SUBTRACT : e == 4 ? VK_BLEND_OP_MIN : e == 5 ? VK_BLEND_OP_MAX : VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }   // NVN NEVER 1 … ALWAYS 8 = Vulkan 0 … 7
static VkStencilOp sop(int o) { static const VkStencilOp m[9] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_CLAMP,
    VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP}; return o >= 0 && o <= 8 ? m[o] : VK_STENCIL_OP_KEEP; }
// A vertex attribute's format: the texture table's where it has one, else the 3-component and the integer-read-as-float (scaled) layouts Metal had no texture format for
static VkFormat vfmt(int f) {
    switch (f) {
    case 0x19: return VK_FORMAT_R8G8B8_UNORM; case 0x1A: return VK_FORMAT_R8G8B8_SNORM; case 0x1B: return VK_FORMAT_R8G8B8_UINT; case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT; case 0x1E: return VK_FORMAT_R16G16B16_UNORM; case 0x1F: return VK_FORMAT_R16G16B16_SNORM; case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT; case 0x22: return VK_FORMAT_R32G32B32_SFLOAT; case 0x23: return VK_FORMAT_R32G32B32_UINT; case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x51: return VK_FORMAT_R8_USCALED; case 0x52: return VK_FORMAT_R8_SSCALED; case 0x53: return VK_FORMAT_R16_USCALED; case 0x54: return VK_FORMAT_R16_SSCALED;
    case 0x57: return VK_FORMAT_R8G8_USCALED; case 0x58: return VK_FORMAT_R8G8_SSCALED; case 0x59: return VK_FORMAT_R16G16_USCALED; case 0x5A: return VK_FORMAT_R16G16_SSCALED;
    case 0x5D: return VK_FORMAT_R8G8B8_USCALED; case 0x5E: return VK_FORMAT_R8G8B8_SSCALED; case 0x5F: return VK_FORMAT_R16G16B16_USCALED; case 0x60: return VK_FORMAT_R16G16B16_SSCALED;
    case 0x63: return VK_FORMAT_R8G8B8A8_USCALED; case 0x64: return VK_FORMAT_R8G8B8A8_SSCALED; case 0x65: return VK_FORMAT_R16G16B16A16_USCALED; case 0x66: return VK_FORMAT_R16G16B16A16_SSCALED;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32; case 0x6A: return VK_FORMAT_A2B10G10R10_SINT_PACK32; case 0x6B: return VK_FORMAT_A2B10G10R10_USCALED_PACK32; case 0x6C: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
    default: return vkf_from_nvn(f, NULL, NULL);
    }
}
static VkPrimitiveTopology topo(int p, int *ok) {
    *ok = 1;
    switch (p) {
    case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case 2: case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;   // a line loop drawn open (open)
    case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; case 5: case 8: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // a quad strip has the triangle strip's vertex order
    case 6: case 9: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;   // a convex polygon as a fan
    case 0xA: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case 0xB: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case 0xC: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case 0xD: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; case 0xE: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    case 7: *ok = 2; return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // quads: two triangles each, through g_quad_ib (vkx_draw)
    default: *ok = 0; return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}
// Quads, which Vulkan has not: each quad i drawn as the triangles 4i+{0,1,2} and 4i+{0,2,3}, the order the Mac replay uses, from a fixed index buffer of QUAD_MAX quads; the draw's
// first vertex is the indexed draw's vertex offset. Unity's blits are quads (Pokémon Shining Pearl: half of all draws, its bloom, the scene's composite and the upscale to the window:
// without them its characters were never on screen and the picture stayed 1280x720 in a corner). Open: indexed quads (the guest's indices would need expanding) and indirect quads
enum { QUAD_MAX = 65536 };
static VkBuffer g_quad_ib; static VkDeviceMemory g_quad_m;
static int quad_ib(void) {
    if (g_quad_ib) return 1;
    uint32_t *ix = malloc(sizeof *ix * 6 * QUAD_MAX); if (!ix) return 0;
    for (uint32_t q = 0; q < QUAD_MAX; q++) { uint32_t *t = ix + 6 * q, v = 4 * q; t[0] = v; t[1] = v + 1; t[2] = v + 2; t[3] = v; t[4] = v + 2; t[5] = v + 3; }
    buffer_with(ix, sizeof *ix * 6 * QUAD_MAX, &g_quad_ib, &g_quad_m); free(ix);
    return g_quad_ib != VK_NULL_HANDLE;
}
// Everything a draw needs set, then the draw (inside the render pass vk_replay.c began). ncolor / rt_h: the render pass's colour attachments and height
static void set_state(VkCommandBuffer c, int ncolor, uint32_t rt_w, uint32_t rt_h, VkPrimitiveTopology tp, int samples) {
    VkViewport vp[16]; int nvp = D.nvp ? D.nvp : 1;
    for (int i = 0; i < nvp; i++) {
        float x = D.nvp ? D.vp[i][0] : 0, y = D.nvp ? D.vp[i][1] : 0, w = D.nvp ? D.vp[i][2] : (float)rt_w, h = D.nvp ? D.vp[i][3] : (float)rt_h;
        float n = D.dn, f = D.df;   // SetDepthRange(s), NVN's default 0..1 until then: the viewport records' near / far are never filled (0..0 put every fragment at depth 0; Pokémon Shining Pearl's characters were not drawn)
        if (w <= 0) w = 1;
        // NVN upper-left origin (D3D, Metal): NDC +1 is the target's first row, Vulkan puts -1 there: a negative height flips it. Lower-left (GL) already agrees with Vulkan
        vp[i] = D.origin_ll ? (VkViewport){x, y, w, h, n, f} : (VkViewport){x, y + h, w, -h, n, f};
    }
    VkRect2D sc[16];
    for (int i = 0; i < nvp; i++) {
        if (D.has_sc && D.sc[2] > 0 && D.sc[3] > 0) {
            // Both origins as given: lower-left keeps GL's rows bottom-up in memory (the viewport above is not flipped either), so the scissor's y is already a memory row
            // (flipping it against the target height left Hollow Knight's 1280x720 picture in a 1920x1080 window texture only rows 360..720)
            sc[i] = (VkRect2D){{D.sc[0] < 0 ? 0 : D.sc[0], D.sc[1] < 0 ? 0 : D.sc[1]}, {(uint32_t)D.sc[2], (uint32_t)D.sc[3]}};
        } else sc[i] = (VkRect2D){{0, 0}, {rt_w, rt_h}};
    }
    vkCmdSetViewportWithCount(c, (uint32_t)nvp, vp); vkCmdSetScissorWithCount(c, (uint32_t)nvp, sc);
    vkCmdSetRasterizerDiscardEnable(c, VK_FALSE);
    p_rs(c, (VkSampleCountFlagBits)(samples > 1 ? samples : 1));
    VkSampleMask sm = 0xffffffff; p_sm(c, (VkSampleCountFlagBits)(samples > 1 ? samples : 1), &sm);
    p_a2c(c, D.ms.a2c ? VK_TRUE : VK_FALSE); if (g_feat.alphaToOne) p_a21(c, VK_FALSE);
    p_poly(c, D.poly.mode == 0 ? VK_POLYGON_MODE_POINT : D.poly.mode == 1 && g_feat.fillModeNonSolid ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL);
    vkCmdSetLineWidth(c, 1.0f);
    static int nocull = -1; if (nocull < 0) nocull = getenv("DM_VK_NOCULL") != NULL;   // diagnostic: no face culling
    vkCmdSetCullMode(c, nocull ? VK_CULL_MODE_NONE : D.poly.cull == 1 ? VK_CULL_MODE_FRONT_BIT : D.poly.cull == 2 ? VK_CULL_MODE_BACK_BIT : D.poly.cull == 3 ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE);
    // Facing: with the upper-left origin the flipped viewport puts NVN's picture on the same memory rows Vulkan does, so the winding agrees as given; with GL's
    // lower-left origin NVN judges winding in y-up window coordinates while Vulkan's framebuffer runs y down: it turns over (the title screen's final quad, culled
    // as a back face, came out black when the upper-left case was the one turned)
    int ccw = D.poly.front == 1; if (D.origin_ll) ccw = !ccw;
    vkCmdSetFrontFace(c, ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    VkCompareOp dop = cmpop((int)D.ds.df);
    vkCmdSetDepthTestEnable(c, D.ds.dt); vkCmdSetDepthWriteEnable(c, D.ds.dw); vkCmdSetDepthCompareOp(c, dop);
    vkCmdSetDepthBoundsTestEnable(c, VK_FALSE); vkCmdSetDepthBounds(c, 0, 1);
    vkCmdSetStencilTestEnable(c, D.ds.st);
    vkCmdSetStencilOp(c, VK_STENCIL_FACE_FRONT_BIT, sop((int)D.ds.fail_f), sop((int)D.ds.zpass_f), sop((int)D.ds.zfail_f), cmpop((int)D.ds.sf_f));
    vkCmdSetStencilOp(c, VK_STENCIL_FACE_BACK_BIT, sop((int)D.ds.fail_b), sop((int)D.ds.zpass_b), sop((int)D.ds.zfail_b), cmpop((int)D.ds.sf_b));
    for (int f = 0; f < 2; f++) { VkStencilFaceFlags fl = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        vkCmdSetStencilCompareMask(c, fl, (uint32_t)D.st_vmask[f]); vkCmdSetStencilWriteMask(c, fl, (uint32_t)D.st_wmask[f]); vkCmdSetStencilReference(c, fl, (uint32_t)D.st_ref[f]); }
    int bias = D.poly.offset_en != 0; vkCmdSetDepthBiasEnable(c, bias); vkCmdSetDepthBias(c, D.pou, g_feat.depthBiasClamp ? D.poc : 0, D.pof);
    vkCmdSetPrimitiveTopology(c, tp); vkCmdSetPrimitiveRestartEnable(c, D.restart ? VK_TRUE : VK_FALSE);
    if (g_feat.depthClamp) p_dce(c, D.clamp ? VK_TRUE : VK_FALSE);
    if (g_feat.logicOp) p_loe(c, VK_FALSE);   // open: NVN logic ops
    vkCmdSetBlendConstants(c, D.bc);
    if (ncolor) {
        VkBool32 en[8]; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8];
        for (int i = 0; i < ncolor; i++) {
            const blend_st *b = &D.blend[i];
            en[i] = (D.color.blend_en >> i) & 1;
            eq[i] = (VkColorBlendEquationEXT){bfac((int)b->src_rgb), bfac((int)b->dst_rgb), bop((int)b->eq_rgb), bfac((int)b->src_a), bfac((int)b->dst_a), bop((int)b->eq_a)};
            if (!b->src_rgb) eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};   // never set: NVN's default
            // The fixed-function NVN mask and the fragment SPH output map
            // both apply.  Vulkan otherwise writes undefined values for
            // components the translated shader has no Store for, while the
            // original GPU leaves those attachment components untouched.
            wm[i] = vkx_fragment_write_mask(D.chmask.mask, S.frag_out_map, (unsigned)i);   // r 1, g 2, b 4, a 8 as Vulkan's bits
        }
        p_cbe(c, 0, (uint32_t)ncolor, en); p_cbq(c, 0, (uint32_t)ncolor, eq); p_cwm(c, 0, (uint32_t)ncolor, wm);
    }
}

// Record the final command only after replay began dynamic rendering.  The
// bridge and both barriers were already recorded outside it.  With the proof
// token unset vkd_mesh_active() is NULL, so neither preparation nor a function
// pointer call is reachable.
int vkx_draw_mesh(VkCommandBuffer c, const vkx_mesh_submit *s,
                  int ncolor, uint32_t rt_w, uint32_t rt_h, int samples) {
    const vkx_mesh_caps *caps = vkd_mesh_active();
    int ok = 0;
    if (!caps || !s) goto out;
    vkx_mesh_program_plan program;
    if (!vkx_prepare_mesh_program(c, &program)) goto out;
    dyn_init();
    set_state(c, ncolor, rt_w, rt_h, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, samples);
    if (s->command.indirect) {
        if (!vkx_mesh_indirect_stage_allowed(caps, s->command.have_task) ||
            !g_vkd.draw_mesh_indirect || !s->indirect_buffer) goto out;
        g_vkd.draw_mesh_indirect(c, s->indirect_buffer, s->indirect_offset, 1,
                                 sizeof(VkDrawMeshTasksIndirectCommandEXT));
    } else {
        if (!caps->direct_ready || !g_vkd.draw_mesh) goto out;
        g_vkd.draw_mesh(c, s->command.group_count, 1, 1);
    }
    ok = 1;
out:
    // A descriptor lease belongs to exactly one attempted mesh command.  The
    // arena bytes remain alive to timeline completion if a command was issued.
    vkx_mesh_abort();
    if (!ok) return 0;
    g_st_drawn++;
    return 1;
}
// Vertex input: the streams and attributes as bound (all of them: the shader takes what it reads), and the streams' buffers. 0 when a buffer is not in GPU-visible memory
static int set_vertex(VkCommandBuffer c) {
    VkVertexInputBindingDescription2EXT bd[16]; VkVertexInputAttributeDescription2EXT ad[16]; uint32_t nb = 0, na = 0; uint16_t used = 0;
    for (int i = 0; i < 16; i++) {
        const va_st *a = &D.attr[i]; if (!a->fmt) continue;
        VkFormat f = vfmt((int)a->fmt); if (!f) { g_st_skip[SK_FORMAT]++; continue; }
        static int8_t vok[256];   // 1 the device fetches the format, -1 it does not (logged once)
        if (a->fmt < 256 && !vok[a->fmt]) {
            VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vkd.pd, f, &fp);
            vok[a->fmt] = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : -1;
            if (vok[a->fmt] < 0) LOG("[RED] vulkan draws: vertex format NVN %#x (vk %d) is not a vertex format on this device", (unsigned)a->fmt, (int)f);
        }
        ad[na++] = (VkVertexInputAttributeDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = (uint32_t)i, .binding = a->stream, .format = f, .offset = a->offset};
        used |= (uint16_t)(1u << a->stream);
    }
    VkBuffer bufs[16]; VkDeviceSize offs[16], sizes[16], strides[16];
    for (int s = 0; s < 16; s++) {
        if (!(used >> s & 1)) continue;
        bd[nb++] = (VkVertexInputBindingDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = (uint32_t)s, .stride = D.stream[s].stride,
            .inputRate = D.stream[s].divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
    }
    p_vin(c, nb, bd, na, ad);
    for (int s = 0; s < 16; s++) {
        if (!(used >> s & 1)) continue;
        VkDeviceAddress g; VkBuffer b; VkDeviceSize o;
        if (!D.vbuf[s].address || !vkm_lookup((const void *)(uintptr_t)D.vbuf[s].address, D.vbuf[s].size ? D.vbuf[s].size : 1, &b, &o, &g)) { g_st_skip[SK_VBUF]++; return 0; }
        // bound one at a time (the streams used need not be contiguous)
        bufs[0] = b; offs[0] = o; sizes[0] = D.vbuf[s].size ? D.vbuf[s].size : VK_WHOLE_SIZE; strides[0] = D.stream[s].stride;
        vkCmdBindVertexBuffers2(c, (uint32_t)s, 1, bufs, offs, sizes, strides);
    }
    return 1;
}
static int lookup_buf(uint64_t addr, uint64_t len, VkBuffer *b, VkDeviceSize *o) { VkDeviceAddress g; return addr && vkm_lookup((const void *)(uintptr_t)addr, len ? len : 1, b, o, &g); }
// A draw record (OP_DRAW / OP_DRAW_IDX), inside the render pass; ncolor, rt_w, rt_h, samples: the pass's colour attachments, size and sample count
int vkx_draw(VkCommandBuffer c, uint32_t op, const void *a, uint32_t size, int ncolor, uint32_t rt_w, uint32_t rt_h, int samples) {
    if (D.discard || !draw_init()) return 0;
    dyn_init();
    p_draw td; p_draw_idx ti; int ok;
    const p_draw *d = op == OP_DRAW ? draw_rec(a, size, &td) : NULL;
    const p_draw_idx *x = op == OP_DRAW_IDX ? draw_idx_rec(a, size, &ti) : NULL;
    VkPrimitiveTopology tp = topo(d ? d->prim : x->prim, &ok);
    if (!ok || (ok == 2 && (x || d->ind || d->count > 4 * QUAD_MAX || !quad_ib()))) { g_st_skip[SK_QUADS]++; return 0; }
    if ((d ? d->mdi_cnt : x->mdi_cnt)) { g_st_skip[SK_INDIRECT]++; return 0; }   // open: indirect-count multi-draws
    // diagnostic: DM_VK_SKIPSH=<shader key>[,...] leaves out the draws whose vertex or pixel shader is one of those (which draw makes which surface: bisect over a pass log's keys)
    { static const char *skip = (const char *)1; if (skip == (const char *)1) skip = getenv("DM_VK_SKIPSH");
      for (int st = 0; skip && st < 2; st++) { uint8_t k[8]; char ks[17];
          if (S.prog[st] && prog_key(S.prog[st], (unsigned)(S.prog[st] == S.prog[0] ? st : 0), k)) { for (int i = 0; i < 8; i++) snprintf(ks + 2 * i, 3, "%02x", k[i]); if (strstr(skip, ks)) return 0; } } }
    if (g_trace) { uint8_t k[8]; char ks[2][17] = {"-", "-"}; for (int st = 0; st < 2; st++) if (S.prog[st] && prog_key(S.prog[st], (unsigned)(S.prog[st] == S.prog[0] ? st : 0), k)) for (int i = 0; i < 8; i++) snprintf(ks[st] + 2 * i, 3, "%02x", k[i]);
        LOG("[pass]   draw%s prim %d count %d instances %d vs %s ps %s, blend %#x mask %#x, depth test %d write %d func %d, stencil %d func %d ref %d, cull %d", x ? " indexed" : "", d ? d->prim : x->prim, d ? d->count : x->count,
            d ? d->instances : x->instances, ks[0], ks[1], D.color.blend_en, D.chmask.mask, (int)D.ds.dt, (int)D.ds.dw, (int)D.ds.df, (int)D.ds.st, (int)D.ds.sf_f, D.st_ref[0], (int)D.poly.cull);
        if (d) LOG("[pass]     draw arguments: first %d, base instance %d, indirect %#llx", d->first, d->base_instance, (unsigned long long)d->ind);
        else LOG("[pass]     indexed arguments: index buffer %#llx, base vertex %d, base instance %d, indirect %#llx", (unsigned long long)x->ibuf, x->base_vertex, x->base_instance, (unsigned long long)x->ind);
        if (D.color.blend_en & 1) LOG("[pass]     blend 0: rgb src %d dst %d eq %d, alpha src %d dst %d eq %d; blend colour %g %g %g %g", (int)D.blend[0].src_rgb, (int)D.blend[0].dst_rgb, (int)D.blend[0].eq_rgb,
            (int)D.blend[0].src_a, (int)D.blend[0].dst_a, (int)D.blend[0].eq_a, D.bc[0], D.bc[1], D.bc[2], D.bc[3]);
        LOG("[pass]     viewports %d: %.1f %.1f %.1f %.1f depth %.2f..%.2f%s; scissor %s %d %d %d %d; front %s; origin %s; depth mode %s; attrs %#x", D.nvp, D.vp[0][0], D.vp[0][1], D.vp[0][2], D.vp[0][3],
            D.dn, D.df, D.has_dr ? " (depth range)" : "", D.has_sc ? "on" : "off", D.sc[0], D.sc[1], D.sc[2], D.sc[3], D.poly.front ? "ccw" : "cw",
            D.origin_ll ? "lower left" : "upper left", D.depth_zero ? "zero" : "gl", ({ unsigned m = 0; for (int i = 0; i < 16; i++) if (D.attr[i].fmt) m |= 1u << i; m; }));
        for (int i = 0; i < 16; i++) if (D.attr[i].fmt) LOG("[pass]     attr %d: NVN format %#x (vk %d) offset %u stream %u; stream stride %u divisor %u, buffer %#llx size %llu", i, (unsigned)D.attr[i].fmt, (int)vfmt((int)D.attr[i].fmt),
            (unsigned)D.attr[i].offset, (unsigned)D.attr[i].stream, D.stream[D.attr[i].stream].stride, D.stream[D.attr[i].stream].divisor, (unsigned long long)D.vbuf[D.attr[i].stream].address, (unsigned long long)D.vbuf[D.attr[i].stream].size); }
    static int slow = -1; if (slow < 0) slow = getenv("DM_VK_SLOWDRAW") != NULL;   // diagnostic: draws whose recording took over 1 ms, by part
    static int prof = -1; if (prof < 0) prof = getenv("DM_VK_PROF") != NULL;   // DM_VK_PROF: the parts' totals (vkx_prof_take)
    uint64_t sd0 = slow || prof ? mk_now() : 0;
    if (!vkx_prepare_draw(c)) { g_st_skip[SK_PREP]++; return 0; }
    if (g_trace) LOG("[pass]     prepared pre-raster stage %d with %s variant%s", g_trace_pre_stage, g_trace_pre_gl ? "GL-depth" : "plain",
        g_trace_pre_fallback ? " (GL-depth requested but unavailable)" : "");
    uint64_t sd1 = slow || prof ? mk_now() : 0;
    set_state(c, ncolor, rt_w, rt_h, tp, samples);
    uint64_t sd1b = prof ? mk_now() : 0;
    if (!set_vertex(c)) return 0;
    uint64_t sd1c = prof ? mk_now() : 0;
    if (prof) { g_prof_ns[0] += sd1 - sd0; g_prof_ns[1] += sd1b - sd1; g_prof_ns[2] += sd1c - sd1b; }
    if (d) {
        if (ok == 2) { vkCmdBindIndexBuffer(c, g_quad_ib, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(c, (uint32_t)d->count / 4 * 6, (uint32_t)(d->instances ? d->instances : 1), 0, (int32_t)d->first, (uint32_t)d->base_instance); }
        else if (d->ind) { VkBuffer b; VkDeviceSize o; if (!lookup_buf(d->ind, 16, &b, &o)) { g_st_skip[SK_INDIRECT]++; return 0; } vkCmdDrawIndirect(c, b, o, 1, 16); }
        else vkCmdDraw(c, (uint32_t)d->count, (uint32_t)(d->instances ? d->instances : 1), (uint32_t)d->first, (uint32_t)d->base_instance);
    } else {
        static const VkIndexType it[3] = {VK_INDEX_TYPE_UINT8_KHR, VK_INDEX_TYPE_UINT16, VK_INDEX_TYPE_UINT32};
        if (x->itype == 0 && !g_vkd.index_u8) { g_st_skip[SK_U8]++; return 0; }   // open: 8-bit indices on a device without them
        VkBuffer b; VkDeviceSize o;
        if (!lookup_buf(x->ibuf, 1, &b, &o)) { g_st_skip[SK_IBUF]++; return 0; }
        vkCmdBindIndexBuffer(c, b, o, it[x->itype > 2 ? 1 : x->itype]);
        if (x->ind) { VkBuffer ib; VkDeviceSize io; if (!lookup_buf(x->ind, 20, &ib, &io)) { g_st_skip[SK_INDIRECT]++; return 0; } vkCmdDrawIndexedIndirect(c, ib, io, 1, 20); }
        else vkCmdDrawIndexed(c, (uint32_t)x->count, (uint32_t)(x->instances ? x->instances : 1), 0, x->base_vertex, (uint32_t)x->base_instance);
    }
    if (prof) g_prof_ns[3] += mk_now() - sd1c;
    if (slow) { uint64_t sd2 = mk_now(); if (sd2 - sd0 > 1000000) LOG("[slowdraw] %.2f ms: shaders + sets %.2f, state + draw %.2f", (sd2 - sd0) / 1e6, (sd1 - sd0) / 1e6, (sd2 - sd1) / 1e6); }
    g_st_drawn++;
    if (drawlog() && !(g_st_drawn & (g_st_drawn - 1))) {
        char m[300]; int q = 0; for (int i = 0; i < 7; i++) if (g_st_skip[i]) q += snprintf(m + q, sizeof m - q, " %s %llu", SKIP[i], (unsigned long long)g_st_skip[i]);
        LOG("vulkan draws: %llu drawn; skipped:%s", (unsigned long long)g_st_drawn, q ? m : " none");
    }
    return 1;
}

// ---- compute dispatches ----
static uint64_t g_st_dispatched, g_st_dskip;

// nn::movie's NDVE shaders contain colour conversion plus a storage-buffer
// enhancement network.  DM_MOVIE_ENHANCE=0 keeps colour conversion, replaces
// the network's final one-texture/one-image dispatch with a Vulkan copy/blit,
// and only then permits the next frame's intermediate stages to be omitted.
// A refused copy disables the substitution and restores the full network.
static vkx_movie_state g_movie_state;
static uint64_t g_movie_copied;
static int movie_enh_off(void) {
    static int value = -1;
    if (value >= 0) return value;
    const char *e = getenv("DM_MOVIE_ENHANCE");
    value = e && !strcmp(e, "0");
    if (e && strcmp(e, "0") && strcmp(e, "1")) LOG("[RED] vulkan draws: DM_MOVIE_ENHANCE=%s: expected 1 or 0; keeping movie enhancement", e);
    if (value) LOG("vulkan draws: movie enhancement off: %d profile shaders, %d reflected network stages; decoded frames replace the network output", DM_GAME_MOVIE_NET_N, g_nmovie_shader);
    return value;
}
static vkx_movie_shader *movie_shader(const char key[17]) {
    for (int i = 0; i < g_nmovie_shader; i++) if (!strcmp(g_movie_shader[i].key, key)) return &g_movie_shader[i];
    return NULL;
}
static int movie_texture_id(const vkbind *b) {
    if (b->k == BK_TEX && b->reg >= 0 && b->reg < 256) return (int)(S.tex[5][b->reg] & 0xfffff);
    if (b->k == BK_TEX_UBO) {
        uint32_t h = ubo_word(5, b->reg, b->off); int id = (int)(h & 0xfffff);
        return id ? id + S.tex_base : 0;
    }
    return 0;
}
static int movie_copy(VkCommandBuffer c, const vkx_movie_shader *m,
                      uint32_t *width, uint32_t *height,
                      VkFormat *src_format, VkFormat *dst_format) {
    int sid = movie_texture_id(&m->input);
    int did = m->output.reg >= 0 && m->output.reg < 32 ? (int)(S.img[5][m->output.reg] & 0xfffff) : 0;
    if (!sid || !did) return 0;
    tex_t so = {0}, dso = {0}; view_t sv = {0}, dv = {0}; int shv = 0, dhv = 0;
    pthread_mutex_lock(&g_tmu);
    int ok = vkr_treg(sid, &so, &sv, &shv) && vkr_treg(did, &dso, &dv, &dhv);
    vtex *src = ok ? vkr_tex_sampled(&so, VK_NULL_HANDLE) : NULL;
    vtex *dst = ok ? vkr_tex_target(&dso) : NULL;
    ok = src && dst && src != dst && src->img != dst->img &&
         src->target == NVN_TEXTURE_TARGET_2D && dst->target == NVN_TEXTURE_TARGET_2D &&
         src->d == 1 && dst->d == 1 && src->layers == 1 && dst->layers == 1 &&
         src->samples == 1 && dst->samples == 1 && src->w == dst->w && src->h == dst->h &&
         (src->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (dst->aspect & VK_IMAGE_ASPECT_COLOR_BIT) &&
         (src->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && (dst->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
         (!shv || (!sv.set_levels && !sv.set_layers && vkr_view_format(src->fmt, &sv, shv) == src->fmt)) &&
         (!dhv || (!dv.set_levels && !dv.set_layers && vkr_view_format(dst->fmt, &dv, dhv) == dst->fmt));
    if (ok && src->fmt == dst->fmt) {
        VkImageCopy copy = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .extent = {src->w, src->h, 1},
        };
        vkCmdCopyImage(c, src->img, VK_IMAGE_LAYOUT_GENERAL, dst->img, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    } else if (ok) {
        VkFormatProperties sf, df; vkGetPhysicalDeviceFormatProperties(g_vkd.pd, src->fmt, &sf); vkGetPhysicalDeviceFormatProperties(g_vkd.pd, dst->fmt, &df);
        ok = (sf.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (df.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
        if (ok) {
            VkImageBlit blit = {
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)src->w, (int32_t)src->h, 1}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)dst->w, (int32_t)dst->h, 1}},
            };
            vkCmdBlitImage(c, src->img, VK_IMAGE_LAYOUT_GENERAL, dst->img, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
        }
    }
    if (ok) {
        vkr_tex_gpu_write(dst); *width = src->w; *height = src->h;
        *src_format = src->fmt; *dst_format = dst->fmt;
    }
    pthread_mutex_unlock(&g_tmu);
    return ok;
}
static int movie_net_dispatch(VkCommandBuffer c, const char key[17], const p_dispatch *p) {
    if (!DM_GAME_MOVIE_NET_N || p->ind || !movie_enh_off()) return 0;
    vkx_movie_shader *m = movie_shader(key); if (!m || m->kind == VKX_MOVIE_NONE) return 0;
    if (m->kind == VKX_MOVIE_STAGE) {
        if (!vkx_movie_skip_stage(&g_movie_state)) return 0;
        return 1;
    }
    uint32_t w = 0, h = 0; VkFormat sf = VK_FORMAT_UNDEFINED, df = VK_FORMAT_UNDEFINED;
    unsigned left = g_movie_state.stages_left;
    int ok = movie_copy(c, m, &w, &h, &sf, &df);
    vkx_movie_copy_result(&g_movie_state, ok);
    if (!ok) {
        if (left) LOG("[RED] vulkan draws: movie network %s copy refused after %u intermediate stages were omitted; this frame is incomplete and the full network is restored", key, left);
        else LOG("[RED] vulkan draws: movie network %s copy refused; the full network remains active", key);
        m->kind = VKX_MOVIE_NONE;
        return 0;
    }
    if (!g_movie_copied++) LOG("vulkan draws: first movie frame without enhancement: %ux%u Vulkan format %d -> %d; later intermediate stages may now be omitted", w, h, (int)sf, (int)df);
    return 1;
}

// Compute shaders left out on purpose (vk_skipcs.h: DM_VK_SKIP_CS exact keys): read once, each selected key logged once. The key is checked before shader_get,
// so this dispatch neither requests a shader object nor reaches the command buffer (background prewarming may still compile it). Unset follows the original path.
static vkx_skipcs g_skipcs; static int g_skipcs_read;
uint64_t vkx_skipcs_count(void) { return g_skipcs.skipped; }
static int skipcs(const uint8_t k8[8]) {
    if (!g_skipcs_read) { g_skipcs_read = 1; int bad = vkx_skipcs_parse(&g_skipcs, getenv("DM_VK_SKIP_CS"));
        if (g_skipcs.on) LOG("vulkan draws: exact compute skip rule: %d key%s", g_skipcs.n, g_skipcs.n == 1 ? "" : "s");
        if (bad) LOG("[RED] vulkan draws: DM_VK_SKIP_CS: %d entries are not 16 hex digits: dropped", bad); }
    if (!g_skipcs.on) return 0;
    char key[17]; for (int i = 0; i < 8; i++) snprintf(key + 2 * i, 3, "%02x", k8[i]);
    if (!vkx_skipcs_match(&g_skipcs, key)) return 0;
    static char seen[VKX_SKIPCS_MAX + 8][17]; static int nseen;
    for (int i = 0; i < nseen; i++) if (!strcmp(seen[i], key)) return 1;
    if (nseen < (int)(sizeof seen / sizeof seen[0])) snprintf(seen[nseen++], 17, "%s", key);
    LOG("vulkan draws: compute shader %s deliberately skipped by exact DM_VK_SKIP_CS; its dispatches are not run", key);
    return 1;
}
// Which compute shaders run, for the DM_VK_OPS line: dispatches per key since the last line (the OPS line is logged at powers of two of the submission count, so the
// heavy section of a run names its kernels without a GPU profiler). Keys enter the table on their first dispatch; 64 is more than a title's live compute set
#define DKEYS 64
static struct { char key[17]; uint64_t n, total; } g_dkeys[DKEYS]; static int g_ndkeys;
static void dkeys_note(const char *key) {
    int i; for (i = 0; i < g_ndkeys; i++) if (!strcmp(g_dkeys[i].key, key)) break;
    if (i == g_ndkeys) { if (g_ndkeys == DKEYS) return; snprintf(g_dkeys[g_ndkeys++].key, 17, "%s", key); }
    g_dkeys[i].n++; g_dkeys[i].total++;
}
// The keys dispatched since the last call, most first, into out (at most n chars); resets the per-interval counts
int vkx_dispatch_keys_log(char *out, size_t n) {
    int q = 0, shown = 0;
    for (;;) {   // selection: the largest remaining interval count
        int best = -1; for (int i = 0; i < g_ndkeys; i++) if (g_dkeys[i].n && (best < 0 || g_dkeys[i].n > g_dkeys[best].n)) best = i;
        if (best < 0 || shown == 10 || q >= (int)n - 32) break;
        q += snprintf(out + q, n - (size_t)q, " %s:%llu", g_dkeys[best].key, (unsigned long long)g_dkeys[best].n); g_dkeys[best].n = 0; shown++;
    }
    for (int i = 0; i < g_ndkeys; i++) g_dkeys[i].n = 0;
    return shown;
}
void vkx_dispatch(VkCommandBuffer c, const p_dispatch *p) {
    static uint64_t why[4];
    #define DSKIP(k) do { g_st_dskip++; why[k]++; if (!(g_st_dskip & (g_st_dskip - 1))) LOG("[RED] vulkan draws: %llu dispatches skipped (no program %llu, no compute shader %llu, indirect buffer %llu, set %llu)", (unsigned long long)g_st_dskip, (unsigned long long)why[0], (unsigned long long)why[1], (unsigned long long)why[2], (unsigned long long)why[3]); return; } while (0)
    if (!draw_init() || g_pool_slot < 0 || !S.prog[5]) DSKIP(0);
    vks *s = NULL; uint8_t k8[8];
    for (int i = 0; i < NST && !s && prog_key(S.prog[5], (unsigned)i, k8); i++) {
        char key[17]; key_hex(k8, key);
        if (movie_net_dispatch(c, key, p)) return;
        if (skipcs(k8)) { g_st_dskip++; return; }
        vks *t = shader_get(S.prog[5], i); if (t && t->stage == 5) s = t;
    }
    if (!s) DSKIP(1);
    if (g_trace) LOG("[pass]   dispatch %s groups %d %d %d%s", s->key, (int)p->x, (int)p->y, (int)p->z, p->ind ? " (indirect)" : "");
    VkBuffer ib = VK_NULL_HANDLE; VkDeviceSize io = 0;
    if (p->ind) {   // DispatchComputeIndirect: the group counts are the GPU's; c0's copy of them reads as written at record time (0 here)
        VkDeviceAddress g; if (!vkm_lookup((const void *)(uintptr_t)p->ind, 12, &ib, &io, &g)) DSKIP(2);
        memset(g_dispatch, 0, sizeof g_dispatch);
    } else { g_dispatch[0] = (uint32_t)p->x; g_dispatch[1] = (uint32_t)p->y; g_dispatch[2] = (uint32_t)p->z; }
    VkDescriptorSet set = stage_set(5, s); if (!set) DSKIP(3);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl_cs, 0, 1, &set, 0, NULL);
    VkShaderStageFlagBits st = VK_SHADER_STAGE_COMPUTE_BIT; g_vkd.bind_shaders(c, 1, &st, &s->sh);
    int gpu_sample = vkr_cs_gpu_begin(c, s->key, (uint32_t)p->x, (uint32_t)p->y,
                                      (uint32_t)p->z, p->ind);
    if (p->ind) vkCmdDispatchIndirect(c, ib, io); else vkCmdDispatch(c, (uint32_t)p->x, (uint32_t)p->y, (uint32_t)p->z);
    vkr_cs_gpu_end(c, gpu_sample);
    g_st_dispatched++; dkeys_note(s->key);
    if (drawlog() && !(g_st_dispatched & (g_st_dispatched - 1))) LOG("vulkan draws: %llu dispatches, %llu skipped", (unsigned long long)g_st_dispatched, (unsigned long long)g_st_dskip);
}
