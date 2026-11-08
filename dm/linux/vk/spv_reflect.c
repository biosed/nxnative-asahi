// Linux build: a small SPIR-V reader for the shaders the translator writes (spv_reflect.h). One pass collects the ids' types, decorations and constants, a second
// classifies each decorated resource variable as a Vulkan descriptor
#include "spv_reflect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OpName = 5, OpEntryPoint = 15, OpExecutionMode = 16, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypeMatrix = 24, OpTypeImage = 25, OpTypeSampler = 26,
       OpTypeSampledImage = 27, OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpVariable = 59, OpDecorate = 71,
       OpMemberDecorate = 72, OpTypeAccelerationStructureKHR = 5341 };
enum { DecBlock = 2, DecBufferBlock = 3, DecArrayStride = 6, DecBuiltIn = 11,
       DecLocation = 30, DecComponent = 31, DecBinding = 33, DecDescriptorSet = 34,
       DecOffset = 35, DecPerPrimitiveEXT = 5271 };
enum { SCUniformConstant = 0, SCUniform = 2, SCOutput = 3, SCWorkgroup = 4,
       SCPushConstant = 9, SCStorageBuffer = 12, SCTaskPayloadWorkgroupEXT = 5402 };
enum { DimBuffer = 5, DimSubpassData = 6 };

typedef struct {
    uint16_t op; uint32_t a[9]; const uint32_t *ins; uint16_t nwords;
    int32_t set, binding, location, component; uint32_t builtin;
    uint8_t block, bufblock, has_builtin, per_primitive;
    uint32_t stride; const char *name;
} ent;

static uint32_t type_size(const ent *e, uint32_t id, uint32_t nid, int depth);
static uint32_t struct_size(const ent *e, uint32_t id, uint32_t nid, int depth, const uint32_t *code, size_t words) {
    // the end of the last member: its Offset decoration plus its size
    uint32_t end = 0, nmem = e[id].nwords - 2; int have_offset = 0;
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xFFFF, n = code[i] >> 16; if (!n) break;
        if (op == OpMemberDecorate && code[i + 1] == id && code[i + 3] == DecOffset && code[i + 2] < nmem) {
            have_offset = 1;
            uint32_t m = code[i + 2], mt = e[id].ins[2 + m], sz = type_size(e, mt, nid, depth + 1);
            if (code[i + 4] + sz > end) end = code[i + 4] + sz;
        }
        i += n;
    }
    if (!have_offset) for (uint32_t m = 0; m < nmem; m++) end += type_size(e, e[id].ins[2 + m], nid, depth + 1);
    return end;
}
static _Thread_local const uint32_t *g_code; static _Thread_local size_t g_words;
static uint32_t type_size(const ent *e, uint32_t id, uint32_t nid, int depth) {
    if (id >= nid || depth > 16) return 0;
    switch (e[id].op) {
    case OpTypeInt: case OpTypeFloat: return e[id].a[0] / 8;
    case OpTypeVector: return e[id].a[1] * type_size(e, e[id].a[0], nid, depth + 1);
    case OpTypeMatrix: return e[id].a[1] * type_size(e, e[id].a[0], nid, depth + 1);
    case OpTypeArray: { uint32_t len = e[id].a[1] < nid && e[e[id].a[1]].op == OpConstant ? e[e[id].a[1]].a[1] : 0;
                        uint32_t st = e[id].stride ? e[id].stride : type_size(e, e[id].a[0], nid, depth + 1); return len * st; }
    case OpTypeStruct: return struct_size(e, id, nid, depth, g_code, g_words);
    default: return 0;
    }
}

static uint32_t output_builtin_bit(uint32_t b) {
    switch (b) {
    case 0: return SPVR_BUILTIN_POSITION;
    case 1: return SPVR_BUILTIN_POINT_SIZE;
    case 3: return SPVR_BUILTIN_CLIP_DISTANCE;
    case 4: return SPVR_BUILTIN_CULL_DISTANCE;
    case 7: return SPVR_BUILTIN_PRIMITIVE_ID;
    case 9: return SPVR_BUILTIN_LAYER;
    case 10: return SPVR_BUILTIN_VIEWPORT_INDEX;
    case 4432: return SPVR_BUILTIN_PRIMITIVE_SHADING_RATE;
    case 5299: return SPVR_BUILTIN_CULL_PRIMITIVE;
    case 5294: case 5295: case 5296: return 0; // required mesh primitive-index arrays, represented by topology
    default: return SPVR_BUILTIN_OTHER;
    }
}

static int output_shape(const ent *e, uint32_t id, uint32_t nid, int depth,
                        uint32_t *bits, uint32_t *components, char *kind) {
    if (id >= nid || depth > 16) return 0;
    switch (e[id].op) {
    case OpTypeBool: *bits = 32; *components = 1; *kind = 'b'; return 1;
    case OpTypeInt: *bits = e[id].a[0]; *components = 1; *kind = e[id].a[1] ? 'i' : 'u'; return 1;
    case OpTypeFloat: *bits = e[id].a[0]; *components = 1; *kind = 'f'; return 1;
    case OpTypeVector: case OpTypeMatrix: case OpTypeArray: {
        uint32_t n = e[id].a[1], b = 0, c = 0; char k = 0;
        if (e[id].op == OpTypeArray) n = n < nid && e[n].op == OpConstant ? e[n].a[1] : 0;
        if (!n || !output_shape(e, e[id].a[0], nid, depth + 1, &b, &c, &k) || c > UINT32_MAX / n) return 0;
        *bits = b; *components = c * n; *kind = k; return 1;
    }
    default: return 0;
    }
}

static int output_add(spvr_info *out, const ent *e, uint32_t type, uint32_t nid,
                      int32_t location, int32_t component, int per_primitive,
                      char *err, size_t errcap) {
    uint32_t bits = 0, components = 0; char kind = 0;
    if (location < 0 || component < 0 || component > UINT16_MAX || location > UINT16_MAX ||
        !output_shape(e, type, nid, 0, &bits, &components, &kind) ||
        components > UINT16_MAX || bits > UINT16_MAX) {
        snprintf(err, errcap, "output location/type shape is not representable"); return -1;
    }
    if (out->noutput == SPVR_MAX_OUTPUT || UINT32_MAX - out->output_components < components) {
        snprintf(err, errcap, "more than %d outputs or component total overflows", SPVR_MAX_OUTPUT); return -1;
    }
    out->output[out->noutput++] = (spvr_output){(uint16_t)location, (uint16_t)component,
        (uint16_t)components, (uint16_t)bits, (uint8_t)per_primitive, kind};
    out->output_components += components;
    return 0;
}

static int output_variable(spvr_info *out, const ent *e, uint32_t id, uint32_t type,
                           uint32_t nid, const uint32_t *code, size_t words,
                           char *err, size_t errcap) {
    uint32_t elem = type;
    if (elem < nid && e[elem].op == OpTypeArray) elem = e[elem].a[0]; // outer mesh vertex/primitive dimension
    if (e[id].has_builtin) { out->output_builtin_mask |= output_builtin_bit(e[id].builtin); return 0; }
    if (elem >= nid || e[elem].op != OpTypeStruct)
        return e[id].location >= 0 ? output_add(out, e, elem, nid, e[id].location, e[id].component,
                                                e[id].per_primitive, err, errcap) : 0;

    uint32_t nmem = e[elem].nwords >= 2 ? e[elem].nwords - 2 : 0;
    int32_t *loc = malloc((nmem ? nmem : 1) * sizeof *loc), *comp = calloc(nmem ? nmem : 1, sizeof *comp);
    uint32_t *builtin = malloc((nmem ? nmem : 1) * sizeof *builtin); uint8_t *has = calloc(nmem ? nmem : 1, 1), *per = calloc(nmem ? nmem : 1, 1);
    if (!loc || !comp || !builtin || !has || !per) { free(loc); free(comp); free(builtin); free(has); free(per); snprintf(err, errcap, "out of memory"); return -1; }
    for (uint32_t m = 0; m < nmem; m++) { loc[m] = -1; builtin[m] = UINT32_MAX; }
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t n = w[0] >> 16, op = w[0] & 0xffff;
        if (!n || i + n > words) break;
        if (op == OpMemberDecorate && n >= 4 && w[1] == elem && w[2] < nmem) {
            uint32_t m = w[2], d = w[3];
            if (d == DecLocation && n >= 5) loc[m] = (int32_t)w[4];
            else if (d == DecComponent && n >= 5) comp[m] = (int32_t)w[4];
            else if (d == DecBuiltIn && n >= 5) { has[m] = 1; builtin[m] = w[4]; }
            else if (d == DecPerPrimitiveEXT) per[m] = 1;
        }
        i += n;
    }
    int rc = 0;
    for (uint32_t m = 0; m < nmem && !rc; m++) {
        if (has[m]) out->output_builtin_mask |= output_builtin_bit(builtin[m]);
        else if (loc[m] >= 0) rc = output_add(out, e, e[elem].ins[2 + m], nid, loc[m], comp[m],
                                              e[id].per_primitive || per[m], err, errcap);
    }
    free(loc); free(comp); free(builtin); free(has); free(per); return rc;
}

int spvr_reflect(const uint32_t *code, size_t words, spvr_info *out, char *err, size_t errcap) {
    memset(out, 0, sizeof *out);
    if (words < 5 || code[0] != 0x07230203) { snprintf(err, errcap, "not SPIR-V"); return -1; }
    uint32_t nid = code[3];
    ent *e = calloc(nid ? nid : 1, sizeof *e); if (!e) { snprintf(err, errcap, "out of memory"); return -1; }
    for (uint32_t i = 0; i < nid; i++) { e[i].set = e[i].binding = e[i].location = -1; e[i].builtin = UINT32_MAX; }
    g_code = code; g_words = words;
    uint32_t entry = 0;
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xFFFF, n = code[i] >> 16;
        if (!n || i + n > words) { snprintf(err, errcap, "bad instruction at word %zu", i); free(e); return -1; }
        const uint32_t *w = code + i;
        switch (op) {
        case OpEntryPoint: if (!entry) { entry = w[2];
            switch (w[1]) { case 0: out->stage = VK_SHADER_STAGE_VERTEX_BIT; break; case 1: out->stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; break;
                            case 2: out->stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; break; case 3: out->stage = VK_SHADER_STAGE_GEOMETRY_BIT; break;
                            case 4: out->stage = VK_SHADER_STAGE_FRAGMENT_BIT; break; case 5: out->stage = VK_SHADER_STAGE_COMPUTE_BIT; break;
                            case 5364: out->stage = VK_SHADER_STAGE_TASK_BIT_EXT; break; case 5365: out->stage = VK_SHADER_STAGE_MESH_BIT_EXT; break; } } break;
        case OpExecutionMode: if (w[1] == entry) {
            if (w[2] == 17 && n >= 6) { out->local_size[0] = w[3]; out->local_size[1] = w[4]; out->local_size[2] = w[5]; } // LocalSize
            else if (w[2] == 26 && n >= 4) out->max_vertices = w[3];
            else if (w[2] == 5270 && n >= 4) out->max_primitives = w[3];
            else if (w[2] == 27) out->output_topology = SPVR_TOPOLOGY_POINTS;
            else if (w[2] == 28 || w[2] == 5269) out->output_topology = SPVR_TOPOLOGY_LINES;
            else if (w[2] == 29 || w[2] == 5298) out->output_topology = SPVR_TOPOLOGY_TRIANGLES;
        } break;
        case OpName: if (w[1] < nid) e[w[1]].name = (const char *)(w + 2); break;
        case OpDecorate:
            if (w[1] >= nid) break;
            if (w[2] == DecBinding) e[w[1]].binding = (int32_t)w[3];
            else if (w[2] == DecDescriptorSet) e[w[1]].set = (int32_t)w[3];
            else if (w[2] == DecBlock) e[w[1]].block = 1;
            else if (w[2] == DecBufferBlock) e[w[1]].bufblock = 1;
            else if (w[2] == DecArrayStride) e[w[1]].stride = w[3];
            else if (w[2] == DecLocation && n >= 4) e[w[1]].location = (int32_t)w[3];
            else if (w[2] == DecComponent && n >= 4) e[w[1]].component = (int32_t)w[3];
            else if (w[2] == DecBuiltIn && n >= 4) { e[w[1]].has_builtin = 1; e[w[1]].builtin = w[3]; }
            else if (w[2] == DecPerPrimitiveEXT) e[w[1]].per_primitive = 1;
            break;
        case OpTypeBool: case OpTypeInt: case OpTypeFloat: case OpTypeVector: case OpTypeMatrix: case OpTypeImage: case OpTypeSampler: case OpTypeSampledImage:
        case OpTypeArray: case OpTypeRuntimeArray: case OpTypeStruct: case OpTypePointer: case OpTypeAccelerationStructureKHR:
            if (w[1] < nid) { e[w[1]].op = (uint16_t)op; e[w[1]].ins = w; e[w[1]].nwords = (uint16_t)n; for (uint32_t k = 0; k < 9 && k + 2 < n; k++) e[w[1]].a[k] = w[2 + k]; } break;
        case OpConstant: if (w[2] < nid) { e[w[2]].op = OpConstant; e[w[2]].a[0] = w[1]; e[w[2]].a[1] = w[3]; } break;   // a[1] = the low word of the value
        case OpVariable: if (w[2] < nid) { e[w[2]].op = OpVariable; e[w[2]].a[0] = w[1]; e[w[2]].a[1] = w[3]; } break;   // a[0] = pointer type, a[1] = storage class
        }
        i += n;
    }
    int rc = 0;
    for (uint32_t id = 0; id < nid && !rc; id++) {
        if (e[id].op != OpVariable) continue;
        uint32_t sc = e[id].a[1], pt = e[id].a[0];
        if (pt >= nid || e[pt].op != OpTypePointer) continue;
        uint32_t t = e[pt].a[1];
        if (sc == SCWorkgroup || sc == SCTaskPayloadWorkgroupEXT || sc == SCOutput) {
            uint32_t z = type_size(e, t, nid, 0), *dst = sc == SCWorkgroup ? &out->shared_size :
                                                       sc == SCTaskPayloadWorkgroupEXT ? &out->task_payload_size : &out->output_size;
            if (UINT32_MAX - *dst < z) { snprintf(err, errcap, "interface storage size overflows"); rc = -1; break; }
            *dst += z;
            if (sc == SCOutput && output_variable(out, e, id, t, nid, code, words, err, errcap)) { rc = -1; break; }
            continue;
        }
        if (sc == SCPushConstant) { uint32_t s = type_size(e, t, nid, 0); if (s > out->push_size) out->push_size = s; continue; }
        if (e[id].binding < 0 || (sc != SCUniformConstant && sc != SCUniform && sc != SCStorageBuffer)) continue;
        uint32_t count = 1;
        for (int d = 0; d < 4 && t < nid && (e[t].op == OpTypeArray || e[t].op == OpTypeRuntimeArray); d++) {
            if (e[t].op == OpTypeRuntimeArray) count = 0;
            else if (count) { uint32_t c = e[t].a[1]; count *= c < nid && e[c].op == OpConstant ? e[c].a[1] : 1; }
            t = e[t].a[0];
        }
        if (t >= nid) continue;
        VkDescriptorType dt;
        if (sc == SCStorageBuffer || (sc == SCUniform && e[t].bufblock)) dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else if (sc == SCUniform) dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (e[t].op == OpTypeSampler) dt = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (e[t].op == OpTypeSampledImage) dt = e[t].a[0] < nid && e[e[t].a[0]].a[1] == DimBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (e[t].op == OpTypeImage) {   // a[1] = Dim, a[5] = Sampled (1 sampled, 2 storage)
            if (e[t].a[1] == DimSubpassData) dt = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            else if (e[t].a[1] == DimBuffer) dt = e[t].a[5] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            else dt = e[t].a[5] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } else if (e[t].op == OpTypeAccelerationStructureKHR) dt = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        else { snprintf(err, errcap, "resource %%%u (%s): type op %u not known", id, e[id].name ? e[id].name : "?", e[t].op); rc = -1; break; }
        if (out->nbind == SPVR_MAX_BIND) { snprintf(err, errcap, "more than %d resources", SPVR_MAX_BIND); rc = -1; break; }
        out->bind[out->nbind++] = (spvr_binding){(uint32_t)(e[id].set < 0 ? 0 : e[id].set), (uint32_t)e[id].binding, count, dt, e[id].name};
    }
    free(e);
    return rc;
}

// ---- spvr_fit ----
enum { OpExtension = 10, OpCapability = 17, OpCopyMemory = 63, OpStore = 62, OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67,
       OpInBoundsPtrAccessChain = 70, OpFunctionCall = 57, OpCopyObject = 83, OpBitcast = 124, OpSelect = 169, OpPhi = 245, OpAtomicFirst = 227, OpAtomicLast = 242, OpBeginInvocationInterlockEXT = 5364, OpEndInvocationInterlockEXT = 5365 };
enum { OpMemoryModel = 14, ModeSignedZeroInfNanPreserve = 4461, CapSignedZeroInfNanPreserve = 4466 };
enum { OpExtInstImport = 11, OpExtInst = 12, GlslFMin = 37, GlslFMax = 40, GlslFClamp = 43, GlslNMin = 79, GlslNMax = 80, GlslNClamp = 81 };
enum { CapFloat64 = 10, CapSampleInterlock = 5363, CapShadingRateInterlock = 5372, CapPixelInterlock = 5378, DecNonWritable = 24 };
uint32_t *spvr_fit(const uint32_t *code, size_t words, const spvr_device *dev, size_t *out_words, unsigned *changes) {
    *changes = 0; *out_words = 0;
    if (words < 5 || code[0] != 0x07230203) return NULL;
    uint32_t nid = code[3];
    // what the module has and uses
    int f64type = 0, interlock_used = 0, model = -1; uint32_t glsl = 0;   // glsl: the GLSL.std.450 import
    uint32_t entries[8]; int nentry = 0, has_nanp = 0;   // entry points; the module already declares SignedZeroInfNanPreserve
    uint8_t *ptr_sc = calloc(nid, 1), *bufblock = calloc(nid, 1), *nonwr = calloc(nid, 1);   // pointer type → storage class + 1; struct ids decorated BufferBlock; NonWritable ids
    uint32_t *ptr_t = calloc(nid, sizeof *ptr_t), *arr = calloc(nid, sizeof *arr), *root = calloc(nid, sizeof *root);   // pointer → pointee; array → element; pointer value → its storage buffer variable
    uint8_t *isbuf = calloc(nid, 1), *written = calloc(nid, 1);
    if (!ptr_sc || !bufblock || !nonwr || !ptr_t || !arr || !root || !isbuf || !written) goto fail;
    for (int pass = 0; pass < 2; pass++)   // twice: an OpPhi can name a pointer defined further down (a loop's back edge)
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16; if (!n || i + n > words) goto fail;
        if (op == OpEntryPoint && !pass && nentry < 8) entries[nentry++] = w[2];
        if (op == OpCapability && w[1] == CapSignedZeroInfNanPreserve) has_nanp = 1;
        if (op == OpEntryPoint && model < 0) model = (int)w[1];
        else if (op == OpExtInstImport && n >= 3 && !strcmp((const char *)(w + 2), "GLSL.std.450")) glsl = w[1];
        else if (op == OpTypeFloat && w[2] == 64) f64type = 1;
        else if (op == OpBeginInvocationInterlockEXT || op == OpEndInvocationInterlockEXT) interlock_used = 1;
        else if (op == OpDecorate && w[1] < nid && w[2] == DecBufferBlock) bufblock[w[1]] = 1;
        else if (op == OpDecorate && w[1] < nid && w[2] == DecNonWritable) nonwr[w[1]] = 1;
        else if (op == OpTypePointer && w[1] < nid) { ptr_sc[w[1]] = (uint8_t)(w[2] + 1); ptr_t[w[1]] = w[3]; }
        else if ((op == OpTypeArray || op == OpTypeRuntimeArray) && w[1] < nid) arr[w[1]] = w[2];
        else if (op == OpVariable && w[2] < nid && w[1] < nid) {
            uint32_t t = ptr_t[w[1]]; for (int k = 0; k < 4 && t < nid && arr[t]; k++) t = arr[t];
            if (w[3] == SCStorageBuffer || (w[3] == SCUniform && t < nid && bufblock[t])) { isbuf[w[2]] = 1; root[w[2]] = w[2]; }
        } else if ((op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain || op == OpInBoundsPtrAccessChain) && w[2] < nid && w[3] < nid && root[w[3]]) root[w[2]] = root[w[3]];
        else if (op == OpStore && w[1] < nid && root[w[1]]) written[root[w[1]]] = 1;
        else if (op == OpCopyMemory && w[1] < nid && root[w[1]]) written[root[w[1]]] = 1;
        else if (op == OpAtomicFirst + 1) { if (w[1] < nid && root[w[1]]) written[root[w[1]]] = 1; }   // OpAtomicStore: pointer first
        else if (op > OpAtomicFirst + 1 && op <= OpAtomicLast && n > 3 && w[3] < nid && root[w[3]]) written[root[w[3]]] = 1;   // read-modify-write atomics: result type, result, pointer (OpAtomicLoad, 227, only reads)
        else if ((op == OpCopyObject || op == OpBitcast) && n > 3 && w[2] < nid && w[3] < nid && root[w[3]]) root[w[2]] = root[w[3]];   // the same pointer under another id
        else if (op == OpSelect && n > 5 && w[2] < nid) { for (uint32_t k = 4; k < 6; k++) if (w[k] < nid && root[w[k]]) { root[w[2]] = root[w[k]]; written[root[w[k]]] = 1; } }   // two buffers may meet: assume written
        else if (op == OpPhi && w[2] < nid) { for (uint32_t k = 3; k + 1 < n; k += 2) if (w[k] < nid && root[w[k]]) { root[w[2]] = root[w[k]]; written[root[w[k]]] = 1; } }
        else if (op == OpFunctionCall) for (uint32_t k = 4; k < n; k++) if (w[k] < nid && root[w[k]]) written[root[w[k]]] = 1;   // a pointer handed to a function: assume written
        i += n;
    }
    int drop_f64 = !dev->float64 && !f64type, drop_il = !dev->interlock && !interlock_used;
    int vertex_stage = model == 0 || model == 1 || model == 2 || model == 3;   // vertex, tessellation, geometry
    uint32_t nmark = 0; int stores = 0;
    if (!dev->vertex_stores && vertex_stage)
        for (uint32_t id = 0; id < nid; id++) if (isbuf[id] && !nonwr[id]) { if (written[id]) stores = 1; else nmark++; }
    int nanp = dev->nan_preserve && !has_nanp && nentry;
    uint32_t *o = malloc((words + 4 * nmark + 16 + 4 * (size_t)nentry) * sizeof *o); if (!o) goto fail;
    size_t k = 0; int marked = 0, nanp_state = 0;   // nanp_state: 1 capability in, 2 extension in, 3 execution modes in
    memcpy(o, code, 5 * sizeof *o); k = 5;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16, drop = 0;
        if (op == OpCapability && drop_f64 && w[1] == CapFloat64) { drop = 1; *changes |= SPVR_FIT_FLOAT64; }
        if ((op == OpDecorate && n >= 3 && w[2] == 0) || (op == 72 && n >= 4 && w[3] == 0)) { drop = 1; *changes |= SPVR_FIT_RELAXED; }   // RelaxedPrecision (decoration 0) on an id or a member
        if (op == OpCapability && drop_il && (w[1] == CapPixelInterlock || w[1] == CapSampleInterlock || w[1] == CapShadingRateInterlock)) { drop = 1; *changes |= SPVR_FIT_INTERLOCK; }
        if (op == OpExtension && drop_il && !strcmp((const char *)(w + 1), "SPV_EXT_fragment_shader_interlock")) { drop = 1; *changes |= SPVR_FIT_INTERLOCK; }
        if (op == OpExecutionMode && drop_il && w[2] >= 5366 && w[2] <= 5371) { drop = 1; *changes |= SPVR_FIT_INTERLOCK; }   // Pixel / Sample / ShadingRate Interlock Ordered / Unordered
        // SignedZeroInfNanPreserve, each part where the module layout wants it: the capability among the capabilities, the extension before the imports and the memory
        // model, an execution mode per entry point after the last entry point. Maxwell keeps NaN and infinities through every operation, and the translated shaders depend on
        // it (their isnan selects, NMin / NMax); without the mode Mesa may assume neither occurs and fold the checks away
        if (nanp && nanp_state == 0 && op == OpCapability) { o[k++] = (2u << 16) | OpCapability; o[k++] = CapSignedZeroInfNanPreserve; nanp_state = 1; }
        if (nanp && nanp_state == 1 && (op == OpExtInstImport || op == OpMemoryModel)) {
            static const char ext[] = "SPV_KHR_float_controls"; uint32_t nw = 1 + (uint32_t)(sizeof ext + 3) / 4;
            o[k] = (nw << 16) | OpExtension; memset(o + k + 1, 0, (nw - 1) * 4); memcpy(o + k + 1, ext, sizeof ext); k += nw; nanp_state = 2;
        }
        if (nanp && nanp_state == 2 && op != OpEntryPoint && op != OpExtInstImport && op != OpMemoryModel && op != OpExtension && op != OpCapability) {
            for (int e = 0; e < nentry; e++) { o[k++] = (4u << 16) | OpExecutionMode; o[k++] = entries[e]; o[k++] = ModeSignedZeroInfNanPreserve; o[k++] = 32; }
            nanp_state = 3; *changes |= SPVR_FIT_NANPRESERVE;
        }
        if (op == OpDecorate && !marked && nmark) {   // the decorations go before the first one
            for (uint32_t id = 0; id < nid; id++) if (isbuf[id] && !nonwr[id] && !written[id]) { o[k++] = (3u << 16) | OpDecorate; o[k++] = id; o[k++] = DecNonWritable; }
            marked = 1; *changes |= SPVR_FIT_NONWRITABLE;
        }
        if (!drop) {
            memcpy(o + k, w, n * sizeof *o);
            // Maxwell's FMNMX returns the other operand when one is NaN, and saturation gives 0 for NaN; GLSL.std.450's FMin / FMax / FClamp leave the result undefined, and
            // Honeykrisp returns the NaN. NMin / NMax / NClamp are defined as Maxwell's (NClamp of NaN is its lower bound). Pokemon Shining Pearl's soft particles take
            // min(exp2(log2(0) * 0), 1) for a particle at the scene's depth: NaN, stored in an R11G11B10 target as its largest value, which the bloom spread over the bedroom.
            // The Mac's libraries do the same with Metal's precise:: functions (docs/reference/shaders.md, min, max and clamp with a NaN operand)
            if (op == OpExtInst && n >= 5 && glsl && w[3] == glsl && (w[4] == GlslFMin || w[4] == GlslFMax || w[4] == GlslFClamp)) {
                o[k + 4] = w[4] == GlslFMin ? GlslNMin : w[4] == GlslFMax ? GlslNMax : GlslNClamp; *changes |= SPVR_FIT_NMINMAX;
            }
            k += n;
        }
        i += n;
    }
    if (stores) *changes |= SPVR_FIT_STORES;
    free(ptr_sc); free(bufblock); free(nonwr); free(ptr_t); free(arr); free(root); free(isbuf); free(written);
    *out_words = k; return o;
fail:
    free(ptr_sc); free(bufblock); free(nonwr); free(ptr_t); free(arr); free(root); free(isbuf); free(written);
    return NULL;
}

// ---- NVN bindless texture/sampler pools ----
uint32_t *spvr_bindless(const uint32_t *code, size_t words, uint32_t set,
                        uint32_t texture_count, size_t *out_words,
                        unsigned *patched) {
    *out_words = 0; *patched = 0;
    if (words < 5 || code[0] != 0x07230203 || !texture_count) return NULL;
    uint32_t nid = code[3];
    uint16_t *op = calloc(nid, sizeof *op);
    uint32_t *a = calloc(nid, sizeof *a), *b = calloc(nid, sizeof *b), *c = calloc(nid, sizeof *c);
    uint32_t *binding = malloc((size_t)nid * sizeof *binding);
    uint8_t *canonical = malloc(nid), *limit_constant = calloc(nid, 1);
    if (!op || !a || !b || !c || !binding || !canonical || !limit_constant) goto fail_bindless;
    memset(binding, 0xff, (size_t)nid * sizeof *binding);
    memset(canonical, 0xff, nid);
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t z = w[0] >> 16, x = w[0] & 0xffff;
        if (!z || i + z > words) goto fail_bindless;
        uint32_t id = UINT32_MAX;
        if ((x == OpTypeImage || x == OpTypeSampler || x == OpTypeArray || x == OpTypePointer) && z >= 2) id = w[1];
        else if ((x == OpConstant || x == OpVariable) && z >= 3) id = w[2];
        if (id < nid) {
            op[id] = (uint16_t)x;
            if (x == OpTypeImage && z >= 7) { a[id] = w[3]; b[id] = w[4]; c[id] = w[5]; } // Dim, Depth, Arrayed
            else if (x == OpTypeArray && z >= 4) { a[id] = w[2]; b[id] = w[3]; }
            else if (x == OpTypePointer && z >= 4) a[id] = w[3];
            else if (x == OpConstant && z >= 4) a[id] = w[3];
            else if (x == OpVariable) a[id] = w[1];
        }
        if (x == OpDecorate && z >= 4 && w[1] < nid && w[2] == DecBinding) binding[w[1]] = w[3];
        i += z;
    }
    for (uint32_t var = 0; var < nid; var++) {
        if (op[var] != OpVariable || (binding[var] < 564 || binding[var] > 566)) continue;
        uint32_t ptr = a[var], arr = ptr < nid && op[ptr] == OpTypePointer ? a[ptr] : UINT32_MAX;
        uint32_t elem = arr < nid && op[arr] == OpTypeArray ? a[arr] : UINT32_MAX;
        uint32_t len = arr < nid && op[arr] == OpTypeArray ? b[arr] : UINT32_MAX;
        if (elem >= nid || len >= nid || op[elem] != OpTypeImage || op[len] != OpConstant || a[len] != 1048576 || a[elem] != 1) continue; // SpvDim2D
        if (b[elem]) canonical[var] = 2;      // depth 2D
        else if (c[elem]) canonical[var] = 1; // 2D array
        else canonical[var] = 0;              // ordinary 2D
        limit_constant[len] = 1; (*patched)++;
    }
    for (uint32_t var = 0; var < nid; var++) {
        if (op[var] != OpVariable || binding[var] != 660) continue;
        uint32_t ptr = a[var], arr = ptr < nid && op[ptr] == OpTypePointer ? a[ptr] : UINT32_MAX;
        uint32_t elem = arr < nid && op[arr] == OpTypeArray ? a[arr] : UINT32_MAX;
        uint32_t len = arr < nid && op[arr] == OpTypeArray ? b[arr] : UINT32_MAX;
        if (elem < nid && len < nid && op[elem] == OpTypeSampler && op[len] == OpConstant && a[len] == 4096) { canonical[var] = 3; (*patched)++; }
    }
    if (!*patched) goto fail_bindless;
    uint32_t *out = malloc(words * sizeof *out); if (!out) goto fail_bindless;
    memcpy(out, code, words * sizeof *out);
    for (size_t i = 5; i < words;) {
        uint32_t *w = out + i; uint32_t z = w[0] >> 16, x = w[0] & 0xffff;
        if (x == OpDecorate && z >= 4 && w[1] < nid && canonical[w[1]] != 0xff) {
            if (w[2] == DecBinding) w[3] = canonical[w[1]];
            else if (w[2] == DecDescriptorSet) w[3] = set;
        } else if (x == OpConstant && z >= 4 && w[2] < nid && limit_constant[w[2]]) w[3] = texture_count;
        i += z;
    }
    free(op); free(a); free(b); free(c); free(binding); free(canonical); free(limit_constant);
    *out_words = words; return out;
fail_bindless:
    free(op); free(a); free(b); free(c); free(binding); free(canonical); free(limit_constant);
    *out_words = 0; *patched = 0; return NULL;
}

// ---- bindless texture access feedback ----
// NX2navi's mature Metal backend avoids materialising Ocarina's entire
// 33,000-entry texture pool by having pool shaders set one bit for the ID they
// actually dereference.  Vulkan needs the same information: descriptor-table
// residency cannot be inferred from registration alone.  Each command-ring
// slot binds a separate host-visible bitmap, so the host reads and clears it
// only after that slot's timeline value has completed.
enum {
    FOpShiftRightLogical = 194, FOpShiftLeftLogical = 196,
    FOpBitwiseAnd = 199, FOpAtomicOr = 241,
};
typedef struct { size_t at; uint32_t index, index_type; } bindless_feedback_patch;

uint32_t *spvr_bindless_feedback(const uint32_t *code, size_t words,
                                 uint32_t bindless_set, uint32_t feedback_set,
                                 size_t *out_words, unsigned *patched) {
    *out_words = 0; *patched = 0;
    if (words < 5 || code[0] != 0x07230203 || !code[3]) return NULL;
    uint32_t bound = code[3], first_fn = 0, first_dec = 0, uint_t = 0;
    uint32_t *set = malloc((size_t)bound * sizeof *set), *binding = malloc((size_t)bound * sizeof *binding);
    uint16_t *op = calloc(bound, sizeof *op); uint32_t *id_type = calloc(bound, sizeof *id_type);
    uint8_t *texture_var = calloc(bound, 1);
    if (!set || !binding || !op || !id_type || !texture_var) goto fail;
    for (uint32_t i = 0; i < bound; i++) set[i] = binding[i] = UINT32_MAX;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t x = w[0] & 0xffff, n = w[0] >> 16;
        if (!n || i + n > words) goto fail;
        if (x == OpDecorate && n >= 4) {
            if (!first_dec) first_dec = (uint32_t)i;
            if (w[1] < bound && w[2] == DecBinding) binding[w[1]] = w[3];
            else if (w[1] < bound && w[2] == DecDescriptorSet) set[w[1]] = w[3];
        } else if (x == OpTypeInt && n >= 4 && w[2] == 32 && !w[3] && !uint_t) uint_t = w[1];
        else if (x == 54 /* OpFunction */ && !first_fn) first_fn = (uint32_t)i;
        if ((x == OpTypeImage || x == OpTypeSampler || x == OpTypeArray || x == OpTypePointer) && n >= 2 && w[1] < bound) op[w[1]] = (uint16_t)x;
        else if ((x == OpConstant || x == OpVariable) && n >= 3 && w[2] < bound) op[w[2]] = (uint16_t)x;
        // All value-producing integer operations used by the translator put
        // result type/id first.  Record only the forms an array index can be.
        switch (x) {
        case 61: /* OpLoad */ case OpCopyObject: case OpBitcast: case OpSelect: case OpPhi:
        case 81: /* OpCompositeExtract */ case 128: /* IAdd */ case 130: /* ISub */
        case 132: /* IMul */ case 134: /* UDiv */ case 135: /* SDiv */
        case 194: case 195: case 196: case 197: case 198: case 199:
            if (n >= 3 && w[2] < bound) id_type[w[2]] = w[1];
            break;
        case 55: /* OpFunctionParameter */
            if (n >= 3 && w[2] < bound) id_type[w[2]] = w[1];
            break;
        default: break;
        }
        i += n;
    }
    if (!first_fn || !first_dec || !uint_t) goto fail;
    for (uint32_t id = 1; id < bound; id++)
        if (op[id] == OpVariable && set[id] == bindless_set && binding[id] <= 2) texture_var[id] = 1;

    bindless_feedback_patch p[512]; unsigned np = 0;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t x = w[0] & 0xffff, n = w[0] >> 16;
        if (!n || i + n > words) goto fail;
        if ((x == OpAccessChain || x == OpInBoundsAccessChain) && n == 5 && w[3] < bound && texture_var[w[3]]) {
            if (np == 512 || w[4] >= bound || !id_type[w[4]]) goto fail;
            p[np++] = (bindless_feedback_patch){i, w[4], id_type[w[4]]};
        }
        i += n;
    }
    if (!np) goto fail;

    uint32_t arr_t = bound++, block_t = bound++, block_ptr_t = bound++, uint_ptr_t = bound++, var = bound++;
    uint32_t c0 = bound++, c1 = bound++, c5 = bound++, c31 = bound++;
    uint32_t ann[] = {
        (4u << 16) | OpDecorate, arr_t, DecArrayStride, 4,
        (3u << 16) | OpDecorate, block_t, DecBlock,
        (5u << 16) | OpMemberDecorate, block_t, 0, DecOffset, 0,
        (4u << 16) | OpDecorate, var, DecBinding, 0,
        (4u << 16) | OpDecorate, var, DecDescriptorSet, feedback_set,
    };
    uint32_t decl[] = {
        (3u << 16) | OpTypeRuntimeArray, arr_t, uint_t,
        (3u << 16) | OpTypeStruct, block_t, arr_t,
        (4u << 16) | OpTypePointer, block_ptr_t, SCStorageBuffer, block_t,
        (4u << 16) | OpTypePointer, uint_ptr_t, SCStorageBuffer, uint_t,
        (4u << 16) | OpConstant, uint_t, c0, 0,
        (4u << 16) | OpConstant, uint_t, c1, 1,
        (4u << 16) | OpConstant, uint_t, c5, 5,
        (4u << 16) | OpConstant, uint_t, c31, 31,
        (4u << 16) | OpVariable, block_ptr_t, var, SCStorageBuffer,
    };
    size_t cap = words + sizeof ann / 4 + sizeof decl / 4 + 32u * np + 16;
    uint32_t *out = malloc(cap * sizeof *out); if (!out) goto fail;
    size_t k = 0; unsigned pi = 0;
#define FPUT(...) do { uint32_t q_[] = {__VA_ARGS__}; size_t nq_ = sizeof q_ / sizeof *q_; memcpy(out + k, q_, nq_ * sizeof *out); k += nq_; } while (0)
    for (size_t i = 0; i < words;) {
        if (i == first_dec) { memcpy(out + k, ann, sizeof ann); k += sizeof ann / 4; }
        if (i == first_fn) { memcpy(out + k, decl, sizeof decl); k += sizeof decl / 4; }
        const uint32_t *w = code + i; uint32_t n = i < 5 ? 5 : w[0] >> 16;
        if (pi < np && i == p[pi].at) {
            uint32_t idx = p[pi].index;
            if (p[pi].index_type != uint_t) { uint32_t cast = bound++; FPUT((4u << 16) | OpBitcast, uint_t, cast, idx); idx = cast; }
            uint32_t word = bound++, bit = bound++, mask = bound++, ptr = bound++, old = bound++;
            FPUT((5u << 16) | FOpShiftRightLogical, uint_t, word, idx, c5);
            FPUT((5u << 16) | FOpBitwiseAnd, uint_t, bit, idx, c31);
            FPUT((5u << 16) | FOpShiftLeftLogical, uint_t, mask, c1, bit);
            FPUT((6u << 16) | OpAccessChain, uint_ptr_t, ptr, var, c0, word);
            FPUT((7u << 16) | FOpAtomicOr, uint_t, old, ptr, c1, c0, mask);
            pi++;
        }
        // Since SPIR-V 1.4, OpEntryPoint's interface must name every global
        // variable used by the entry point, not only Input/Output variables.
        // Without this, Mesa deliberately skips the injected SSBO declaration
        // and rejects its first OpAccessChain as an invalid pointer.
        if (code[1] >= 0x00010400 && (w[0] & 0xffff) == 15 /* OpEntryPoint */) {
            memcpy(out + k, w, n * sizeof *out);
            out[k] = ((n + 1) << 16) | 15u;
            out[k + n] = var;
            k += n + 1;
        } else {
            memcpy(out + k, w, n * sizeof *out); k += n;
        }
        i += n;
    }
#undef FPUT
    out[3] = bound; *out_words = k; *patched = np;
    free(set); free(binding); free(op); free(id_type); free(texture_var);
    return out;
fail:
    free(set); free(binding); free(op); free(id_type); free(texture_var);
    return NULL;
}

// ---- spvr_gl_depth ----
enum { OpTypeVoid2 = 19, OpFunction = 54, OpFunctionEnd = 56, OpReturn = 253, OpLoad = 61, OpCompositeExtract = 81, OpCompositeInsert = 82,
       OpIAdd = 128, OpFAdd = 129, OpFMul = 133, OpControlBarrier = 224,
       OpMemberDecorate2 = 72, BuiltInPosition = 0, BuiltInLocalInvocationIndex = 29,
       ExecMeshEXT = 5365, ExecModeLocalSize = 17, ExecModeOutputVertices = 26 };
uint32_t *spvr_gl_depth(const uint32_t *code, size_t words, size_t *out_words) {
    if (words < 5 || code[0] != 0x07230203) return NULL;
    uint32_t bound = code[3], entry = 0, model = UINT32_MAX, t_float = 0, t_v4 = 0, t_int = 0, t_ptr_v4 = 0, c0 = 0, chalf = 0;
    uint32_t pos_var = 0, pos_struct = 0, pos_member = UINT32_MAX, block_var = 0, pos_array = 0, pos_count_id = 0, pos_count = 0;
    uint32_t pos_var_ptr = 0, pos_pointee = 0; int direct_pos_array = 0;
    uint32_t local_index_var = 0, local_index_ptr = 0, local_index_type = 0, local_size_x = 0, max_vertices = 0;
    size_t first_fn = 0;
    // what is there: the entry point, float / vec4 / int types, an Output vec4 pointer, constants 0 and 0.5, where Position is
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16; if (!n || i + n > words) return NULL;
        if (op == OpEntryPoint && !entry) { model = w[1]; entry = w[2]; }
        else if (op == OpExecutionMode && n >= 4 && w[1] == entry) {
            if (w[2] == ExecModeLocalSize && n >= 6) local_size_x = w[3];
            else if (w[2] == ExecModeOutputVertices) max_vertices = w[3];
        }
        else if (op == OpTypeFloat && w[2] == 32 && !t_float) t_float = w[1];
        else if (op == OpTypeInt && w[2] == 32 && !t_int) t_int = w[1];
        else if (op == OpDecorate && w[2] == DecBuiltIn) {
            if (w[3] == BuiltInPosition) pos_var = w[1];
            else if (w[3] == BuiltInLocalInvocationIndex) local_index_var = w[1];
        }
        else if (op == OpMemberDecorate2 && w[3] == DecBuiltIn && w[4] == BuiltInPosition) { pos_struct = w[1]; pos_member = w[2]; }
        else if (op == OpFunction && !first_fn) first_fn = i;
        i += n;
    }
    for (size_t i = 5; i < words;) {   // second pass: what depends on the first
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
        if (op == OpTypeVector && w[2] == t_float && w[3] == 4 && !t_v4) t_v4 = w[1];
        else if (op == OpConstant && w[1] == t_int && w[3] == 0 && !c0) c0 = w[2];
        else if (op == OpConstant && w[1] == t_float && w[3] == 0x3f000000u && !chalf) chalf = w[2];
        i += n;
    }
    // Mesh translators use either an array of gl_PerVertex structs with a
    // Position member, or a directly decorated Position vec4 array.  The
    // latter is legal SPIR-V too (NX2navi emits it) and must take the same
    // per-vertex shadow path rather than being mistaken for one scalar vec4.
    if (pos_var && model == ExecMeshEXT) {
        for (size_t i = 5; i < words;) {
            const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
            if (op == OpVariable && n >= 4 && w[2] == pos_var && w[3] == SCOutput) { pos_var_ptr = w[1]; break; }
            i += n;
        }
        for (size_t i = 5; i < words && pos_var_ptr; ) {
            const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
            if (op == OpTypePointer && n >= 4 && w[1] == pos_var_ptr && w[2] == SCOutput) { pos_pointee = w[3]; break; }
            i += n;
        }
        for (size_t i = 5; i < words && pos_pointee; ) {
            const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
            if (op == OpTypeArray && n >= 4 && w[1] == pos_pointee && w[2] == t_v4) {
                pos_array = w[1]; pos_count_id = w[3]; direct_pos_array = 1; break;
            }
            i += n;
        }
    }
    if (!pos_array && pos_struct) for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
        if (op == OpTypeArray && w[2] == pos_struct) { pos_array = w[1]; pos_count_id = w[3]; break; }
        i += n;
    }
    if (pos_count_id) for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
        if (op == OpConstant && w[2] == pos_count_id) { pos_count = w[3]; break; }
        i += n;
    }
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
        if (op == OpTypePointer && w[2] == SCOutput && w[3] == t_v4 && t_v4 && !t_ptr_v4) t_ptr_v4 = w[1];
        else if (op == OpVariable && w[2] == local_index_var) local_index_ptr = w[1];
        i += n;
    }
    if (local_index_ptr) for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
        if (op == OpTypePointer && w[1] == local_index_ptr && w[2] == 1) { local_index_type = w[3]; break; }
        i += n;
    }
    int mesh_array = model == ExecMeshEXT && pos_array;
    if (mesh_array && direct_pos_array) block_var = pos_var;
    else if (pos_struct && !pos_var) {   // the Output variable of the block that holds Position
        uint32_t ptr_to_struct = 0;
        for (size_t i = 5; i < words;) { const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
            if (op == OpTypePointer && w[2] == SCOutput && w[3] == (mesh_array ? pos_array : pos_struct)) ptr_to_struct = w[1]; i += n; }
        for (size_t i = 5; i < words && ptr_to_struct;) { const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16;
            if (op == OpVariable && w[1] == ptr_to_struct && w[3] == SCOutput) block_var = w[2]; i += n; }
    }
    if ((!pos_var && !block_var) || !t_float || !t_v4 || !entry || !first_fn) return NULL;
    uint32_t mesh_reps = 0;
    if (mesh_array) {
        if (!block_var || !local_index_var || !local_index_type || !local_size_x || !max_vertices || max_vertices > pos_count || max_vertices % local_size_x) return NULL;
        mesh_reps = max_vertices / local_size_x;
        if (!mesh_reps || mesh_reps > 32) return NULL;
        int in_entry = 0, returns = 0;
        for (size_t i = first_fn; i < words;) { const uint32_t *w = code + i; uint32_t op = w[0] & 0xffff, n = w[0] >> 16;
            if (op == OpFunction) in_entry = w[2] == entry;
            else if (op == OpFunctionEnd) in_entry = 0;
            else if (op == OpReturn && in_entry) returns++;
            i += n;
        }
        // The common-exit barrier makes every output write visible before invocations fix their disjoint vertices.
        if (returns != 1) return NULL;
    }
    // G13 mesh lowering cannot read an Output variable.  Mirror every store
    // to Position into a Workgroup vec4 array while the original program runs,
    // then have the common-exit conversion read that shadow after a barrier.
    // The original output stores remain in place; only the conversion's read
    // is redirected.  Track both whole-vector and scalar component pointers,
    // including access chains formed in helper functions.
    uint32_t original_bound = bound, *ptr_pointee = NULL, *const_value = NULL, *shadow_of = NULL;
    if (mesh_array) {
        ptr_pointee = calloc(original_bound, sizeof *ptr_pointee);
        const_value = malloc((size_t)original_bound * sizeof *const_value);
        shadow_of = calloc(original_bound, sizeof *shadow_of);
        if (!ptr_pointee || !const_value || !shadow_of) { free(ptr_pointee); free(const_value); free(shadow_of); return NULL; }
        for (uint32_t i = 0; i < original_bound; i++) const_value[i] = UINT32_MAX;
        for (size_t i = 5; i < words;) {
            const uint32_t *w = code + i; uint32_t op = w[0] & 0xffff, n = w[0] >> 16;
            if (op == OpTypePointer && n >= 4 && w[1] < original_bound) ptr_pointee[w[1]] = w[3];
            else if (op == OpConstant && n >= 4 && w[2] < original_bound) const_value[w[2]] = w[3];
            i += n;
        }
    }
    // the declarations it needs and lacks, inserted before the first function (after every type and constant)
    uint32_t decl[256], mesh_off[32] = {0}; size_t nd = 0;
    if (!t_int && block_var && !mesh_array) { t_int = bound++; decl[nd++] = (4u << 16) | OpTypeInt; decl[nd++] = t_int; decl[nd++] = 32; decl[nd++] = 1; }
    if (!c0 && block_var && !mesh_array) { c0 = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = t_int; decl[nd++] = c0; decl[nd++] = 0; }
    if (!chalf) { chalf = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = t_float; decl[nd++] = chalf; decl[nd++] = 0x3f000000u; }
    if (!t_ptr_v4) { t_ptr_v4 = bound++; decl[nd++] = (4u << 16) | OpTypePointer; decl[nd++] = t_ptr_v4; decl[nd++] = SCOutput; decl[nd++] = t_v4; }
    uint32_t c_member = 0, c_scope = 0, c_semantics = 0;
    uint32_t t_shadow_array = 0, t_ptr_shadow_array = 0, t_ptr_shadow_v4 = 0, t_ptr_shadow_float = 0, shadow_var = 0;
    if (mesh_array) {
        if (!direct_pos_array) { c_member = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = local_index_type; decl[nd++] = c_member; decl[nd++] = pos_member; }
        c_scope = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = local_index_type; decl[nd++] = c_scope; decl[nd++] = 2;
        c_semantics = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = local_index_type; decl[nd++] = c_semantics; decl[nd++] = 0x108;
        for (uint32_t r = 1; r < mesh_reps; r++) { mesh_off[r] = bound++; decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = local_index_type; decl[nd++] = mesh_off[r]; decl[nd++] = r * local_size_x; }
        t_shadow_array = bound++; decl[nd++] = (4u << 16) | OpTypeArray; decl[nd++] = t_shadow_array; decl[nd++] = t_v4; decl[nd++] = pos_count_id;
        t_ptr_shadow_array = bound++; decl[nd++] = (4u << 16) | OpTypePointer; decl[nd++] = t_ptr_shadow_array; decl[nd++] = SCWorkgroup; decl[nd++] = t_shadow_array;
        t_ptr_shadow_v4 = bound++; decl[nd++] = (4u << 16) | OpTypePointer; decl[nd++] = t_ptr_shadow_v4; decl[nd++] = SCWorkgroup; decl[nd++] = t_v4;
        t_ptr_shadow_float = bound++; decl[nd++] = (4u << 16) | OpTypePointer; decl[nd++] = t_ptr_shadow_float; decl[nd++] = SCWorkgroup; decl[nd++] = t_float;
        shadow_var = bound++; decl[nd++] = (4u << 16) | OpVariable; decl[nd++] = t_ptr_shadow_array; decl[nd++] = shadow_var; decl[nd++] = SCWorkgroup;
    }
    size_t cap = words * 2 + nd + 64 * 64;
    uint32_t *o = malloc(cap * sizeof *o); if (!o) { free(ptr_pointee); free(const_value); free(shadow_of); return NULL; }
    size_t k = 0; int in_entry = 0, nret = 0, shadow_stores = 0;
    for (size_t i = 0; i < words;) {
        if (i == first_fn) { memcpy(o + k, decl, nd * sizeof *o); k += nd; }
        const uint32_t *w = code + i; uint32_t op = i < 5 ? 0 : w[0] & 0xFFFF, n = i < 5 ? 5 : w[0] >> 16;
        if (op == OpFunction) in_entry = w[2] == entry;
        if (op == OpFunctionEnd) in_entry = 0;
        if (op == OpReturn && in_entry) {
            size_t need = mesh_array ? 8 + (size_t)mesh_reps * 56 : 40;
            if (k + need > cap) { while (k + need > cap) cap *= 2; uint32_t *no = realloc(o, cap * sizeof *o); if (!no) { free(o); free(ptr_pointee); free(const_value); free(shadow_of); return NULL; } o = no; }
            uint32_t lid = 0;
            if (mesh_array) {
                o[k++] = (4u << 16) | OpControlBarrier; o[k++] = c_scope; o[k++] = c_scope; o[k++] = c_semantics;
                lid = bound++; o[k++] = (4u << 16) | OpLoad; o[k++] = local_index_type; o[k++] = lid; o[k++] = local_index_var;
            }
            for (uint32_t r = 0; r < (mesh_array ? mesh_reps : 1); r++) {
                uint32_t p = bound++, sp = mesh_array ? bound++ : 0, v = bound++, z = bound++, ww = bound++, s = bound++, hh = bound++, v2 = bound++;
                if (mesh_array) {
                    uint32_t index = lid;
                    if (r) { index = bound++; o[k++] = (5u << 16) | OpIAdd; o[k++] = local_index_type; o[k++] = index; o[k++] = lid; o[k++] = mesh_off[r]; }
                    o[k++] = ((direct_pos_array ? 5u : 6u) << 16) | OpAccessChain; o[k++] = t_ptr_v4; o[k++] = p; o[k++] = block_var; o[k++] = index;
                    if (!direct_pos_array) o[k++] = c_member;
                    o[k++] = (5u << 16) | OpAccessChain; o[k++] = t_ptr_shadow_v4; o[k++] = sp; o[k++] = shadow_var; o[k++] = index;
                } else if (block_var) {
                    o[k++] = (5u << 16) | OpAccessChain; o[k++] = t_ptr_v4; o[k++] = p; o[k++] = block_var; o[k++] = c0;
                    if (pos_member != 0) { free(o); free(ptr_pointee); free(const_value); free(shadow_of); return NULL; }
                } else p = pos_var;
                o[k++] = (4u << 16) | OpLoad; o[k++] = t_v4; o[k++] = v; o[k++] = mesh_array ? sp : p;
                o[k++] = (5u << 16) | OpCompositeExtract; o[k++] = t_float; o[k++] = z; o[k++] = v; o[k++] = 2;
                o[k++] = (5u << 16) | OpCompositeExtract; o[k++] = t_float; o[k++] = ww; o[k++] = v; o[k++] = 3;
                o[k++] = (5u << 16) | OpFAdd; o[k++] = t_float; o[k++] = s; o[k++] = z; o[k++] = ww;
                o[k++] = (5u << 16) | OpFMul; o[k++] = t_float; o[k++] = hh; o[k++] = s; o[k++] = chalf;
                o[k++] = (6u << 16) | OpCompositeInsert; o[k++] = t_v4; o[k++] = v2; o[k++] = hh; o[k++] = v; o[k++] = 2;
                o[k++] = (3u << 16) | OpStore; o[k++] = p; o[k++] = v2;
            }
            nret++;
        }
        if (mesh_array && op == OpEntryPoint && n >= 3 && w[2] == entry) {
            if (k + n + 1 > cap) { while (k + n + 1 > cap) cap *= 2; uint32_t *no = realloc(o, cap * sizeof *o); if (!no) { free(o); free(ptr_pointee); free(const_value); free(shadow_of); return NULL; } o = no; }
            memcpy(o + k, w, n * sizeof *o); o[k] = ((n + 1) << 16) | OpEntryPoint; o[k + n] = shadow_var; k += n + 1; i += n; continue;
        }
        size_t extra = 0; uint32_t shadow_result = 0, shadow_type = 0; size_t first_index = 0;
        if (mesh_array && (op == OpAccessChain || op == OpInBoundsAccessChain) && n >= 5 && w[1] < original_bound && w[2] < original_bound) {
            uint32_t pointee = ptr_pointee[w[1]], base = w[3] < original_bound ? shadow_of[w[3]] : 0;
            if (w[3] == block_var && direct_pos_array && n >= 5) { base = shadow_var; first_index = 4; }
            else if (w[3] == block_var && n >= 6 && w[5] < original_bound && const_value[w[5]] == pos_member) { base = shadow_var; first_index = 4; }
            else if (base) first_index = 4;
            if (base && (pointee == t_v4 || pointee == t_float)) {
                shadow_type = pointee == t_v4 ? t_ptr_shadow_v4 : t_ptr_shadow_float;
                shadow_result = bound++; shadow_of[w[2]] = shadow_result;
                // A root chain drops the Position struct-member index.  A
                // chain based on an already mirrored pointer keeps all indices.
                extra = w[3] == block_var && !direct_pos_array ? n - 1 : n;
            }
        } else if (mesh_array && op == OpStore && n >= 3 && w[1] < original_bound && shadow_of[w[1]]) extra = n;
        if (k + n + extra > cap) { while (k + n + extra > cap) cap *= 2; uint32_t *no = realloc(o, cap * sizeof *o); if (!no) { free(o); free(ptr_pointee); free(const_value); free(shadow_of); return NULL; } o = no; }
        memcpy(o + k, w, n * sizeof *o); k += n; i += n;
        if (shadow_result) {
            size_t q = k; o[k++] = ((uint32_t)extra << 16) | op; o[k++] = shadow_type; o[k++] = shadow_result; o[k++] = shadow_var;
            if (w[3] == block_var && !direct_pos_array) { o[k++] = w[4]; for (size_t x = 6; x < n; x++) o[k++] = w[x]; }
            else if (w[3] == block_var) { for (size_t x = 4; x < n; x++) o[k++] = w[x]; }
            else { o[q + 3] = shadow_of[w[3]]; for (size_t x = first_index; x < n; x++) o[k++] = w[x]; }
        } else if (mesh_array && op == OpStore && extra) {
            memcpy(o + k, w, n * sizeof *o); o[k + 1] = shadow_of[w[1]]; k += n; shadow_stores++;
        }
    }
    if (!nret || (mesh_array && !shadow_stores)) { free(o); free(ptr_pointee); free(const_value); free(shadow_of); return NULL; }
    o[3] = bound;
    free(ptr_pointee); free(const_value); free(shadow_of);
    *out_words = k; return o;
}

// ---- spvr_init_outputs ----
enum { OpTypeFunction2 = 33, OpConstantComposite = 44, OpLabel = 248, OpVariable2 = 59, SCFunction = 7 };
uint32_t *spvr_init_outputs(const uint32_t *code, size_t words, size_t *out_words) {
    if (words < 5 || code[0] != 0x07230203) return NULL;
    uint32_t bound = code[3], entry = 0, model = UINT32_MAX, t_float = 0, t_v4 = 0;
    size_t first_fn = 0;
    uint8_t *builtin = calloc(bound, 1); uint32_t *ptr_sc = calloc(bound, sizeof *ptr_sc), *ptr_t = calloc(bound, sizeof *ptr_t), outs[64]; int nout = 0;
    if (!builtin || !ptr_sc || !ptr_t) goto fail;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16; if (!n || i + n > words) goto fail;
        if (op == OpEntryPoint && !entry) { model = w[1]; entry = w[2]; }
        else if (op == OpTypeFloat && w[2] == 32 && !t_float) t_float = w[1];
        else if (op == OpTypeVector && w[2] == t_float && t_float && w[3] == 4 && !t_v4) t_v4 = w[1];
        else if (op == OpDecorate && w[2] == DecBuiltIn && w[1] < bound) builtin[w[1]] = 1;
        else if (op == OpTypePointer && w[1] < bound) { ptr_sc[w[1]] = w[2] + 1; ptr_t[w[1]] = w[3]; }
        else if (op == OpVariable2 && w[1] < bound && w[2] < bound && w[3] == SCOutput && ptr_t[w[1]] == t_v4 && t_v4 && !builtin[w[2]] && nout < 64) outs[nout++] = w[2];
        else if (op == OpFunction && !first_fn) first_fn = i;
        i += n;
    }
    if ((model != 0 && model != 2) || !nout || !first_fn || !t_v4) goto fail;   // vertex and tessellation evaluation shaders (a geometry shader's outputs are per emitted vertex)
    uint32_t c0 = bound++, c1 = bound++, cv = bound++;
    uint32_t decl[] = {(4u << 16) | OpConstant, t_float, c0, 0, (4u << 16) | OpConstant, t_float, c1, 0x3f800000u, (7u << 16) | OpConstantComposite, t_v4, cv, c0, c0, c0, c1};
    size_t nd = sizeof decl / sizeof *decl;
    uint32_t *o = malloc((words + nd + 3 * (size_t)nout) * sizeof *o); if (!o) goto fail;
    size_t k = 0; int in_entry = 0, state = 0;   // state: 1 after the entry's first OpLabel (its OpVariables follow), 2 once the stores are in
    for (size_t i = 0; i < words;) {
        if (i == first_fn) { memcpy(o + k, decl, sizeof decl); k += nd; }
        const uint32_t *w = code + i; uint32_t op = i < 5 ? 0 : w[0] & 0xFFFF, n = i < 5 ? 5 : w[0] >> 16;
        if (op == OpFunction) in_entry = w[2] == entry;
        if (in_entry && state == 1 && !(op == OpVariable2 && w[3] == SCFunction)) {
            for (int j = 0; j < nout; j++) { o[k++] = (3u << 16) | OpStore; o[k++] = outs[j]; o[k++] = cv; }
            state = 2;
        }
        memcpy(o + k, w, n * sizeof *o); k += n;
        if (in_entry && op == OpLabel && !state) state = 1;
        i += n;
    }
    o[3] = bound;
    free(builtin); free(ptr_sc); free(ptr_t);
    *out_words = k; return o;
fail:
    free(builtin); free(ptr_sc); free(ptr_t);
    return NULL;
}

// ---- spvr_nan ----
// Two places where Maxwell and this GPU part ways on non-finite values (Pokemon Shining Pearl's bedroom: its TV's flicker divides by a speed of 0, the angle becomes
// infinity - infinity, and the colour comes from cos(NaN)):
// - MUFU.SIN / MUFU.COS of NaN or of an infinity give NaN; GLSL.std.450's Sin / Cos leave it undefined, and the driver gave a large finite number, which no NaN check
//   downstream catches. Each float Sin / Cos is made select(isnan(x) || isinf(x), NaN, sin(x)).
// - A NaN written to a float render target stays NaN on Maxwell, and what reads it later (FMNMX, saturation: the bloom's threshold, the tonemapper) drops it. This GPU
//   stores it as the format's largest value (65024 in R11G11B10, 65504 in RGBA16F), which the bloom spread over the bedroom as a white glow. Fragment outputs are
//   written as select(isnan(v), 0, v): what a Maxwell reader using min / max or saturation makes of the NaN. Open: a reader that tests isnan itself sees 0 instead.
// - A NaN output of the vertex, tessellation or geometry stage reaches the fragment shader as NaN on Maxwell; here interpolation turns it into a large finite number
//   (the TV's colour arrived as one, and the fragment shader's own NaN check could not see it). Their user outputs (not the builtins) get the same select: the
//   fragment shader computes from 0 where Maxwell's would have computed NaN and had it stored as 0
enum { OpIsNan = 156, OpIsInf = 157, OpLogicalOr = 166, GlslSin = 13, GlslCos = 14, ExecFragment = 4 };
uint32_t *spvr_nan(const uint32_t *code, size_t words, size_t *out_words, unsigned *changes) {
    *changes = 0; *out_words = 0;
    if (words < 5 || code[0] != 0x07230203) return NULL;
    uint32_t bound = code[3], model = UINT32_MAX, glsl = 0, t_float = 0, t_v4 = 0, t_bool = 0, t_bv4 = 0;
    size_t first_fn = 0, nsin = 0, nstore = 0;
    uint8_t *builtin = calloc(bound, 1), *out_ptr = calloc(bound, 1);   // ids decorated BuiltIn (variables, and blocks with a BuiltIn member: gl_PerVertex); pointers into a user Output variable
    uint32_t *ptr_t = calloc(bound, sizeof *ptr_t), *id_t = calloc(bound, sizeof *id_t), *arr_t = calloc(bound, sizeof *arr_t);   // pointer type → pointee; pointer id → its type; array → element
    uint32_t *o = NULL;
    if (!builtin || !out_ptr || !ptr_t || !id_t || !arr_t) goto done;
    for (int pass = 0; pass < 2; pass++)   // twice: the counts need the types and output pointers the first pass found
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xFFFF, n = w[0] >> 16; if (!n || i + n > words) goto done;
        if (!pass) {
            if (op == OpEntryPoint && model == UINT32_MAX) model = w[1];
            else if (op == OpExtInstImport && n >= 3 && !strcmp((const char *)(w + 2), "GLSL.std.450")) glsl = w[1];
            else if (op == OpTypeFloat && w[2] == 32 && !t_float) t_float = w[1];
            else if (op == OpTypeBool && !t_bool) t_bool = w[1];
            else if (op == OpTypeVector && w[3] == 4 && t_float && w[2] == t_float && !t_v4) t_v4 = w[1];
            else if (op == OpTypeVector && w[3] == 4 && t_bool && w[2] == t_bool && !t_bv4) t_bv4 = w[1];
            else if (op == OpDecorate && w[2] == DecBuiltIn && w[1] < bound) builtin[w[1]] = 1;
            else if (op == OpMemberDecorate2 && n >= 4 && w[3] == DecBuiltIn && w[1] < bound) builtin[w[1]] = 1;   // the struct type
            else if (op == OpTypePointer && w[1] < bound) ptr_t[w[1]] = w[3];
            else if (op == OpVariable2 && w[1] < bound && w[2] < bound) {
                id_t[w[2]] = w[1]; uint32_t pt = ptr_t[w[1]];   // an arrayed output (tessellation control): its element type
                for (int a = 0; a < 2 && pt < bound && arr_t[pt]; a++) pt = arr_t[pt];
                if (w[3] == SCOutput && !builtin[w[2]] && !(pt < bound && builtin[pt])) out_ptr[w[2]] = 1;
            }
            else if ((op == OpTypeArray || op == OpTypeRuntimeArray) && w[1] < bound) arr_t[w[1]] = w[2];
            else if ((op == OpAccessChain || op == OpInBoundsAccessChain) && w[1] < bound && w[2] < bound && w[3] < bound) { id_t[w[2]] = w[1]; if (out_ptr[w[3]]) out_ptr[w[2]] = 1; }
            else if (op == OpFunction && !first_fn) first_fn = i;
        } else if (op == OpExtInst && n >= 6 && glsl && w[3] == glsl && (w[4] == GlslSin || w[4] == GlslCos) && w[1] == t_float) nsin++;
        else if (op == OpStore && model <= ExecFragment && w[1] < bound && out_ptr[w[1]] && id_t[w[1]] < bound) { uint32_t vt = ptr_t[id_t[w[1]]]; if ((vt == t_float && t_float) || (vt == t_v4 && t_v4)) nstore++; }
        i += n;
    }
    if ((!nsin && !nstore) || !first_fn || !t_float) goto done;   // nothing to do: NULL, the caller keeps its code
    // declarations, before the first function: the bool types where missing, 0 and NaN, a vec4 of 0
    uint32_t decl[32]; size_t nd = 0;
    if (!t_bool) { t_bool = bound++; decl[nd++] = (2u << 16) | OpTypeBool; decl[nd++] = t_bool; }
    if (t_v4 && !t_bv4) { t_bv4 = bound++; decl[nd++] = (4u << 16) | OpTypeVector; decl[nd++] = t_bv4; decl[nd++] = t_bool; decl[nd++] = 4; }
    uint32_t c0 = bound++, cnan = bound++, cv0 = t_v4 ? bound++ : 0;
    decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = t_float; decl[nd++] = c0; decl[nd++] = 0;
    decl[nd++] = (4u << 16) | OpConstant; decl[nd++] = t_float; decl[nd++] = cnan; decl[nd++] = 0x7fc00000u;
    if (t_v4) { decl[nd++] = (7u << 16) | OpConstantComposite; decl[nd++] = t_v4; decl[nd++] = cv0; decl[nd++] = c0; decl[nd++] = c0; decl[nd++] = c0; decl[nd++] = c0; }
    o = malloc((words + nd + 25 * nsin + 10 * nstore) * sizeof *o); if (!o) goto done;
    size_t k = 0;
    for (size_t i = 0; i < words;) {
        if (i == first_fn) { memcpy(o + k, decl, nd * sizeof *o); k += nd; }
        const uint32_t *w = code + i; uint32_t op = i < 5 ? 0 : w[0] & 0xFFFF, n = i < 5 ? 5 : w[0] >> 16;
        if (i >= 5 && op == OpExtInst && n >= 6 && glsl && w[3] == glsl && (w[4] == GlslSin || w[4] == GlslCos) && w[1] == t_float) {
            uint32_t x = w[5], r0 = bound++, isn = bound++, isi = bound++, bad = bound++;
            uint32_t seq[] = {(6u << 16) | OpExtInst, t_float, r0, glsl, w[4], x,
                              (4u << 16) | OpIsNan, t_bool, isn, x, (4u << 16) | OpIsInf, t_bool, isi, x, (5u << 16) | OpLogicalOr, t_bool, bad, isn, isi,
                              (6u << 16) | OpSelect, t_float, w[2], bad, cnan, r0};
            memcpy(o + k, seq, sizeof seq); k += sizeof seq / sizeof *seq; *changes |= 1;
        } else if (i >= 5 && op == OpStore && model <= ExecFragment && w[1] < bound && out_ptr[w[1]] && id_t[w[1]] < bound
                   && ((ptr_t[id_t[w[1]]] == t_float && t_float) || (ptr_t[id_t[w[1]]] == t_v4 && t_v4))) {
            int v4 = ptr_t[id_t[w[1]]] == t_v4; uint32_t isn = bound++, sel = bound++;
            uint32_t seq[] = {(4u << 16) | OpIsNan, v4 ? t_bv4 : t_bool, isn, w[2], (6u << 16) | OpSelect, v4 ? t_v4 : t_float, sel, isn, v4 ? cv0 : c0, w[2]};
            memcpy(o + k, seq, sizeof seq); k += sizeof seq / sizeof *seq;
            memcpy(o + k, w, n * sizeof *o); o[k + 2] = sel; k += n;   // the store itself (with any memory-access operands), of the selected value
            *changes |= 2;
        } else { memcpy(o + k, w, n * sizeof *o); k += n; }
        i += n;
    }
    o[3] = bound;
    *out_words = k;
done:
    free(builtin); free(out_ptr); free(ptr_t); free(id_t); free(arr_t);
    if (o && !*out_words) { free(o); o = NULL; }
    return o;
}

// ---- spvr_gmem_trace ----
// The translator's raw-global helper receives address-low, address-high and
// value as pointer parameters. It atomically sets one bit per gmem-map entry.
// The one invocation whose atomic saw the bit clear appends the full guest page
// to a bounded extension after that bitmap. No game resource or reflection
// changes; an unpatched shader keeps the original bitmap ABI.
enum {
    GOpFunctionParameter = 55, GOpIAdd = 128, GOpIEqual = 170,
    GOpShiftRightLogical = 194, GOpShiftLeftLogical = 196,
    GOpBitwiseOr = 197, GOpBitwiseAnd = 199,
    GOpAtomicExchange = 229, GOpAtomicIAdd = 234, GOpAtomicOr = 241,
    GOpSelectionMerge = 247, GOpBranch = 249, GOpBranchConditional = 250,
};
typedef struct {
    size_t at;
    uint32_t addr_lo, addr_hi, atom_t, uint_t, bool_t, ptr_t;
    uint32_t old, bit, scope, sem, log_var;
} gmem_patch;

uint32_t *spvr_gmem_trace(const uint32_t *code, size_t words, uint32_t binding,
                          size_t *out_words, unsigned *patched) {
    *out_words = 0; *patched = 0;
    if (words < 5 || code[0] != 0x07230203 || !code[3]) return NULL;
    uint32_t bound = code[3], log_var = 0, first_fn = 0, t_bool = 0;
    uint32_t *bind = malloc((size_t)bound * sizeof *bind);
    uint32_t *ptr_t = calloc(bound, sizeof *ptr_t), *ptr_sc = calloc(bound, sizeof *ptr_sc);
    uint32_t *id_t = calloc(bound, sizeof *id_t), *root = calloc(bound, sizeof *root);
    if (!bind || !ptr_t || !ptr_sc || !id_t || !root) goto fail;
    for (uint32_t i = 0; i < bound; i++) bind[i] = UINT32_MAX;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xffff, n = w[0] >> 16;
        if (!n || i + n > words) goto fail;
        if (op == OpDecorate && n >= 4 && w[1] < bound && w[2] == DecBinding) bind[w[1]] = w[3];
        else if (op == OpTypeBool && n >= 2 && !t_bool) t_bool = w[1];
        else if (op == OpTypePointer && n >= 4 && w[1] < bound) { ptr_sc[w[1]] = w[2]; ptr_t[w[1]] = w[3]; }
        else if (op == OpFunction && !first_fn) first_fn = (uint32_t)i;
        i += n;
    }
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xffff, n = w[0] >> 16;
        if (op == OpVariable && n >= 4 && w[2] < bound && bind[w[2]] == binding) { log_var = w[2]; break; }
        i += n;
    }
    if (!log_var || !first_fn || !t_bool) goto fail;
    root[log_var] = log_var;
    gmem_patch p[64]; unsigned np = 0; uint32_t params[8], ptypes[8]; int nparam = 0;
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i; uint32_t op = w[0] & 0xffff, n = w[0] >> 16;
        if (!n || i + n > words) goto fail;
        if (op == OpFunction) nparam = 0;
        else if (op == GOpFunctionParameter && n >= 3 && nparam < 8) { ptypes[nparam] = w[1]; params[nparam++] = w[2]; }
        else if ((op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain || op == OpInBoundsPtrAccessChain) && n >= 4 && w[2] < bound) {
            id_t[w[2]] = w[1]; if (w[3] < bound && root[w[3]]) root[w[2]] = root[w[3]];
        } else if ((op == OpCopyObject || op == OpBitcast) && n >= 4 && w[2] < bound) {
            id_t[w[2]] = w[1]; if (w[3] < bound && root[w[3]]) root[w[2]] = root[w[3]];
        } else if (op == GOpAtomicOr && n >= 7 && w[3] < bound && root[w[3]] == log_var) {
            if (np == 64 || nparam < 2 || ptypes[0] >= bound || ptypes[1] >= bound ||
                ptr_sc[ptypes[0]] != SCFunction || ptr_sc[ptypes[1]] != SCFunction ||
                !ptr_t[ptypes[0]] || ptr_t[ptypes[0]] != ptr_t[ptypes[1]] || !id_t[w[3]]) goto fail;
            p[np++] = (gmem_patch){i, params[0], params[1], w[1], ptr_t[ptypes[0]], t_bool,
                                   id_t[w[3]], w[2], w[6], w[4], w[5], log_var};
        }
        i += n;
    }
    if (!np) goto fail;
    for (unsigned i = 1; i < np; i++) if (p[i].atom_t != p[0].atom_t || p[i].uint_t != p[0].uint_t || p[i].ptr_t != p[0].ptr_t) goto fail;

    uint32_t atom_t = p[0].atom_t, uint_t = p[0].uint_t;
    uint32_t ca0 = bound++, ca1 = bound++, cu0 = bound++, cu1 = bound++, cu12 = bound++, cu20 = bound++;
    uint32_t cubm = bound++, cumask = bound++, cubase = bound++;
    uint32_t decl[] = {
        (4u << 16) | OpConstant, atom_t, ca0, 0,
        (4u << 16) | OpConstant, atom_t, ca1, 1,
        (4u << 16) | OpConstant, uint_t, cu0, 0,
        (4u << 16) | OpConstant, uint_t, cu1, 1,
        (4u << 16) | OpConstant, uint_t, cu12, 12,
        (4u << 16) | OpConstant, uint_t, cu20, 20,
        (4u << 16) | OpConstant, uint_t, cubm, 256,
        (4u << 16) | OpConstant, uint_t, cumask, 1023,
        (4u << 16) | OpConstant, uint_t, cubase, 257,
    };
    const size_t nd = sizeof decl / sizeof *decl, cap = words + nd + 160u * np;
    uint32_t *o = malloc(cap * sizeof *o); if (!o) goto fail;
    size_t k = 0; unsigned pi = 0;
#define GPUT(...) do { uint32_t q_[] = {__VA_ARGS__}; size_t nq_ = sizeof q_ / sizeof *q_; memcpy(o + k, q_, nq_ * sizeof *o); k += nq_; } while (0)
    for (size_t i = 0; i < words;) {
        if (i == first_fn) { memcpy(o + k, decl, sizeof decl); k += nd; }
        const uint32_t *w = code + i; uint32_t n = i < 5 ? 5 : w[0] >> 16;
        memcpy(o + k, w, n * sizeof *o); k += n;
        if (pi < np && i == p[pi].at) {
            const gmem_patch *x = &p[pi++];
            uint32_t oldbit = bound++, first = bound++, llog = bound++, ldone = bound++;
            uint32_t countp = bound++, seqi = bound++, seq = bound++, slot = bound++, pair = bound++, word = bound++;
            uint32_t pl = bound++, whi = bound++, ph = bound++, alo = bound++, ahi = bound++;
            uint32_t page0 = bound++, carry = bound++, pagelo = bound++, pagehi = bound++, pagelo_i = bound++, pagehi_i = bound++;
            uint32_t drop0 = bound++, drop1 = bound++;
            GPUT((5u << 16) | GOpBitwiseAnd, x->atom_t, oldbit, x->old, x->bit);
            GPUT((5u << 16) | GOpIEqual, x->bool_t, first, oldbit, ca0);
            GPUT((3u << 16) | GOpSelectionMerge, ldone, 0);
            GPUT((4u << 16) | GOpBranchConditional, first, llog, ldone);
            GPUT((2u << 16) | OpLabel, llog);
            GPUT((6u << 16) | OpAccessChain, x->ptr_t, countp, x->log_var, cu0, cubm);
            GPUT((7u << 16) | GOpAtomicIAdd, x->atom_t, seqi, countp, x->scope, x->sem, ca1);
            GPUT((4u << 16) | OpBitcast, x->uint_t, seq, seqi);
            GPUT((5u << 16) | GOpBitwiseAnd, x->uint_t, slot, seq, cumask);
            GPUT((5u << 16) | GOpShiftLeftLogical, x->uint_t, pair, slot, cu1);
            GPUT((5u << 16) | GOpIAdd, x->uint_t, word, pair, cubase);
            GPUT((6u << 16) | OpAccessChain, x->ptr_t, pl, x->log_var, cu0, word);
            GPUT((5u << 16) | GOpIAdd, x->uint_t, whi, word, cu1);
            GPUT((6u << 16) | OpAccessChain, x->ptr_t, ph, x->log_var, cu0, whi);
            GPUT((4u << 16) | OpLoad, x->uint_t, alo, x->addr_lo);
            GPUT((4u << 16) | OpLoad, x->uint_t, ahi, x->addr_hi);
            GPUT((5u << 16) | GOpShiftRightLogical, x->uint_t, page0, alo, cu12);
            GPUT((5u << 16) | GOpShiftLeftLogical, x->uint_t, carry, ahi, cu20);
            GPUT((5u << 16) | GOpBitwiseOr, x->uint_t, pagelo, page0, carry);
            GPUT((5u << 16) | GOpShiftRightLogical, x->uint_t, pagehi, ahi, cu12);
            GPUT((4u << 16) | OpBitcast, x->atom_t, pagelo_i, pagelo);
            GPUT((4u << 16) | OpBitcast, x->atom_t, pagehi_i, pagehi);
            GPUT((7u << 16) | GOpAtomicExchange, x->atom_t, drop0, pl, x->scope, x->sem, pagelo_i);
            GPUT((7u << 16) | GOpAtomicExchange, x->atom_t, drop1, ph, x->scope, x->sem, pagehi_i);
            GPUT((2u << 16) | GOpBranch, ldone);
            GPUT((2u << 16) | OpLabel, ldone);
        }
        i += n;
    }
#undef GPUT
    o[3] = bound; *out_words = k; *patched = np;
    free(bind); free(ptr_t); free(ptr_sc); free(id_t); free(root);
    return o;
fail:
    free(bind); free(ptr_t); free(ptr_sc); free(id_t); free(root);
    return NULL;
}
