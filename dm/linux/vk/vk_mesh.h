// Fail-closed VK_EXT_mesh_shader application boundary.  This file is pure:
// device discovery, shader loading and command recording consume its plans,
// while device-free tests exercise every gate without a Vulkan device.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vk_stage.h"
#include "dmsh_abi.h"
#include "dmsh_abi_linux_vk.h"

// VK_EXT_mesh_shader required limits (Khronos Required_Limits).  The current
// title inventory is smaller (32 invocations, 160 vertices, 128 primitives,
// 3200-byte payload), and each actual module is checked again below.
enum {
    VKX_MESH_REQ_INVOCATIONS = 128,
    VKX_MESH_REQ_VERTICES = 256,
    VKX_MESH_REQ_PRIMITIVES = 256,
    VKX_MESH_REQ_TASK_PAYLOAD = 16 * 1024,
    VKX_MESH_REQ_TASK_SHARED = 32 * 1024,
    VKX_MESH_REQ_TASK_PAYLOAD_SHARED = 32 * 1024,
    VKX_MESH_REQ_MESH_SHARED = 28 * 1024,
    VKX_MESH_REQ_MESH_PAYLOAD_SHARED = 28 * 1024,
    VKX_MESH_REQ_OUTPUT_MEMORY = 32 * 1024,
    VKX_MESH_REQ_PAYLOAD_OUTPUT = 47 * 1024,
    VKX_MESH_REQ_OUTPUT_COMPONENTS = 128,
    VKX_MESH_REQ_OUTPUT_LAYERS = 8,
    VKX_MESH_REQ_GROUP_TOTAL = 1u << 22,
    VKX_MESH_REQ_GROUP_DIM = 65535,
    // Exact maxima of the 506 validated title modules in Cycle 29.  These are
    // only for an explicit experimental profile; they never weaken the
    // conformant Required_Limits gate above.
    VKX_MESH_TITLE_INVOCATIONS = 32,
    VKX_MESH_TITLE_VERTICES = 160,
    VKX_MESH_TITLE_PRIMITIVES = 128,
    VKX_MESH_TITLE_PAYLOAD = 3200,
};

enum {
    VKX_MESH_GATE_NO_EXTENSION = 1u << 0,
    VKX_MESH_GATE_NO_MESH_FEATURE = 1u << 1,
    VKX_MESH_GATE_INVOCATIONS = 1u << 2,
    VKX_MESH_GATE_VERTICES = 1u << 3,
    VKX_MESH_GATE_PRIMITIVES = 1u << 4,
    VKX_MESH_GATE_TASK_PAYLOAD = 1u << 5,
    VKX_MESH_GATE_OUTPUT_MEMORY = 1u << 6,
    VKX_MESH_GATE_GROUP_COUNTS = 1u << 7,
    VKX_MESH_GATE_NO_DIRECT_FN = 1u << 8,
    VKX_MESH_GATE_NO_INDIRECT_FN = 1u << 9,
    VKX_MESH_GATE_SHARED_COMBINED = 1u << 10,
    VKX_MESH_GATE_OUTPUT_COMBINED = 1u << 11,
};

typedef struct {
    unsigned reject, direct_reject, indirect_reject;
    int enable_extension;     // feature/property gate passed: safe to request at vkCreateDevice
    int direct_ready;         // extension + features + limits + vkCmdDrawMeshTasksEXT
    int indirect_ready;       // extension + features + limits + vkCmdDrawMeshTasksIndirectEXT
    int task;                 // taskShader and its limits are present
    int experimental;         // explicit non-conformant title profile, never silently substituted
    int indirect_task_only;   // driver-specific restriction when indirect mesh requires a task stage
    int runtime_ready;        // renderer physical proof; deliberately left zero by the application today
    uint32_t task_inv, mesh_inv, task_payload, mesh_vertices, mesh_primitives;
    uint32_t task_shared, task_payload_shared, mesh_shared, mesh_payload_shared;
    uint32_t mesh_output_memory, mesh_payload_output;
    uint32_t mesh_output_components, mesh_output_layers;
    uint32_t task_group_total, mesh_group_total;
    uint32_t task_groups[3], mesh_groups[3];
    uint32_t task_group_size[3], mesh_group_size[3];
} vkx_mesh_caps;

static inline void vkx_mesh_caps_properties(vkx_mesh_caps *c, const VkPhysicalDeviceMeshShaderPropertiesEXT *p) {
    if (!c || !p) return;
    c->task_inv = p->maxTaskWorkGroupInvocations;
    c->mesh_inv = p->maxMeshWorkGroupInvocations;
    c->task_payload = p->maxTaskPayloadSize;
    c->mesh_vertices = p->maxMeshOutputVertices;
    c->mesh_primitives = p->maxMeshOutputPrimitives;
    c->task_shared = p->maxTaskSharedMemorySize;
    c->task_payload_shared = p->maxTaskPayloadAndSharedMemorySize;
    c->mesh_shared = p->maxMeshSharedMemorySize;
    c->mesh_payload_shared = p->maxMeshPayloadAndSharedMemorySize;
    c->mesh_output_memory = p->maxMeshOutputMemorySize;
    c->mesh_payload_output = p->maxMeshPayloadAndOutputMemorySize;
    c->mesh_output_components = p->maxMeshOutputComponents;
    c->mesh_output_layers = p->maxMeshOutputLayers;
    c->task_group_total = p->maxTaskWorkGroupTotalCount;
    c->mesh_group_total = p->maxMeshWorkGroupTotalCount;
    memcpy(c->task_groups, p->maxTaskWorkGroupCount, sizeof c->task_groups);
    memcpy(c->mesh_groups, p->maxMeshWorkGroupCount, sizeof c->mesh_groups);
    memcpy(c->task_group_size, p->maxTaskWorkGroupSize, sizeof c->task_group_size);
    memcpy(c->mesh_group_size, p->maxMeshWorkGroupSize, sizeof c->mesh_group_size);
}

static inline unsigned vkx_mesh_property_reject(const VkPhysicalDeviceMeshShaderPropertiesEXT *p, int want_task) {
    unsigned r = 0;
    if (!p || p->maxMeshWorkGroupInvocations < VKX_MESH_REQ_INVOCATIONS ||
        (want_task && p->maxTaskWorkGroupInvocations < VKX_MESH_REQ_INVOCATIONS)) r |= VKX_MESH_GATE_INVOCATIONS;
    for (int i = 0; i < 3; i++) if (!p || p->maxMeshWorkGroupSize[i] < VKX_MESH_REQ_INVOCATIONS ||
        (want_task && p->maxTaskWorkGroupSize[i] < VKX_MESH_REQ_INVOCATIONS)) r |= VKX_MESH_GATE_INVOCATIONS;
    if (!p || p->maxMeshOutputVertices < VKX_MESH_REQ_VERTICES) r |= VKX_MESH_GATE_VERTICES;
    if (!p || p->maxMeshOutputPrimitives < VKX_MESH_REQ_PRIMITIVES) r |= VKX_MESH_GATE_PRIMITIVES;
    if (want_task && (!p || p->maxTaskPayloadSize < VKX_MESH_REQ_TASK_PAYLOAD)) r |= VKX_MESH_GATE_TASK_PAYLOAD;
    if (!p || p->maxMeshOutputMemorySize < VKX_MESH_REQ_OUTPUT_MEMORY) r |= VKX_MESH_GATE_OUTPUT_MEMORY;
    if (!p || p->maxMeshSharedMemorySize < VKX_MESH_REQ_MESH_SHARED ||
        p->maxMeshPayloadAndSharedMemorySize < VKX_MESH_REQ_MESH_PAYLOAD_SHARED ||
        (want_task && (p->maxTaskSharedMemorySize < VKX_MESH_REQ_TASK_SHARED ||
                       p->maxTaskPayloadAndSharedMemorySize < VKX_MESH_REQ_TASK_PAYLOAD_SHARED))) r |= VKX_MESH_GATE_SHARED_COMBINED;
    if (!p || p->maxMeshPayloadAndOutputMemorySize < VKX_MESH_REQ_PAYLOAD_OUTPUT ||
        p->maxMeshOutputComponents < VKX_MESH_REQ_OUTPUT_COMPONENTS ||
        p->maxMeshOutputLayers < VKX_MESH_REQ_OUTPUT_LAYERS) r |= VKX_MESH_GATE_OUTPUT_COMBINED;
    if (!p || p->maxMeshWorkGroupTotalCount < VKX_MESH_REQ_GROUP_TOTAL ||
        (want_task && p->maxTaskWorkGroupTotalCount < VKX_MESH_REQ_GROUP_TOTAL)) r |= VKX_MESH_GATE_GROUP_COUNTS;
    for (int i = 0; i < 3; i++) if (!p || p->maxMeshWorkGroupCount[i] < VKX_MESH_REQ_GROUP_DIM ||
        (want_task && p->maxTaskWorkGroupCount[i] < VKX_MESH_REQ_GROUP_DIM)) r |= VKX_MESH_GATE_GROUP_COUNTS;
    return r;
}

static inline vkx_mesh_caps vkx_mesh_caps_make(
        int have_extension, const VkPhysicalDeviceMeshShaderFeaturesEXT *f,
        const VkPhysicalDeviceMeshShaderPropertiesEXT *p,
        int have_direct_fn, int have_indirect_fn) {
    vkx_mesh_caps c = {0};
    if (!have_extension) c.reject |= VKX_MESH_GATE_NO_EXTENSION;
    if (!f || !f->meshShader) c.reject |= VKX_MESH_GATE_NO_MESH_FEATURE;
    c.reject |= vkx_mesh_property_reject(p, f && f->taskShader);
    unsigned pre = c.reject;
    c.enable_extension = pre == 0;
    c.direct_reject = pre | (have_direct_fn ? 0 : VKX_MESH_GATE_NO_DIRECT_FN);
    c.indirect_reject = pre | (have_indirect_fn ? 0 : VKX_MESH_GATE_NO_INDIRECT_FN);
    c.direct_ready = c.direct_reject == 0;
    c.indirect_ready = c.indirect_reject == 0;
    vkx_mesh_caps_properties(&c, p);
    c.task = c.direct_ready && f && f->taskShader;
    return c;
}

// Separate opt-in model for a non-conformant driver under development.  It
// checks only the exact Cycle-29 module maxima. It recognizes task/indirect
// only when the driver reports their feature, limits and function; it always
// returns runtime_ready == 0 until the separate root proof gate is applied.
static inline vkx_mesh_caps vkx_mesh_title_caps_make(
        int opt_in, int have_extension, const VkPhysicalDeviceMeshShaderFeaturesEXT *f,
        const VkPhysicalDeviceMeshShaderPropertiesEXT *p,
        int have_direct_fn, int have_indirect_fn) {
    vkx_mesh_caps c = {0}; c.experimental = opt_in != 0;
    if (!opt_in || !have_extension) c.reject |= VKX_MESH_GATE_NO_EXTENSION;
    if (!f || !f->meshShader) c.reject |= VKX_MESH_GATE_NO_MESH_FEATURE;
    if (!p || p->maxMeshWorkGroupInvocations < VKX_MESH_TITLE_INVOCATIONS ||
        p->maxMeshWorkGroupSize[0] < VKX_MESH_TITLE_INVOCATIONS ||
        p->maxMeshWorkGroupSize[1] < 1 || p->maxMeshWorkGroupSize[2] < 1) c.reject |= VKX_MESH_GATE_INVOCATIONS;
    if (!p || p->maxMeshOutputVertices < VKX_MESH_TITLE_VERTICES) c.reject |= VKX_MESH_GATE_VERTICES;
    if (!p || p->maxMeshOutputPrimitives < VKX_MESH_TITLE_PRIMITIVES) c.reject |= VKX_MESH_GATE_PRIMITIVES;
    if (!p || p->maxMeshPayloadAndSharedMemorySize < VKX_MESH_TITLE_PAYLOAD) c.reject |= VKX_MESH_GATE_SHARED_COMBINED;
    if (!p || !p->maxMeshOutputMemorySize) c.reject |= VKX_MESH_GATE_OUTPUT_MEMORY;
    if (!p || !p->maxMeshPayloadAndOutputMemorySize || !p->maxMeshOutputComponents || !p->maxMeshOutputLayers)
        c.reject |= VKX_MESH_GATE_OUTPUT_COMBINED;
    if (!p || !p->maxMeshWorkGroupTotalCount || !p->maxMeshWorkGroupCount[0] ||
        !p->maxMeshWorkGroupCount[1] || !p->maxMeshWorkGroupCount[2]) c.reject |= VKX_MESH_GATE_GROUP_COUNTS;
    c.enable_extension = c.reject == 0;
    c.direct_reject = c.reject | (have_direct_fn ? 0 : VKX_MESH_GATE_NO_DIRECT_FN);
    c.indirect_reject = c.reject | (have_indirect_fn ? 0 : VKX_MESH_GATE_NO_INDIRECT_FN);
    c.direct_ready = c.direct_reject == 0;
    c.indirect_ready = c.indirect_reject == 0;
    /* The explicit proof profile may admit taskless indirect.  Honeykrisp
     * bounds this path to one group for the live title, and Cycle 147's clean
     * second precompile pass proves both depth forms are in the binary cache. */
    c.indirect_task_only = 0;
    vkx_mesh_caps_properties(&c, p);
    // The title profile may recognize task only when the driver reports the
    // feature and its exact title-sized limits.  This is still not runtime
    // permission: runtime_ready remains controlled by the proof gate below.
    c.task = c.direct_ready && f && f->taskShader && p &&
        p->maxTaskWorkGroupInvocations >= VKX_MESH_TITLE_INVOCATIONS &&
        p->maxTaskWorkGroupSize[0] >= VKX_MESH_TITLE_INVOCATIONS &&
        p->maxTaskWorkGroupSize[1] >= 1 && p->maxTaskWorkGroupSize[2] >= 1 &&
        p->maxTaskPayloadSize >= VKX_MESH_TITLE_PAYLOAD &&
        p->maxTaskPayloadAndSharedMemorySize >= VKX_MESH_TITLE_PAYLOAD &&
        p->maxTaskWorkGroupTotalCount && p->maxTaskWorkGroupCount[0] &&
        p->maxTaskWorkGroupCount[1] && p->maxTaskWorkGroupCount[2];
    c.runtime_ready = 0;
    return c;
}

#define VKX_MESH_PROOF_TOKEN "root-physical-proof"
static inline int vkx_mesh_proof_gate(const char *value) {
    return value && !strcmp(value, VKX_MESH_PROOF_TOKEN);
}

static inline const vkx_mesh_caps *vkx_mesh_active_caps(
        const vkx_mesh_caps *conformant, const vkx_mesh_caps *title) {
    if (conformant && conformant->runtime_ready) return conformant;
    if (title && title->runtime_ready) return title;
    return NULL;
}

static inline int vkx_mesh_indirect_stage_allowed(const vkx_mesh_caps *c, int have_task) {
    return c && c->indirect_ready && (!have_task || c->task) &&
        (!c->indirect_task_only || (have_task && c->task));
}

// A task-bound command launches task workgroups and is limited by the queried
// task properties.  A taskless command launches mesh workgroups and is limited
// by the separately queried mesh properties.  Keep these distinct: the G13
// title profile may truthfully expose 4096 task inputs but only 2048 emitted
// mesh workgroups.
static inline int vkx_mesh_dispatch_limits(const vkx_mesh_caps *c, int have_task,
                                            uint32_t *max_x, uint32_t *max_total) {
    if (!c || !max_x || !max_total) return 0;
    *max_x = have_task ? c->task_groups[0] : c->mesh_groups[0];
    *max_total = have_task ? c->task_group_total : c->mesh_group_total;
    return *max_x && *max_total;
}

typedef struct {
    int present, stage, host_slot, descriptor_set;
    int mesh_has_task, mesh_has_task_known, mesh_first;
    uint32_t local[3];
    uint32_t payload, source_payload, shared, output_memory, output_components, max_vertices, max_primitives;
    uint32_t per_primitive_locations;
} vkx_mesh_stage_meta;

static inline const char *vkx_mesh_json_value(const char *json, const char *key) {
    const char *p = json && key ? strstr(json, key) : NULL;
    p = p ? strchr(p + strlen(key), ':') : NULL;
    return p ? p + 1 : NULL;
}

static inline int vkx_mesh_json_bool(const char *json, const char *key, int *out) {
    const char *p = vkx_mesh_json_value(json, key);
    while (p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    if (p && !strncmp(p, "true", 4)) { *out = 1; return 1; }
    if (p && !strncmp(p, "false", 5)) { *out = 0; return 1; }
    return 0;
}

static inline int vkx_mesh_json_u32(const char **at, uint32_t *out) {
    const char *p = at ? *at : NULL;
    uint64_t v = 0; int any = 0;
    while (p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == '[' || *p == ',')) p++;
    while (p && *p >= '0' && *p <= '9') { any = 1; v = v * 10 + (unsigned)(*p++ - '0'); if (v > UINT32_MAX) return 0; }
    if (!any) return 0;
    *out = (uint32_t)v; *at = p; return 1;
}

// Parse the exact MIT adapter job ABI copied beside each Cycle-29 module.
// The first three mesh words are {max vertices, max primitives, payload};
// task jobs use {0,0,payload}.  This is source metadata, never inferred by
// relabelling a compute shader.
static inline int vkx_mesh_job_meta(const char *json, int stage, vkx_mesh_stage_meta *out) {
    if (!json || !out || (stage != VKX_STAGE_TS && stage != VKX_STAGE_MS)) return 0;
    const char *sv = vkx_mesh_json_value(json, "\"stage\"");
    while (sv && (*sv == ' ' || *sv == '\t' || *sv == '\r' || *sv == '\n')) sv++;
    const char *want = stage == VKX_STAGE_TS ? "\"ts\"" : "\"ms\"";
    if (!sv || strncmp(sv, want, 4)) return 0;
    const char *local = vkx_mesh_json_value(json, "\"local\"");
    const char *shared = vkx_mesh_json_value(json, "\"shared\"");
    const char *mesh = vkx_mesh_json_value(json, "\"mesh\"");
    uint32_t l[3], sh, m[3];
    if (!vkx_mesh_json_u32(&local, &l[0]) || !vkx_mesh_json_u32(&local, &l[1]) ||
        !vkx_mesh_json_u32(&local, &l[2]) || !vkx_mesh_json_u32(&shared, &sh) ||
        !vkx_mesh_json_u32(&mesh, &m[0]) || !vkx_mesh_json_u32(&mesh, &m[1]) ||
        !vkx_mesh_json_u32(&mesh, &m[2])) return 0;
    if (!l[0] || !l[1] || !l[2] || (stage == VKX_STAGE_TS && (m[0] || m[1])) ||
        (stage == VKX_STAGE_MS && (!m[0] || !m[1]))) return 0;
    *out = (vkx_mesh_stage_meta){.present = 1, .stage = stage,
        .host_slot = vkx_stage_set(stage), .descriptor_set = vkx_stage_set(stage),
        .local = {l[0], l[1], l[2]}, .payload = m[2], .source_payload = m[2], .shared = sh,
        .max_vertices = m[0], .max_primitives = m[1]};
    const char *mh = strstr(json, "\"mesh_has_task\"");
    if (mh) {
        if (!vkx_mesh_json_bool(json, "\"mesh_has_task\"", &out->mesh_has_task)) return 0;
        out->mesh_has_task_known = 1;
    }
    return 1;
}

static inline int vkx_mesh_first_descriptor(uint32_t binding, char desc, const char *kind) {
    return binding == DMSH_LINUX_VK_MESH_FIRST_BINDING && desc == 'u' &&
           kind && !strcmp(kind, "mesh_first");
}

enum {
    VKX_MESH_PROGRAM_NO_RUNTIME = 1u << 0,
    VKX_MESH_PROGRAM_NO_MESH = 1u << 1,
    VKX_MESH_PROGRAM_NO_FRAGMENT = 1u << 2,
    VKX_MESH_PROGRAM_BAD_STAGE = 1u << 3,
    VKX_MESH_PROGRAM_NO_TASK_FEATURE = 1u << 4,
    VKX_MESH_PROGRAM_LIMIT = 1u << 5,
    VKX_MESH_PROGRAM_PRIMITIVE_INTERFACE = 1u << 6,
    VKX_MESH_PROGRAM_PAYLOAD_INTERFACE = 1u << 7,
    VKX_MESH_PROGRAM_FIRST_INTERFACE = 1u << 8,
};

typedef struct {
    unsigned reject;
    int count;
    int stage[3], host_slot[3], descriptor_set[3];
} vkx_mesh_program_plan;

static inline uint64_t vkx_mesh_local_invocations(const vkx_mesh_stage_meta *s) {
    return (uint64_t)s->local[0] * s->local[1] * s->local[2];
}

// Form the VS-free task(optional)+mesh+fragment program.  host_slot and set
// are checked separately from the actual stage so remapped slots 3/4 can
// never make a hull/domain shader masquerade as task/mesh.
static inline vkx_mesh_program_plan vkx_mesh_program(
        const vkx_mesh_caps *c, const vkx_mesh_stage_meta *task,
        const vkx_mesh_stage_meta *mesh, const vkx_mesh_stage_meta *frag) {
    vkx_mesh_program_plan p = {0};
    if (!c || !c->direct_ready || !c->runtime_ready) p.reject |= VKX_MESH_PROGRAM_NO_RUNTIME;
    if (!mesh || !mesh->present) p.reject |= VKX_MESH_PROGRAM_NO_MESH;
    if (!frag || !frag->present) p.reject |= VKX_MESH_PROGRAM_NO_FRAGMENT;
    if (mesh && mesh->present && (mesh->stage != VKX_STAGE_MS || mesh->host_slot != 4 || mesh->descriptor_set != 4)) p.reject |= VKX_MESH_PROGRAM_BAD_STAGE;
    if (frag && frag->present && (frag->stage != VKX_STAGE_PS || frag->host_slot != 1 || frag->descriptor_set != 1)) p.reject |= VKX_MESH_PROGRAM_BAD_STAGE;
    if (task && task->present) {
        if (task->stage != VKX_STAGE_TS || task->host_slot != 3 || task->descriptor_set != 3) p.reject |= VKX_MESH_PROGRAM_BAD_STAGE;
        if (!c || !c->task) p.reject |= VKX_MESH_PROGRAM_NO_TASK_FEATURE;
    }
    if (mesh && mesh->present && c) {
        uint64_t inv = vkx_mesh_local_invocations(mesh);
        if (!inv || inv > c->mesh_inv || mesh->local[0] > c->mesh_group_size[0] ||
            mesh->local[1] > c->mesh_group_size[1] || mesh->local[2] > c->mesh_group_size[2] ||
            mesh->max_vertices > c->mesh_vertices || mesh->max_primitives > c->mesh_primitives ||
            mesh->shared > c->mesh_shared || (uint64_t)mesh->payload + mesh->shared > c->mesh_payload_shared ||
            mesh->output_memory > c->mesh_output_memory ||
            (uint64_t)mesh->payload + mesh->output_memory > c->mesh_payload_output ||
            mesh->output_components > c->mesh_output_components)
            p.reject |= VKX_MESH_PROGRAM_LIMIT;
        // output_components counts user varyings, not the built-in Position.
        // A position-only mesh therefore truthfully reports zero here.  Keep
        // output-memory reflection mandatory so unknown metadata still fails
        // closed; the G13 position-only physical readback proves the zero-user-
        // varying case itself.
        if (c->experimental && !mesh->output_memory)
            p.reject |= VKX_MESH_PROGRAM_LIMIT;
    }
    if (task && task->present && c) {
        uint64_t inv = vkx_mesh_local_invocations(task);
        if (!inv || inv > c->task_inv || task->local[0] > c->task_group_size[0] ||
            task->local[1] > c->task_group_size[1] || task->local[2] > c->task_group_size[2] ||
            task->payload > c->task_payload || task->shared > c->task_shared ||
            (uint64_t)task->payload + task->shared > c->task_payload_shared) p.reject |= VKX_MESH_PROGRAM_LIMIT;
    }
    if (mesh && frag && mesh->present && frag->present &&
        (frag->per_primitive_locations & ~mesh->per_primitive_locations))
        p.reject |= VKX_MESH_PROGRAM_PRIMITIVE_INTERFACE;
    if (mesh && mesh->present && ((task && task->present && task->payload != mesh->payload) ||
                                  ((!task || !task->present) && mesh->payload)))
        p.reject |= VKX_MESH_PROGRAM_PAYLOAD_INTERFACE;
    if (mesh && mesh->present && task && task->present && mesh->mesh_first)
        p.reject |= VKX_MESH_PROGRAM_FIRST_INTERFACE;
    if (mesh && mesh->present && mesh->mesh_has_task_known &&
        mesh->mesh_has_task != !!(task && task->present))
        p.reject |= VKX_MESH_PROGRAM_FIRST_INTERFACE;
    if (!p.reject) {
        if (task && task->present) {
            p.stage[p.count] = VKX_STAGE_TS; p.host_slot[p.count] = 3; p.descriptor_set[p.count++] = 3;
        }
        p.stage[p.count] = VKX_STAGE_MS; p.host_slot[p.count] = 4; p.descriptor_set[p.count++] = 4;
        p.stage[p.count] = VKX_STAGE_PS; p.host_slot[p.count] = 1; p.descriptor_set[p.count++] = 1;
    }
    return p;
}

// PerPrimitiveEXT (5271) variables and their Location decorations.  The
// mask lets the application prove that every fragment per-primitive input is
// written per primitive by the mesh stage; extra mesh outputs are harmless.
static inline int vkx_spv_per_primitive_locations(const uint32_t *w, size_t nw, uint32_t *mask) {
    enum { OP_DECORATE = 71, DEC_LOCATION = 30, DEC_PER_PRIMITIVE = 5271, MAX_IDS = 64 };
    uint32_t ids[MAX_IDS]; size_t ni = 0;
    if (mask) *mask = 0;
    if (!w || nw < 5 || w[0] != 0x07230203) return 0;
    for (size_t i = 5; i < nw;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || i + n > nw) return 0;
        if (op == OP_DECORATE && n >= 3 && w[i + 2] == DEC_PER_PRIMITIVE) {
            if (ni == MAX_IDS) return 0;
            ids[ni++] = w[i + 1];
        }
        i += n;
    }
    uint32_t out = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == OP_DECORATE && n >= 4 && w[i + 2] == DEC_LOCATION && w[i + 3] < 32)
            for (size_t k = 0; k < ni; k++) if (ids[k] == w[i + 1]) out |= 1u << w[i + 3];
        i += n;
    }
    if (mask) *mask = out;
    return 1;
}

typedef struct {
    int indirect;
    uint32_t count, first, max_groups, max_total;
    uint64_t guest_address;
    uint8_t guest_count_offset, guest_first_offset, guest_stride;
    uint8_t vulkan_stride;
    int needs_first_base, needs_gpu_conversion;
} vkx_mesh_draw_meta;

enum { VKX_MESH_FIRST_DIRECT_WORD = 1, VKX_MESH_FIRST_INDIRECT_WORD = 2 };
typedef struct {
    uint32_t abi_slot, descriptor_binding, byte_size, alignment;
    int owner_stage, source, add_exactly_once;
    uint32_t direct_value;
    uint64_t guest_address;
    int submit_lifetime;            // backing word must live until this submit's timeline completion
    int translator_access_ready;    // Cycle33 owner modules load this word; renderer runtime_ready remains separate/closed
} vkx_mesh_first_plan;

// dmsh_meshstage.py's existing ABI adds first in the task stage when present,
// otherwise in mesh. Never add it in both stages. The Vulkan import uses the
// separate binding-512 UBO and this submit-lifetime plan.
static inline vkx_mesh_first_plan vkx_mesh_first(const vkx_mesh_draw_meta *m, int have_task) {
    vkx_mesh_first_plan p = {.abi_slot = DMSH_MESH_FIRST_SLOT,
        .descriptor_binding = DMSH_LINUX_VK_MESH_FIRST_BINDING,
        .byte_size = 256, .alignment = 256,
        .owner_stage = have_task ? VKX_STAGE_TS : VKX_STAGE_MS, .add_exactly_once = 1,
        .submit_lifetime = 1, .translator_access_ready = 1};
    if (!m) return p;
    if (m->indirect) { p.source = VKX_MESH_FIRST_INDIRECT_WORD; p.guest_address = m->guest_address + m->guest_first_offset; }
    else { p.source = VKX_MESH_FIRST_DIRECT_WORD; p.direct_value = m->first; }
    return p;
}

typedef struct { uint64_t src, dst; uint32_t bytes; } vkx_mesh_copy_word;
typedef struct {
    uint32_t vk_record_yz[2];       // {y,z}; X has a single writer: copied NVN count
    uint64_t record_init_offset;
    uint32_t record_init_size;
    vkx_mesh_copy_word count, first;
    uint32_t vk_record_size, first_scratch_size;
    int record_init_transfer, transfer_to_indirect_barrier, transfer_to_shader_barrier;
    vkx_mesh_first_plan first_plan;
} vkx_mesh_indirect_bridge;

typedef struct {
    uint64_t src_size, src_offset;
    uint64_t dst_size, record_offset, first_offset;
    int same_buffer;
} vkx_mesh_indirect_layout;

static inline int vkx_mesh_range(uint64_t off, uint64_t n, uint64_t size) {
    return off <= size && n <= size - off;
}

static inline int vkx_mesh_overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b ? b - a < an : a - b < bn;
}

// No CPU readback: preinitialize Vulkan y/z to one, copy the two NVN words
// independently, then make them visible to indirect-command and shader reads.
// dst addresses are supplied by the future submit-owned scratch allocator.
static inline int vkx_mesh_indirect_plan(const vkx_mesh_draw_meta *m, int have_task,
                                         const vkx_mesh_indirect_layout *l,
                                         vkx_mesh_indirect_bridge *out) {
    if (!m || !m->indirect || !m->needs_gpu_conversion || !l || !out ||
        !vkx_mesh_range(l->src_offset, 8, l->src_size) ||
        !vkx_mesh_range(l->record_offset, sizeof(VkDrawMeshTasksIndirectCommandEXT), l->dst_size) ||
        !vkx_mesh_range(l->first_offset, 256, l->dst_size) ||
        (l->record_offset & 3) || (l->first_offset & 255) ||
        vkx_mesh_overlap(l->record_offset, sizeof(VkDrawMeshTasksIndirectCommandEXT), l->first_offset, 256) ||
        (l->same_buffer && (vkx_mesh_overlap(l->src_offset, 8, l->record_offset,
                                             sizeof(VkDrawMeshTasksIndirectCommandEXT)) ||
                            vkx_mesh_overlap(l->src_offset, 8, l->first_offset, 256)))) return 0;
    *out = (vkx_mesh_indirect_bridge){.vk_record_yz = {1, 1},
        .record_init_offset = l->record_offset + 4, .record_init_size = 8,
        .count = {l->src_offset + m->guest_count_offset, l->record_offset, 4},
        .first = {l->src_offset + m->guest_first_offset, l->first_offset, 4},
        .vk_record_size = sizeof(VkDrawMeshTasksIndirectCommandEXT), .first_scratch_size = 256,
        .record_init_transfer = 1,
        .transfer_to_indirect_barrier = 1, .transfer_to_shader_barrier = 1,
        .first_plan = vkx_mesh_first(m, have_task)};
    return 1;
}

enum {
    VKX_MESH_CMD_ALLOC_FIRST = 1,
    VKX_MESH_CMD_WRITE_FIRST,
    VKX_MESH_CMD_ALLOC_INDIRECT,
    VKX_MESH_CMD_INIT_INDIRECT,
    VKX_MESH_CMD_COPY_COUNT,
    VKX_MESH_CMD_COPY_FIRST,
    VKX_MESH_CMD_BARRIER_INDIRECT,
    VKX_MESH_CMD_BARRIER_OWNER,
    VKX_MESH_CMD_BEGIN_RENDERING,
    VKX_MESH_CMD_PREPARE_PROGRAM,
    VKX_MESH_CMD_DRAW_DIRECT,
    VKX_MESH_CMD_DRAW_INDIRECT,
};

typedef struct {
    int indirect, have_task, owner_stage, nstep;
    uint32_t group_count;
    uint8_t step[12];
    VkPipelineStageFlags copy_src_stage, indirect_dst_stage, owner_dst_stage;
    VkAccessFlags copy_src_access, indirect_dst_access, owner_dst_access;
    vkx_mesh_first_plan first;
    vkx_mesh_indirect_bridge bridge;
} vkx_mesh_command_plan;

static inline int vkx_mesh_command_plan_make(
        const vkx_mesh_draw_meta *m, int have_task,
        const vkx_mesh_indirect_layout *layout, vkx_mesh_command_plan *out) {
    if (!m || !out) return 0;
    vkx_mesh_command_plan p = {.indirect = m->indirect, .have_task = have_task,
        .owner_stage = have_task ? VKX_STAGE_TS : VKX_STAGE_MS,
        .group_count = m->count,
        .copy_src_stage = m->indirect ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_HOST_BIT,
        .indirect_dst_stage = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        .owner_dst_stage = have_task ? VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT : VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
        .copy_src_access = m->indirect ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_HOST_WRITE_BIT,
        .indirect_dst_access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        .owner_dst_access = VK_ACCESS_UNIFORM_READ_BIT,
        .first = vkx_mesh_first(m, have_task)};
    p.step[p.nstep++] = VKX_MESH_CMD_ALLOC_FIRST;
    if (m->indirect) {
        if (!vkx_mesh_indirect_plan(m, have_task, layout, &p.bridge)) return 0;
        p.step[p.nstep++] = VKX_MESH_CMD_ALLOC_INDIRECT;
        p.step[p.nstep++] = VKX_MESH_CMD_INIT_INDIRECT;
        p.step[p.nstep++] = VKX_MESH_CMD_COPY_COUNT;
        p.step[p.nstep++] = VKX_MESH_CMD_COPY_FIRST;
        p.step[p.nstep++] = VKX_MESH_CMD_BARRIER_INDIRECT;
    } else p.step[p.nstep++] = VKX_MESH_CMD_WRITE_FIRST;
    p.step[p.nstep++] = VKX_MESH_CMD_BARRIER_OWNER;
    p.step[p.nstep++] = VKX_MESH_CMD_BEGIN_RENDERING;
    p.step[p.nstep++] = VKX_MESH_CMD_PREPARE_PROGRAM;
    p.step[p.nstep++] = m->indirect ? VKX_MESH_CMD_DRAW_INDIRECT : VKX_MESH_CMD_DRAW_DIRECT;
    *out = p; return 1;
}

static inline int vkx_mesh_draw_direct(uint32_t first, uint32_t count,
                                        uint32_t max_groups, uint32_t max_total,
                                        vkx_mesh_draw_meta *m) {
    if (!m || count > max_groups || count > max_total || first > UINT32_MAX - count) return 0;
    *m = (vkx_mesh_draw_meta){.count = count, .first = first, .max_groups = max_groups, .max_total = max_total,
                             .needs_first_base = first != 0};
    return 1;
}

// The NVN2 GPU-written indirect record is {count, first}, not Vulkan's
// {groupCountX,Y,Z}.  Preserve the offsets and require a GPU conversion; no
// CPU readback and no direct vkCmdDrawMeshTasksIndirectEXT on these bytes.
static inline int vkx_mesh_draw_indirect(uint64_t address, uint32_t max_groups, uint32_t max_total,
                                         vkx_mesh_draw_meta *m) {
    if (!m || !address || address > UINT64_MAX - 8 || !max_groups || !max_total) return 0;
    *m = (vkx_mesh_draw_meta){.indirect = 1, .max_groups = max_groups, .max_total = max_total,
                             .guest_address = address, .guest_count_offset = 0,
                             .guest_first_offset = 4, .guest_stride = 8,
                             .vulkan_stride = sizeof(VkDrawMeshTasksIndirectCommandEXT),
                             .needs_first_base = 1, .needs_gpu_conversion = 1};
    return 1;
}
