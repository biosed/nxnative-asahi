#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_mesh.h"
#include "vk_mesh_trace.h"
#include "spv_reflect.h"

static int bad;
static void check(int c, const char *what) { if (!c) { printf("FAIL: %s\n", what); bad++; } }

static unsigned char *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long z = ftell(f); fseek(f, 0, SEEK_SET);
    if (z < 0) { fclose(f); return NULL; }
    unsigned char *p = malloc((size_t)z + 1); if (!p) { fclose(f); return NULL; }
    *n = fread(p, 1, (size_t)z, f); p[*n] = 0; fclose(f); return p;
}

static void mesh_gl_depth(void) {
#define I(op, n) (((uint32_t)(n) << 16) | (op))
    uint32_t m[] = {
        0x07230203, 0x00010600, 0, 22, 0,
        I(15, 7), 5365, 13, 0x6e69616d, 0, 11, 9,
        I(16, 6), 13, 17, 32, 1, 1,
        I(16, 4), 13, 26, 64,
        I(71, 4), 11, 11, 29,
        I(72, 5), 5, 0, 11, 0,
        I(19, 2), 1,
        I(21, 4), 2, 32, 0,
        I(22, 3), 3, 32,
        I(23, 4), 4, 3, 4,
        I(30, 3), 5, 4,
        I(43, 4), 2, 6, 64,
        I(28, 4), 7, 5, 6,
        I(32, 4), 8, 3, 7,
        I(59, 4), 8, 9, 3,
        I(32, 4), 10, 1, 2,
        I(59, 4), 10, 11, 1,
        I(43, 4), 3, 15, 0,
        I(44, 7), 4, 16, 15, 15, 15, 15,
        I(43, 4), 2, 17, 0,
        I(32, 4), 18, 3, 4,
        I(33, 3), 12, 1,
        I(54, 5), 1, 13, 0, 12,
        I(248, 2), 14,
        I(61, 4), 2, 20, 11,
        I(65, 6), 18, 21, 9, 20, 17,
        I(62, 3), 21, 16,
        I(253, 1),
        I(56, 1),
    };
    size_t no = 0; uint32_t *o = spvr_gl_depth(m, sizeof m / sizeof *m, &no);
    check(o && no > sizeof m / sizeof *m && o[3] > m[3], "MeshEXT GL-depth form is produced");
    int barriers = 0, positions = 0, stores = 0, workgroup_vars = 0, output_loads = 0;
    uint8_t output_ptr[256] = {0};
    for (size_t i = 5; o && i < no;) {
        uint32_t op = o[i] & 0xffff, n = o[i] >> 16;
        if (!n || i + n > no) { check(0, "MeshEXT GL-depth form remains structurally bounded"); break; }
        if (op == 224) barriers++;
        if (op == 59 && n >= 4 && o[i + 3] == 4) workgroup_vars++;
        if (op == 65 && n >= 6 && o[i + 3] == 9) { positions++; if (o[i + 2] < sizeof output_ptr) output_ptr[o[i + 2]] = 1; }
        if (op == 61 && n >= 4 && o[i + 3] < sizeof output_ptr && output_ptr[o[i + 3]]) output_loads++;
        if (op == 62) stores++;
        i += n;
    }
    check(barriers == 1 && positions == 3 && stores == 4 && workgroup_vars == 1 && output_loads == 0,
          "MeshEXT GL-depth form shadows Position stores and fixes two disjoint vertices without reading Output");
    free(o);
#undef I
}

static void mesh_gl_depth_direct_position_array(void) {
#define I(op, n) (((uint32_t)(n) << 16) | (op))
    // NX2navi's legal alternative to gl_PerVertex[]: the Position built-in is
    // attached directly to an Output vec4 array.  Treating variable 9 as one
    // vec4 would make the injected OpLoad type-invalid.
    uint32_t m[] = {
        0x07230203, 0x00010600, 0, 22, 0,
        I(15, 7), 5365, 13, 0x6e69616d, 0, 11, 9,
        I(16, 6), 13, 17, 32, 1, 1,
        I(16, 4), 13, 26, 64,
        I(71, 4), 11, 11, 29,
        I(71, 4), 9, 11, 0,
        I(19, 2), 1,
        I(21, 4), 2, 32, 0,
        I(22, 3), 3, 32,
        I(23, 4), 4, 3, 4,
        I(43, 4), 2, 6, 64,
        I(28, 4), 7, 4, 6,
        I(32, 4), 8, 3, 7,
        I(59, 4), 8, 9, 3,
        I(32, 4), 10, 1, 2,
        I(59, 4), 10, 11, 1,
        I(43, 4), 3, 15, 0,
        I(44, 7), 4, 16, 15, 15, 15, 15,
        I(32, 4), 18, 3, 4,
        I(33, 3), 12, 1,
        I(54, 5), 1, 13, 0, 12,
        I(248, 2), 14,
        I(61, 4), 2, 20, 11,
        I(65, 5), 18, 21, 9, 20,
        I(62, 3), 21, 16,
        I(253, 1),
        I(56, 1),
    };
    size_t no = 0; uint32_t *o = spvr_gl_depth(m, sizeof m / sizeof *m, &no);
    int barriers = 0, direct_position_loads = 0, workgroup_vars = 0;
    for (size_t i = 5; o && i < no;) {
        uint32_t op = o[i] & 0xffff, n = o[i] >> 16;
        if (!n || i + n > no) { check(0, "direct Position-array GL-depth form remains structurally bounded"); break; }
        if (op == 224) barriers++;
        if (op == 59 && n >= 4 && o[i + 3] == 4) workgroup_vars++;
        if (op == 61 && n >= 4 && o[i + 3] == 9) direct_position_loads++;
        i += n;
    }
    check(o && barriers == 1 && workgroup_vars == 1 && direct_position_loads == 0,
          "MeshEXT direct Position array is shadowed and never loaded as one vec4");
    free(o);
#undef I
}

// Optional device-free audit of the segregated Cycle-29 directory.  It uses
// only sidecars and canonical SPIR-V; the normal make target stays game-free.
static void inventory(int argc, char **argv) {
    if (argc == 1) return;
    check(argc >= 5 && !strcmp(argv[1], "--inventory"), "inventory arguments");
    if (argc < 5 || strcmp(argv[1], "--inventory")) return;
    int want_ts = atoi(argv[2]), want_ms = atoi(argv[3]), nts = 0, nms = 0;
    int g13_mesh_shape = 0, mesh_no_payload = 0;
    uint32_t max_shared = 0, max_payload = 0, max_output = 0, max_components = 0;
    for (int i = 4; i < argc; i++) {
        size_t n = 0; unsigned char *job = read_all(argv[i], &n); (void)n;
        vkx_mesh_stage_meta m; int st = job && vkx_mesh_job_meta((const char *)job, VKX_STAGE_TS, &m) ? VKX_STAGE_TS :
                                       job && vkx_mesh_job_meta((const char *)job, VKX_STAGE_MS, &m) ? VKX_STAGE_MS : -1;
        if (st == VKX_STAGE_TS) nts++; else if (st == VKX_STAGE_MS) nms++;
        check(st >= 0, "inventory job metadata");
        size_t l = strlen(argv[i]); check(l > 9 && !strcmp(argv[i] + l - 9, ".job.json"), "inventory job suffix");
        if (st >= 0 && l > 9) {
            char path[4096]; check(l + 1 < sizeof path, "inventory path bound");
            if (l + 1 < sizeof path) {
                memcpy(path, argv[i], l - 8); strcpy(path + l - 8, "vkb");
                size_t nv = 0; unsigned char *vkb = read_all(path, &nv); (void)nv;
                char header[32]; snprintf(header, sizeof header, " %s set %d", st == VKX_STAGE_TS ? "ts" : "ms", vkx_stage_set(st));
                check(vkb && strstr((const char *)vkb, header), "inventory vkb stage/set agreement"); free(vkb);
                memcpy(path, argv[i], l - 8); strcpy(path + l - 8, "vk.spv");
                size_t ns = 0; unsigned char *spv = read_all(path, &ns); uint32_t pm = 0;
                check(spv && !(ns & 3) && vkx_spv_per_primitive_locations((const uint32_t *)spv, ns / 4, &pm),
                      "inventory canonical SPIR-V reflection");
                if (spv && !(ns & 3)) {
                    spvr_info ri; char err[160];
                    int ok = spvr_reflect((const uint32_t *)spv, ns / 4, &ri, err, sizeof err) == 0;
                    int agree = ok && ri.stage == vkx_stage_vk_bit(st) && !memcmp(ri.local_size, m.local, sizeof ri.local_size) &&
                                ri.shared_size >= m.shared;
                    if (!agree) printf("inventory mismatch %s: reflect=%d stage=%#x/%#x local=%u,%u,%u/%u,%u,%u payload=%u/%u shared=%u/%u%s%s\n",
                        argv[i], ok, ri.stage, vkx_stage_vk_bit(st), ri.local_size[0], ri.local_size[1], ri.local_size[2], m.local[0], m.local[1], m.local[2],
                        ri.task_payload_size, m.source_payload, ri.shared_size, m.shared, ok ? "" : " err=", ok ? "" : err);
                    check(agree, "inventory emitted stage/local/payload/shared agreement");
                    if (ok) {
                        if (ri.shared_size > max_shared) max_shared = ri.shared_size;
                        if (ri.task_payload_size > max_payload) max_payload = ri.task_payload_size;
                        if (ri.output_size > max_output) max_output = ri.output_size;
                        if (ri.output_components > max_components) max_components = ri.output_components;
                        if (st == VKX_STAGE_MS) {
                            if (!ri.task_payload_size) mesh_no_payload++;
                            if (m.local[0] <= 1024 && m.local[1] <= 1024 && m.local[2] <= 64 &&
                                m.max_vertices <= VKX_MESH_TITLE_VERTICES && m.max_primitives <= VKX_MESH_TITLE_PRIMITIVES &&
                                ri.shared_size <= 24 * 1024 && (uint64_t)ri.task_payload_size + ri.shared_size <= 24 * 1024 &&
                                ri.output_size <= 8 * 1024 && (uint64_t)ri.task_payload_size + ri.output_size <= 8 * 1024 &&
                                ri.output_components <= 4) g13_mesh_shape++;
                        }
                    }
                }
                free(spv);
            }
        }
        free(job);
    }
    check(nts == want_ts && nms == want_ms, "inventory stage counts");
    printf("t_vk_mesh: inventory %d task + %d mesh; emitted maxima shared=%u payload=%u output=%u components=%u; G13 shape-only mesh=%d, no-payload mesh=%d (task unsupported)\n",
           nts, nms, max_shared, max_payload, max_output, max_components, g13_mesh_shape, mesh_no_payload);
}

static VkPhysicalDeviceMeshShaderPropertiesEXT limits(void) {
    VkPhysicalDeviceMeshShaderPropertiesEXT p = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    p.maxTaskWorkGroupInvocations = p.maxMeshWorkGroupInvocations = VKX_MESH_REQ_INVOCATIONS;
    for (int i = 0; i < 3; i++) {
        p.maxTaskWorkGroupSize[i] = p.maxMeshWorkGroupSize[i] = VKX_MESH_REQ_INVOCATIONS;
        p.maxTaskWorkGroupCount[i] = p.maxMeshWorkGroupCount[i] = VKX_MESH_REQ_GROUP_DIM;
    }
    p.maxTaskWorkGroupTotalCount = p.maxMeshWorkGroupTotalCount = VKX_MESH_REQ_GROUP_TOTAL;
    p.maxTaskPayloadSize = VKX_MESH_REQ_TASK_PAYLOAD;
    p.maxTaskSharedMemorySize = VKX_MESH_REQ_TASK_SHARED;
    p.maxTaskPayloadAndSharedMemorySize = VKX_MESH_REQ_TASK_PAYLOAD_SHARED;
    p.maxMeshSharedMemorySize = VKX_MESH_REQ_MESH_SHARED;
    p.maxMeshPayloadAndSharedMemorySize = VKX_MESH_REQ_MESH_PAYLOAD_SHARED;
    p.maxMeshOutputVertices = VKX_MESH_REQ_VERTICES;
    p.maxMeshOutputPrimitives = VKX_MESH_REQ_PRIMITIVES;
    p.maxMeshOutputMemorySize = VKX_MESH_REQ_OUTPUT_MEMORY;
    p.maxMeshPayloadAndOutputMemorySize = VKX_MESH_REQ_PAYLOAD_OUTPUT;
    p.maxMeshOutputComponents = VKX_MESH_REQ_OUTPUT_COMPONENTS;
    p.maxMeshOutputLayers = VKX_MESH_REQ_OUTPUT_LAYERS;
    return p;
}

static vkx_mesh_caps ready_caps(void) {
    VkPhysicalDeviceMeshShaderFeaturesEXT f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, .taskShader = 1, .meshShader = 1};
    VkPhysicalDeviceMeshShaderPropertiesEXT p = limits();
    vkx_mesh_caps c = vkx_mesh_caps_make(1, &f, &p, 1, 1);
    c.runtime_ready = 1;
    return c;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--gl-depth")) {
        size_t n = 0, no = 0; unsigned char *p = read_all(argv[2], &n);
        uint32_t *o = p && !(n & 3) ? spvr_gl_depth((const uint32_t *)p, n / 4, &no) : NULL;
        FILE *f = o ? fopen(argv[3], "wb") : NULL;
        int ok = f && fwrite(o, 4, no, f) == no && !fclose(f);
        if (f && !ok) fclose(f);
        free(p); free(o);
        if (!ok) { fprintf(stderr, "t_vk_mesh: GL-depth transform failed\n"); return 1; }
        printf("t_vk_mesh: wrote %zu words\n", no); return 0;
    }
    mesh_gl_depth();
    mesh_gl_depth_direct_position_array();
    VkPhysicalDeviceMeshShaderFeaturesEXT f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, .taskShader = 1, .meshShader = 1};
    VkPhysicalDeviceMeshShaderPropertiesEXT p = limits(), q;
    vkx_mesh_caps c = vkx_mesh_caps_make(1, &f, &p, 1, 1);
    check(c.enable_extension && c.direct_ready && c.indirect_ready && c.task && !c.runtime_ready,
          "valid driver passes but runtime exposure remains closed");
    check(vkx_mesh_caps_make(0, &f, &p, 1, 1).reject & VKX_MESH_GATE_NO_EXTENSION,
          "missing extension rejected");
    VkPhysicalDeviceMeshShaderFeaturesEXT no_mesh = f; no_mesh.meshShader = 0;
    check(vkx_mesh_caps_make(1, &no_mesh, &p, 1, 1).reject & VKX_MESH_GATE_NO_MESH_FEATURE,
          "missing mesh feature rejected");
    check(vkx_mesh_caps_make(1, &f, &p, 0, 1).direct_reject & VKX_MESH_GATE_NO_DIRECT_FN,
          "missing direct function rejected");
    vkx_mesh_caps direct_only = vkx_mesh_caps_make(1, &f, &p, 1, 0);
    check(direct_only.direct_ready && !direct_only.indirect_ready &&
          direct_only.indirect_reject & VKX_MESH_GATE_NO_INDIRECT_FN,
          "missing indirect function closes only indirect path");
    q = p; q.maxMeshWorkGroupInvocations--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_INVOCATIONS, "invocation limit rejected");
    q = p; q.maxMeshOutputVertices--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_VERTICES, "vertex limit rejected");
    q = p; q.maxMeshOutputPrimitives--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_PRIMITIVES, "primitive limit rejected");
    q = p; q.maxTaskPayloadSize--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_TASK_PAYLOAD, "payload limit rejected");
    q = p; q.maxMeshOutputMemorySize--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_OUTPUT_MEMORY, "output-memory limit rejected");
    q = p; q.maxMeshSharedMemorySize--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_SHARED_COMBINED, "shared-memory limit rejected");
    q = p; q.maxMeshPayloadAndOutputMemorySize--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_OUTPUT_COMBINED, "combined output limit rejected");
    for (int i = 0; i < 3; i++) {
        q = p; q.maxMeshWorkGroupCount[i]--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_GROUP_COUNTS, "each mesh group-count dimension checked");
        q = p; q.maxTaskWorkGroupCount[i]--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_GROUP_COUNTS, "each task group-count dimension checked");
        q = p; q.maxMeshWorkGroupSize[i]--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_INVOCATIONS, "each mesh local-size dimension checked");
        q = p; q.maxTaskWorkGroupSize[i]--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_INVOCATIONS, "each task local-size dimension checked");
    }
    q = p; q.maxTaskWorkGroupTotalCount--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_GROUP_COUNTS, "task total group-count limit rejected");
    q = p; q.maxMeshWorkGroupTotalCount--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_GROUP_COUNTS, "mesh total group-count limit rejected");
    q = p; q.maxMeshOutputComponents--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_OUTPUT_COMBINED, "output-component limit rejected");
    q = p; q.maxMeshOutputLayers--; check(vkx_mesh_property_reject(&q, 1) & VKX_MESH_GATE_OUTPUT_COMBINED, "output-layer limit rejected");
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_feature_only = f; mesh_feature_only.taskShader = 0;
    q = p; q.maxTaskWorkGroupInvocations = q.maxTaskWorkGroupSize[0] = q.maxTaskPayloadSize = q.maxTaskWorkGroupCount[0] = 0;
    vkx_mesh_caps mesh_device = vkx_mesh_caps_make(1, &mesh_feature_only, &q, 1, 1);
    check(mesh_device.direct_ready && !mesh_device.task, "mesh-only device passes without task limits");

    VkPhysicalDeviceMeshShaderPropertiesEXT ep = p;
    ep.maxMeshWorkGroupInvocations = 1024;
    ep.maxMeshWorkGroupSize[0] = 1024; ep.maxMeshWorkGroupSize[1] = 1024; ep.maxMeshWorkGroupSize[2] = 64;
    // Model queried G13 truth, not one shared provisional clamp: task input is
    // 4096 while the driver may admit only 2048 emitted mesh workgroups.
    ep.maxTaskWorkGroupInvocations = 1024;
    ep.maxTaskWorkGroupSize[0] = 1024; ep.maxTaskWorkGroupSize[1] = 1024; ep.maxTaskWorkGroupSize[2] = 64;
    ep.maxTaskWorkGroupTotalCount = 4096;
    ep.maxMeshWorkGroupTotalCount = 2048;
    for (int i = 0; i < 3; i++) {
        ep.maxTaskWorkGroupCount[i] = 4096;
        ep.maxMeshWorkGroupCount[i] = 2048;
    }
    ep.maxTaskPayloadSize = 4 * 1024;
    ep.maxTaskSharedMemorySize = 28 * 1024;
    ep.maxTaskPayloadAndSharedMemorySize = 32 * 1024;
    ep.maxMeshSharedMemorySize = ep.maxMeshPayloadAndSharedMemorySize = 24 * 1024;
    ep.maxMeshOutputMemorySize = 8 * 1024;
    ep.maxMeshPayloadAndOutputMemorySize = 8 * 1024;
    ep.maxMeshOutputComponents = 4; ep.maxMeshOutputLayers = 1;
    ep.maxMeshOutputVertices = 256;
    ep.maxMeshOutputPrimitives = 256;
    VkPhysicalDeviceMeshShaderFeaturesEXT g13 = f; g13.taskShader = 0;
    vkx_mesh_caps exp = vkx_mesh_title_caps_make(1, 1, &g13, &ep, 1, 1);
    check(exp.experimental && exp.enable_extension && exp.direct_ready && exp.indirect_ready && !exp.indirect_task_only && !exp.task && !exp.runtime_ready,
          "explicit G13 title profile recognizes mesh direct/indirect but remains runtime closed");
    check(vkx_mesh_indirect_stage_allowed(&exp, 0) && !vkx_mesh_indirect_stage_allowed(&exp, 1) &&
          vkx_mesh_indirect_stage_allowed(&c, 0),
          "experimental taskless indirect is admitted only after the runtime proof gate");
    vkx_mesh_caps exp_future_task_bit = vkx_mesh_title_caps_make(1, 1, &f, &ep, 1, 1);
    check(exp_future_task_bit.direct_ready && exp_future_task_bit.indirect_ready && exp_future_task_bit.task,
          "experimental title profile recognizes reported task plus exact title-sized limits");
    check(vkx_mesh_indirect_stage_allowed(&exp_future_task_bit, 0) &&
          vkx_mesh_indirect_stage_allowed(&exp_future_task_bit, 1),
          "experimental indirect admits both taskless and task-bound programs");
    uint32_t dispatch_x = 0, dispatch_total = 0;
    check(vkx_mesh_dispatch_limits(&exp_future_task_bit, 1, &dispatch_x, &dispatch_total) &&
          dispatch_x == 4096 && dispatch_total == 4096,
          "task-bound commands derive the queried 4096 task-input limit");
    check(vkx_mesh_dispatch_limits(&exp_future_task_bit, 0, &dispatch_x, &dispatch_total) &&
          dispatch_x == 2048 && dispatch_total == 2048,
          "taskless commands derive the separate queried 2048 mesh-emission limit");
    check(!vkx_mesh_dispatch_limits(NULL, 1, &dispatch_x, &dispatch_total),
          "missing capability properties fail the dispatch-limit boundary closed");
    check(!vkx_mesh_caps_make(1, &g13, &ep, 1, 1).enable_extension,
          "experimental title limits never weaken the conformant gate");
    check(!vkx_mesh_title_caps_make(0, 1, &g13, &ep, 1, 1).direct_ready &&
          !vkx_mesh_title_caps_make(1, 1, &g13, &ep, 0, 1).direct_ready,
          "title profile requires explicit opt-in and a direct function");
    q = ep; q.maxMeshOutputPrimitives = VKX_MESH_TITLE_PRIMITIVES - 1;
    check(!vkx_mesh_title_caps_make(1, 1, &g13, &q, 1, 1).direct_ready,
          "title profile rejects properties below its exact inventory maxima");
    q = ep; q.maxTaskPayloadSize = VKX_MESH_TITLE_PAYLOAD - 1;
    check(vkx_mesh_title_caps_make(1, 1, &f, &q, 1, 1).direct_ready &&
          !vkx_mesh_title_caps_make(1, 1, &f, &q, 1, 1).task,
          "title profile rejects task only when reported task limits are too small");
    check(!vkx_mesh_proof_gate(NULL) && !vkx_mesh_proof_gate("1") &&
          vkx_mesh_proof_gate(VKX_MESH_PROOF_TOKEN),
          "only the exact root physical-proof token opens the final gate");
    vkx_mesh_caps gated = exp_future_task_bit;
    check(!vkx_mesh_active_caps(&c, &gated), "driver-ready caps are inactive without runtime proof");
    gated.runtime_ready = 1;
    check(vkx_mesh_active_caps(&c, &gated) == &gated,
          "the explicitly proof-marked title caps become the sole active boundary");

    vkx_mesh_caps run = ready_caps();
    const char *tj = "{\"stage\":\"ts\",\"local\":[32,1,1],\"shared\":64,\"mesh\":[0,0,3200],\"mesh_has_task\":false}";
    const char *mj = "{\"stage\":\"ms\",\"local\":[32,1,1],\"shared\":96,\"mesh\":[160,128,3200,0,1],\"mesh_has_task\":true}";
    vkx_mesh_stage_meta parsed;
    check(vkx_mesh_job_meta(tj, VKX_STAGE_TS, &parsed) && parsed.host_slot == 3 && parsed.descriptor_set == 3 &&
          parsed.local[0] == 32 && parsed.payload == 3200 && parsed.shared == 64 && parsed.mesh_has_task_known && !parsed.mesh_has_task,
          "task job metadata forms actual stage/set 3");
    check(vkx_mesh_job_meta(mj, VKX_STAGE_MS, &parsed) && parsed.host_slot == 4 && parsed.descriptor_set == 4 &&
          parsed.max_vertices == 160 && parsed.max_primitives == 128 && parsed.payload == 3200 && parsed.shared == 96 && parsed.mesh_has_task,
          "mesh job metadata forms actual stage/set 4");
    check(!vkx_mesh_job_meta(tj, VKX_STAGE_MS, &parsed) &&
          !vkx_mesh_job_meta("{\"stage\":\"ms\",\"local\":[0,1,1],\"shared\":0,\"mesh\":[1,1,0]}", VKX_STAGE_MS, &parsed),
          "job stage mismatch and zero local size fail closed");
    vkx_mesh_stage_meta ts = {.present = 1, .stage = VKX_STAGE_TS, .host_slot = 3, .descriptor_set = 3,
                              .mesh_first = 1, .local = {32, 1, 1}, .payload = 3200};
    vkx_mesh_stage_meta ms = {.present = 1, .stage = VKX_STAGE_MS, .host_slot = 4, .descriptor_set = 4,
                              .mesh_has_task = 1, .mesh_has_task_known = 1,
                              .local = {32, 1, 1}, .payload = 3200, .max_vertices = 160, .max_primitives = 128,
                              .per_primitive_locations = 1u << 7};
    vkx_mesh_stage_meta fs = {.present = 1, .stage = VKX_STAGE_PS, .host_slot = 1, .descriptor_set = 1,
                              .per_primitive_locations = 1u << 7};
    vkx_mesh_program_plan plan = vkx_mesh_program(&run, &ts, &ms, &fs);
    check(!plan.reject && plan.count == 3 && plan.stage[0] == VKX_STAGE_TS && plan.stage[1] == VKX_STAGE_MS && plan.stage[2] == VKX_STAGE_PS &&
          plan.host_slot[0] == 3 && plan.host_slot[1] == 4 && plan.descriptor_set[0] == 3 && plan.descriptor_set[1] == 4,
          "task+mesh+fragment forms from actual stages in host slots/sets 3/4/1");
    vkx_mesh_stage_meta ms_no_task = ms; ms_no_task.payload = 0; ms_no_task.mesh_has_task = 0; ms_no_task.mesh_first = 1;
    plan = vkx_mesh_program(&run, NULL, &ms_no_task, &fs);
    check(!plan.reject && plan.count == 2 && plan.stage[0] == VKX_STAGE_MS && plan.stage[1] == VKX_STAGE_PS,
          "mesh+fragment forms without a vertex or task stage");
    check(vkx_mesh_program(&run, &ts, &ms_no_task, &fs).reject & VKX_MESH_PROGRAM_FIRST_INTERFACE,
          "a stale task binding cannot take ownership from a taskless mesh program");
    vkx_mesh_caps exp_run = exp; exp_run.runtime_ready = 1;
    vkx_mesh_stage_meta g13_module = ms_no_task; g13_module.output_memory = 4096; g13_module.output_components = 4;
    plan = vkx_mesh_program(&exp_run, NULL, &g13_module, &fs);
    check(!plan.reject && plan.count == 2,
          "hypothetical title profile admits only exact mesh+fragment metadata within G13 limits");
    vkx_mesh_caps exp_task_run = exp_future_task_bit; exp_task_run.runtime_ready = 1;
    vkx_mesh_stage_meta g13_position_only = ms;
    g13_position_only.output_memory = 2536; g13_position_only.output_components = 0;
    plan = vkx_mesh_program(&exp_task_run, &ts, &g13_position_only, &fs);
    check(!plan.reject && plan.count == 3,
          "physically proven position-only task+mesh+fragment admits zero user output components");
    check(vkx_mesh_program(&exp_run, NULL, &ms_no_task, &fs).reject & VKX_MESH_PROGRAM_LIMIT,
          "experimental profile still rejects unknown zero output-memory reflection metadata");
    check(vkx_mesh_program(&exp_run, &ts, &ms, &fs).reject & VKX_MESH_PROGRAM_NO_TASK_FEATURE,
          "G13 title profile cannot form a task program");
    vkx_mesh_stage_meta g13_over = g13_module; g13_over.output_memory = 8193;
    check(vkx_mesh_program(&exp_run, NULL, &g13_over, &fs).reject & VKX_MESH_PROGRAM_LIMIT,
          "G13 title profile rejects a module over its exact output-memory limit");
    g13_over = g13_module; g13_over.output_components = 5;
    check(vkx_mesh_program(&exp_run, NULL, &g13_over, &fs).reject & VKX_MESH_PROGRAM_LIMIT,
          "G13 title profile rejects unfinished arbitrary-varying breadth");
    vkx_mesh_stage_meta bad_payload = ms; bad_payload.payload--;
    check(vkx_mesh_program(&run, &ts, &bad_payload, &fs).reject & VKX_MESH_PROGRAM_PAYLOAD_INTERFACE,
          "task/mesh payload interface mismatch rejected");
    check(vkx_mesh_program(&run, NULL, &ms, &fs).reject & VKX_MESH_PROGRAM_PAYLOAD_INTERFACE,
          "mesh-only program cannot consume an unwritten task payload");
    vkx_mesh_stage_meta double_first = ms; double_first.mesh_first = 1;
    check(vkx_mesh_program(&run, &ts, &double_first, &fs).reject & VKX_MESH_PROGRAM_FIRST_INTERFACE,
          "task-fed mesh cannot own mesh_first a second time");
    vkx_mesh_caps closed = run; closed.runtime_ready = 0;
    check(vkx_mesh_program(&closed, &ts, &ms, &fs).reject & VKX_MESH_PROGRAM_NO_RUNTIME,
          "renderer proof gate keeps program closed");
    vkx_mesh_caps mesh_only = run; mesh_only.task = 0;
    check(vkx_mesh_program(&mesh_only, &ts, &ms, &fs).reject & VKX_MESH_PROGRAM_NO_TASK_FEATURE,
          "task program rejected when taskShader is absent");
    vkx_mesh_stage_meta fake_ms = ms; fake_ms.stage = VKX_STAGE_DS;
    check(vkx_mesh_program(&run, NULL, &fake_ms, &fs).reject & VKX_MESH_PROGRAM_BAD_STAGE,
          "host slot 4 domain shader cannot masquerade as mesh");
    vkx_mesh_stage_meta too_big = ms; too_big.local[0] = VKX_MESH_REQ_INVOCATIONS + 1;
    check(vkx_mesh_program(&run, NULL, &too_big, &fs).reject & VKX_MESH_PROGRAM_LIMIT,
          "per-program workgroup limit rejected");
    too_big = ms; too_big.max_vertices = VKX_MESH_REQ_VERTICES + 1;
    check(vkx_mesh_program(&run, NULL, &too_big, &fs).reject & VKX_MESH_PROGRAM_LIMIT,
          "per-program output limit rejected");
    vkx_mesh_stage_meta bad_fs = fs; bad_fs.per_primitive_locations |= 1u << 9;
    check(vkx_mesh_program(&run, &ts, &ms, &bad_fs).reject & VKX_MESH_PROGRAM_PRIMITIVE_INTERFACE,
          "fragment per-primitive input without a mesh output rejected");

    uint32_t spv[] = {0x07230203, 0x00010600, 0, 32, 0,
                      (3u << 16) | 71u, 10, 5271,
                      (4u << 16) | 71u, 10, 30, 7};
    uint32_t pm = 0;
    check(vkx_spv_per_primitive_locations(spv, sizeof spv / sizeof *spv, &pm) && pm == (1u << 7),
          "PerPrimitiveEXT locations are reflected from SPIR-V");
    spv[5] = 71; check(!vkx_spv_per_primitive_locations(spv, sizeof spv / sizeof *spv, &pm),
                       "malformed SPIR-V fails closed");

    check(vkx_stage_c0_ssbo_base(VKX_STAGE_TS) == 0x670 && vkx_stage_c0_ssbo_base(VKX_STAGE_MS) == 0x770 &&
          vkx_stage_c0_ssbo_base(VKX_STAGE_HS) < 0 && vkx_stage_c0_ssbo_base(VKX_STAGE_DS) < 0,
          "c0 base follows actual task/mesh stage, not remapped hull/domain slot");
    check(vkx_program_host_mask(0xc0) == ((1u << 3) | (1u << 4)),
          "raw BindProgram task/mesh bits remain remapped");

    vkx_mesh_draw_meta dm;
    check(vkx_mesh_draw_direct(9, 23, 64, 64, &dm) && dm.first == 9 && dm.count == 23 && dm.needs_first_base,
          "direct NVN {first,count} metadata is preserved");
    check(!vkx_mesh_draw_direct(0, 65, 64, 128, &dm) && !vkx_mesh_draw_direct(0, 65, 128, 64, &dm) &&
          !vkx_mesh_draw_direct(UINT32_MAX, 2, UINT32_MAX, UINT32_MAX, &dm),
          "direct draw limit and first+count overflow rejected");
    check(vkx_mesh_draw_indirect(0x12340000, 65535, 1u << 22, &dm) && dm.indirect && dm.guest_count_offset == 0 &&
          dm.guest_first_offset == 4 && dm.guest_stride == 8 && dm.vulkan_stride == 12 &&
          dm.needs_gpu_conversion && dm.needs_first_base,
          "GPU-written NVN {count,first} indirect record requires conversion");
    vkx_mesh_first_plan fp = vkx_mesh_first(&dm, 1);
    check(fp.abi_slot == DMSH_MESH_FIRST_SLOT && fp.descriptor_binding == DMSH_LINUX_VK_MESH_FIRST_BINDING &&
          fp.byte_size == 256 && fp.alignment == 256 && fp.owner_stage == VKX_STAGE_TS &&
          fp.source == VKX_MESH_FIRST_INDIRECT_WORD && fp.guest_address == 0x12340004 &&
          fp.add_exactly_once && fp.submit_lifetime && fp.translator_access_ready,
          "indirect first is owned exactly once by task at dedicated binding 512 with submit-lifetime UBO storage");
    fp = vkx_mesh_first(&dm, 0);
    check(fp.owner_stage == VKX_STAGE_MS && fp.add_exactly_once,
          "mesh owns first exactly once only when there is no task stage");
    vkx_mesh_indirect_bridge bridge;
    vkx_mesh_indirect_layout il = {.src_size = 0x100, .src_offset = 0x80,
        .dst_size = 0x400, .record_offset = 0x200, .first_offset = 0x100};
    check(vkx_mesh_indirect_plan(&dm, 1, &il, &bridge) &&
          bridge.vk_record_yz[0] == 1 && bridge.vk_record_yz[1] == 1 &&
          bridge.record_init_offset == 0x204 && bridge.record_init_size == 8 &&
          bridge.count.src == 0x80 && bridge.count.dst == 0x200 && bridge.count.bytes == 4 &&
          bridge.first.src == 0x84 && bridge.first.dst == 0x100 && bridge.first.bytes == 4 &&
          bridge.vk_record_size == 12 && bridge.first_scratch_size == 256 && bridge.record_init_transfer &&
          bridge.transfer_to_indirect_barrier && bridge.transfer_to_shader_barrier &&
          bridge.first_plan.owner_stage == VKX_STAGE_TS && bridge.first_plan.translator_access_ready,
          "GPU bridge initializes disjoint Y/Z, copies {count,first} without readback and specifies both barriers/lifetimes");
    vkx_mesh_indirect_layout bad_il = il; bad_il.src_size = 0x87;
    check(!vkx_mesh_indirect_plan(&dm, 1, &bad_il, &bridge), "indirect source bounds fail closed");
    bad_il = il; bad_il.dst_size = 0x20b;
    check(!vkx_mesh_indirect_plan(&dm, 1, &bad_il, &bridge), "indirect destination bounds fail closed");
    bad_il = il; bad_il.first_offset = 0x101;
    check(!vkx_mesh_indirect_plan(&dm, 1, &bad_il, &bridge), "mesh_first UBO alignment fails closed");
    bad_il = il; bad_il.record_offset = 0x100;
    check(!vkx_mesh_indirect_plan(&dm, 1, &bad_il, &bridge), "record/first overlap fails closed");
    bad_il = il; bad_il.same_buffer = 1; bad_il.src_offset = 0x100; bad_il.src_size = 0x400;
    check(!vkx_mesh_indirect_plan(&dm, 1, &bad_il, &bridge), "same-buffer copy overlap fails closed");
    check(!vkx_mesh_draw_indirect(UINT64_MAX, 1, 1, &dm),
          "indirect source address overflow fails closed");
    check(vkx_mesh_draw_direct(9, 23, 64, 64, &dm), "restore direct metadata");
    fp = vkx_mesh_first(&dm, 0);
    check(fp.source == VKX_MESH_FIRST_DIRECT_WORD && fp.direct_value == 9 && fp.owner_stage == VKX_STAGE_MS &&
          fp.add_exactly_once && fp.translator_access_ready,
          "nonzero direct first propagates to mesh-owned ABI word while renderer runtime_ready stays closed");
    check(vkx_mesh_first_descriptor(512, 'u', "mesh_first") &&
          !vkx_mesh_first_descriptor(29, 'u', "mesh_first") &&
          !vkx_mesh_first_descriptor(512, 's', "mesh_first"),
          "runtime recognizes only the exact binding-512 uniform mesh_first descriptor ABI");

    vkx_mesh_command_plan cp;
    check(vkx_mesh_command_plan_make(&dm, 0, NULL, &cp) && !cp.indirect &&
          cp.owner_stage == VKX_STAGE_MS && cp.group_count == 23 && cp.first.direct_value == 9 &&
          cp.copy_src_stage == VK_PIPELINE_STAGE_HOST_BIT && cp.copy_src_access == VK_ACCESS_HOST_WRITE_BIT &&
          cp.owner_dst_stage == VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT && cp.owner_dst_access == VK_ACCESS_UNIFORM_READ_BIT &&
          cp.nstep == 6 && cp.step[0] == VKX_MESH_CMD_ALLOC_FIRST &&
          cp.step[1] == VKX_MESH_CMD_WRITE_FIRST && cp.step[2] == VKX_MESH_CMD_BARRIER_OWNER &&
          cp.step[3] == VKX_MESH_CMD_BEGIN_RENDERING && cp.step[4] == VKX_MESH_CMD_PREPARE_PROGRAM &&
          cp.step[5] == VKX_MESH_CMD_DRAW_DIRECT,
          "direct first=9 command plan keeps submit lifetime and records count/1/1 after owner visibility");
    check(vkx_mesh_draw_direct(0, 1, 64, 64, &dm) &&
          vkx_mesh_command_plan_make(&dm, 0, NULL, &cp) &&
          cp.first.source == VKX_MESH_FIRST_DIRECT_WORD && cp.first.direct_value == 0 &&
          cp.first.submit_lifetime && cp.group_count == 1,
          "direct first=0 still gets a lifetime-safe owner UBO and one group");
    check(vkx_mesh_draw_indirect(0x12340000, 65535, 1u << 22, &dm) &&
          vkx_mesh_command_plan_make(&dm, 1, &il, &cp) && cp.indirect &&
          cp.owner_stage == VKX_STAGE_TS && cp.copy_src_stage == VK_PIPELINE_STAGE_TRANSFER_BIT &&
          cp.indirect_dst_stage == VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT &&
          cp.owner_dst_stage == VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT &&
          cp.indirect_dst_access == VK_ACCESS_INDIRECT_COMMAND_READ_BIT &&
          cp.owner_dst_access == VK_ACCESS_UNIFORM_READ_BIT && cp.nstep == 10 &&
          cp.step[1] == VKX_MESH_CMD_ALLOC_INDIRECT && cp.step[2] == VKX_MESH_CMD_INIT_INDIRECT &&
          cp.step[3] == VKX_MESH_CMD_COPY_COUNT && cp.step[4] == VKX_MESH_CMD_COPY_FIRST &&
          cp.step[5] == VKX_MESH_CMD_BARRIER_INDIRECT && cp.step[6] == VKX_MESH_CMD_BARRIER_OWNER &&
          cp.step[7] == VKX_MESH_CMD_BEGIN_RENDERING && cp.step[8] == VKX_MESH_CMD_PREPARE_PROGRAM &&
          cp.step[9] == VKX_MESH_CMD_DRAW_INDIRECT,
          "indirect command plan orders init/copies/exact barriers before rendering and task-owned issue");

    _Atomic unsigned trace_seen = 0; char trace[320];
    int trace_stage[3] = {VKX_STAGE_MS, VKX_STAGE_PS, VKX_STAGE_VS};
    const char trace_key[3][17] = {"554cf8c05126a133", "1122334455667788", "0011223344556677"};
    char picked[17] = "-";
    check(vkx_mesh_trace_pick(trace_stage, trace_key, 3, VKX_STAGE_PS, picked) && !strcmp(picked, "1122334455667788"),
          "mixed-stage telemetry selects only the requested key");
    strcpy(picked, "-");
    check(!vkx_mesh_trace_pick(trace_stage, trace_key, 3, VKX_STAGE_TS, picked) && !strcmp(picked, "-"),
          "no-match telemetry leaves the caller's dash untouched");
    vkx_mesh_trace_event te = {.task_key = "efaa4fa6a80ba719", .mesh_key = "554cf8c05126a133", .fragment_key = "1122334455667788", .first = 9, .count = 23};
    check(vkx_mesh_trace_take(&trace_seen, 0) && !vkx_mesh_trace_take(&trace_seen, 0) &&
          vkx_mesh_trace_format(trace, sizeof trace, &te) > 0 && strstr(trace, "first=9 count=23") && strstr(trace, "ps=1122334455667788"),
          "direct mesh boundary telemetry is one-shot and contains bound keys/{first,count}");
    te.indirect = 1; te.address = 0x12340000;
    check(vkx_mesh_trace_take(&trace_seen, 1) && !vkx_mesh_trace_take(&trace_seen, 1) &&
          vkx_mesh_trace_format(trace, sizeof trace, &te) > 0 && strstr(trace, "address=0x12340000") && strstr(trace, "count@+0,first@+4,stride=8"),
          "indirect mesh boundary telemetry is independently one-shot and names the GPU record ABI");

    inventory(argc, argv);
    printf("t_vk_mesh: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
