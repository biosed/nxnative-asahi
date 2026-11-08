// Linux build: what the Vulkan backend needs to know about a translated shader's SPIR-V (dm/gfx/tools/dmsh_spv.py) to build its descriptor set and pipeline layouts
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#define SPVR_MAX_BIND 96
#define SPVR_MAX_OUTPUT 128
typedef struct { uint32_t set, binding, count; VkDescriptorType type; const char *name; } spvr_binding;   // count 0 = a runtime array
enum { SPVR_TOPOLOGY_NONE, SPVR_TOPOLOGY_POINTS, SPVR_TOPOLOGY_LINES, SPVR_TOPOLOGY_TRIANGLES };
enum {
    SPVR_BUILTIN_POSITION = 1u << 0, SPVR_BUILTIN_POINT_SIZE = 1u << 1,
    SPVR_BUILTIN_CLIP_DISTANCE = 1u << 2, SPVR_BUILTIN_CULL_DISTANCE = 1u << 3,
    SPVR_BUILTIN_PRIMITIVE_ID = 1u << 4, SPVR_BUILTIN_LAYER = 1u << 5,
    SPVR_BUILTIN_VIEWPORT_INDEX = 1u << 6, SPVR_BUILTIN_CULL_PRIMITIVE = 1u << 7,
    SPVR_BUILTIN_PRIMITIVE_SHADING_RATE = 1u << 8, SPVR_BUILTIN_OTHER = 1u << 31,
};
typedef struct {
    uint16_t location, component, components, bits;
    uint8_t per_primitive;
    char kind;                    // f=float, i=signed integer, u=unsigned integer, b=bool, ?=mixed/unknown
} spvr_output;
typedef struct {
    VkShaderStageFlagBits stage;   // from the entry point's execution model; 0 = unknown
    int nbind; spvr_binding bind[SPVR_MAX_BIND];
    uint32_t push_size;            // bytes of the push constant block (0 = none)
    uint32_t local_size[3];        // compute: LocalSize
    uint32_t shared_size;          // bytes in emitted Workgroup variables
    uint32_t task_payload_size;    // bytes in TaskPayloadWorkgroupEXT variables
    uint32_t output_size;          // conservative bytes in emitted Output variables
    uint32_t output_components;    // scalar user-output components per mesh vertex/primitive
    uint32_t max_vertices, max_primitives;
    uint32_t output_builtin_mask;
    uint8_t output_topology;
    int noutput; spvr_output output[SPVR_MAX_OUTPUT];
} spvr_info;
// 0 on success; -1 = not SPIR-V or a structure this reader does not know (err says which). Names point into the module (they stay valid while it does)
int spvr_reflect(const uint32_t *code, size_t words, spvr_info *out, char *err, size_t errcap);

// Fit a module to the device: what the translator declares but the shader does not use, the device may lack. Only provably unused declarations are dropped and only
// never-written buffers marked read-only, so the shader's behaviour is unchanged. Returns a malloc'd copy (*out_words), and in *changes what was done (a bit per kind)
typedef struct { int float64, interlock, vertex_stores, nan_preserve; } spvr_device;   // nan_preserve: shaderSignedZeroInfNanPreserveFloat32
#define SPVR_FIT_FLOAT64 1     // OpCapability Float64 dropped: no 64-bit float type in the module
#define SPVR_FIT_INTERLOCK 2   // the fragment shader interlock extension, capabilities and execution modes dropped: no interlock begun
#define SPVR_FIT_NONWRITABLE 4 // vertex-stage storage buffers that nothing writes decorated NonWritable
#define SPVR_FIT_STORES 8      // a vertex-stage storage buffer is written, which the device cannot do (not changed: the caller must not use the shader as is)
#define SPVR_FIT_RELAXED 16    // RelaxedPrecision decorations dropped: Mesa computes such values in 16 bits (Metal did not); the translator's precision hints do not
#define SPVR_FIT_NMINMAX 32  // FMin / FMax / FClamp made NMin / NMax / NClamp: a NaN operand gives the other one, as Maxwell's FMNMX and saturation do
#define SPVR_FIT_NANPRESERVE 64 // SignedZeroInfNanPreserve on every entry point: without it Mesa may fold isnan() to false and reorder away infinities and NaN, which Maxwell keeps
#define SPVR_FIT_VERSION 8     // what the fitted module rewrites change; in the disk cache's directory name (vk_draw.c), so binaries made before a change are not taken
                               // describe Maxwell, which computes everything in 32 bits
uint32_t *spvr_fit(const uint32_t *code, size_t words, const spvr_device *dev, size_t *out_words, unsigned *changes);

// Move the translator's NVN texture/sampler pool arrays out of the per-stage
// set and into one canonical bindless set.  Texture arrays are canonicalised
// by their SPIR-V image shape (2D, 2D-array, depth-2D) because the translator's
// original bindings 564..566 are assigned in per-module discovery order.
// The million-entry texture arrays are bounded to texture_count; the 4096
// sampler array is preserved.  Returns NULL when the module has no pool.
uint32_t *spvr_bindless(const uint32_t *code, size_t words, uint32_t set,
                        uint32_t texture_count, size_t *out_words,
                        unsigned *patched);

// Instrument accesses to the canonical texture arrays in bindless_set with a
// bit in feedback_set/binding 0.  The feedback buffer is uint32_t bits[] and
// the array index is the registered texture ID.  Returns NULL when the module
// has no texture-pool access.
uint32_t *spvr_bindless_feedback(const uint32_t *code, size_t words,
                                 uint32_t bindless_set, uint32_t feedback_set,
                                 size_t *out_words, unsigned *patched);

// GL's clip depth (NVN_DEPTH_MODE_NEAR_IS_MINUS_W, NVN's default) in a module for Vulkan's: before each return of the entry point, gl_Position.z = (z + w) / 2, so -w..w
// becomes 0..w. For the last stage before rasterization (vertex, or geometry / tessellation evaluation). Returns a malloc'd copy, or NULL when the module writes no Position
uint32_t *spvr_gl_depth(const uint32_t *code, size_t words, size_t *out_words);

// Every vec4 output of a vertex or tessellation evaluation shader set to (0, 0, 0, 1) at the start of the entry point: on Maxwell a component the shader never writes reads as 0 in
// the next stage, where Vulkan leaves it undefined. Mario Wonder's tile and model shaders write one component of each tangent-frame output (_41.y = 1.0) and Mesa passed garbage
// for the rest, so the normals overflowed to NaN and lit surfaces came out black. Returns a malloc'd copy, or NULL when there is nothing to do
uint32_t *spvr_init_outputs(const uint32_t *code, size_t words, size_t *out_words);
// Maxwell's non-finite values (spv_reflect.c): Sin / Cos of NaN or of an infinity give NaN, and a NaN user output of any graphics stage is written as 0. A malloc'd copy, or NULL when
// there is nothing to change; changes: 1 Sin / Cos, 2 outputs
uint32_t *spvr_nan(const uint32_t *code, size_t words, size_t *out_words, unsigned *changes);

// Extend a translated raw-global store bitmap (the storage-buffer resource at
// binding) with vk_gmem_trace.h's full guest-page append. The original bitmap
// atomic remains. Returns a malloc'd module and the number of patched atomics,
// or NULL when the translator's expected shape is not present.
uint32_t *spvr_gmem_trace(const uint32_t *code, size_t words, uint32_t binding,
                          size_t *out_words, unsigned *patched);
