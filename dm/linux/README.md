# Linux/Asahi renderer bring-up

This directory forward-ports the Android host's ELF compatibility boundary and
graphics backends to current nxnative. Runtime owns the Linux loader, services,
audio, movies, and thread platform; this tree builds the current NVN/NVN2 layer
against a null or Vulkan renderer without duplicating that work.

The bounded null link needs no game content. It uses only the tracked game
profile to select compile-time NVN behavior:

```sh
make -C dm/linux GAME=0400486024DEC000 null-renderer
```

The result, `gen/<id>/libdm-renderer-null.a`, is a relocatable renderer archive.
Undefined symbols in it are the explicit host/runtime integration seam.

## Native task/mesh staging

`dmsh_spv.py` omits NVN2 task and mesh jobs by default. A native Vulkan
translator can provide `<key>.spv`, `.res.json`, and `.job.json` through
`DMSH_VK_MESH_EXT_DIR`; the tool imports a key only when all metadata agrees
with its TaskEXT or MeshEXT execution model and the module contains the
`SPV_EXT_mesh_shader` capability, extension, and required stage operation.
Task uses descriptor set 3 and mesh set 4, matching this host's remap of guest
stages 6/7 into replay state slots 3/4.

`DMSH_VK_MESH_XLATE` is the higher-level, provenance-checked route: it names a
separately built native mesh translator, and `dmsh_spv.py` derives the original
NVN2 jobs from the package, requires `spirv-val --target-env vulkan1.3`, writes
`_native_provenance.json`, then feeds that directory through the same strict
importer. The validated BananaNative implementation was introduced by commit
`96cc0e883c80117e51342c60facfae2f0e600b9a`; its stage-memory decoder makes
the combined translator GPL-3.0-or-later, so its source and binary remain
outside this MIT tree. `SPIRV_VAL` may name the validator;
the embedded source hash is pinned and cannot be overridden. For title-wide
translation, `DMSH_VK_MESH_MAP` and `DMSH_VK_MESH_FACTS` can name a matching
fresh split (the latter carries pixel-pair-derived per-primitive and padded
mesh outputs), while `DMSH_VK_MESH_BATCH` bounds each external process.
`DMSH_VK_MESH_ALLOW_FAILURES=1` retains validated successes and writes exact
per-key failures to `_native_status.json`; it does not weaken the importer.

The Linux device boundary queries `VK_EXT_mesh_shader`, but its conformant
path requests the extension only when the feature, every Khronos
`Required_Limits` property, and (after device creation) the relevant function
pointer pass. Direct and indirect readiness are separate. The loader then
requires the job stage/local metadata to agree with the emitted SPIR-V and
reflects the actual Workgroup, TaskPayloadWorkgroupEXT, output-memory and
user-output-component footprint before program admission. Task uses c0 base
`0x670` from host slot/set 3 and mesh uses `0x770` from slot/set 4; hull/domain
metadata in the same host slots cannot masquerade as the native stages.

`DM_VK_MESH_TITLE_PROFILE=1` is a separate, explicit model for the current
non-conformant G13 development driver. It is direct-only and mesh-only, checks
the Cycle-29 title maxima and requires nonzero reflected output memory and
component metadata for each hypothetical program. It never weakens the
conformant gate, never enables task, and is not consumed by program
preparation. Both profiles leave `runtime_ready` hard zero: `vk_replay` has no
mesh command call, so merely placing modules in `DMSH_VK_MESH_EXT_DIR` cannot execute
unsupported work. The VS-free task(optional)+mesh+fragment preparation
function is present but unreachable until a later renderer-advertisement
change after physical proof.

The Linux resolver keeps the ordinary `DM_VK_SPV` set and the native
`DMSH_VK_MESH_EXT_DIR` separate. It accepts task/mesh only from the latter,
requires the exact `.vk.spv`/`.vkb`/`.job.json` triplet with set 3/4 agreement,
and never adds native keys to the background precompiler while the proof gate
is closed. `t_vk_shader_resolve --title <standard-dir> <mesh-dir>` audits the
untouched 876-module standard set and segregated 506-module Cycle-29 set.

`DM_VK_MESH_LOG=1` emits at most two boundary lines: the first direct and the
first indirect mesh draw. Each line records the actually bound task, mesh and
fragment keys. Direct records include `{first,count}`; indirect records include
the guest address and exact `{count@+0,first@+4,stride=8}` layout. Stage identity
is retained in capture padding without changing `cap_prog` size, so a new
capture remains equally bounded on replay. The option is telemetry only.

`DM_VK_MESH_TASKLESS=0` is a temporary, explicit low-memory bring-up gate for
the experimental title profile. It rejects taskless indirect mesh commands
while retaining native task-bound MeshEXT commands; unset or `1` leaves the
proved taskless path enabled. It is a stopgap for the current fixed guest-memory
footprint, not a compute or ordinary-draw fallback.

The no-readback bridge initializes a
12-byte Vulkan record to `{0,1,1}`, copy the guest count word to X and the first
word to a 256-byte/aligned submit-lifetime uniform scratch allocation, then
apply transfer-to-indirect-command and transfer-to-owner-shader-read barriers.
Metal `DMSH_MESH_FIRST_SLOT=29` is semantic lineage only: Linux reserves the
collision-free uniform binding 512 (`mesh_first`) because binding 29 is already
a storage buffer in the title union. The MIT postprocessor adds
`effective WorkGroupId.x = WorkGroupId.x + first` exactly once in task when
present, otherwise taskless mesh. Paired mesh and the one owner task that never
reads WorkGroupId stay byte-identical and have no descriptor row.

The dormant command path allocates binding 512 from the current aligned
command-ring arena, whose slot is reused only after its timeline value
completes. Direct writes `first`, applies a host-to-owner uniform dependency,
and records `vkCmdDrawMeshTasksEXT(count,1,1)`. Indirect performs the two
four-byte GPU copies outside dynamic rendering, applies the exact
transfer-to-indirect and transfer-to-task/mesh-uniform dependencies, then
records `vkCmdDrawMeshTasksIndirectEXT`.

The experimental G13 boundary accepts indirect commands only when task support
was queried and a real task stage is bound; taskless indirect remains closed because the current driver
path rejects it. Dispatch bounds always come from the queried stage-specific
properties (currently 4096 task inputs versus 2048 emitted mesh workgroups),
not a shared application constant. A failed render begin, program preparation,
or command check clears the pending binding-512 owner descriptor before replay
continues; its arena allocation is still retained until timeline completion.
The Vulkan indirect record's disjoint Y/Z range is initialized to `{1,1}` with
`vkCmdUpdateBuffer(record+4, 8)`; X has only the guest-count copy as a writer.
The same transfer-to-indirect barrier covers all three words, without an
overlapping-write dependency or implicit host visibility. The indirect first
word likewise has only its GPU copy as a writer; host writes are reserved for
the direct-first path.

The command path is unreachable by default. A conformant or explicit title
profile must pass its extension/feature/property/function gates, and the root
physical proof must then provide the exact opt-in
`DM_VK_MESH_PROOF=root-physical-proof`. Any other value leaves
`runtime_ready=0`; the token cannot override missing task or indirect support.

`DM_VK_PRECOMPILE_ONLY=1` uses every online CPU by default. Setting
`DM_VK_PRECOMPILE=<threads>` explicitly also bounds tool mode; this is useful
for driver bring-up where concurrent shader-object compilation is not yet
proven, and does not change the ordinary all-core default.
`DM_VK_PRECOMPILE_KEYS=<key[,key...]>` additionally limits tool mode to exact
16-hex shader keys while retaining the full production descriptor layout.

The offline fixture test builds genuine TaskEXT and MeshEXT modules with
`glslc`; the C tests prove stage/slot/set/c0 mapping, every negative device
gate, program/interface/limit rejection, direct `{count,first}` preservation,
indirect GPU-conversion metadata, and default closure:

```sh
python3 dm/gfx/tests/test_dmsh_spv_stage.py
python3 dm/gfx/tests/test_dmsh_spv_mesh_first.py
make -C dm/linux GAME=0400486024DEC000 t_vk_stage t_vk_mesh t_vk_shader_resolve
```

For the segregated Cycle-29 output, `t_vk_mesh --inventory 19 487
<spv-dir>/*.job.json` additionally checks all 506 job/VKB/canonical-SPIR-V
triples without opening a Vulkan device. GPL donor code and binaries remain
outside this MIT runtime tree.

## Bounded NDVE diagnostic

`DM_VK_SKIP_CS` is an opt-in exact-key diagnostic for isolating expensive or
suspect compute kernels. With the variable unset or empty, every compute
dispatch follows the normal path. Values are comma- or whitespace-separated
16-hex-digit shader keys; invalid entries are rejected, and there is no size or
prefix matching. A selected dispatch is dropped: the replay does not replace
it with a blit, scaler, tone map or another title-specific operation.

The first M1 Pro/Honeykrisp investigation isolated four measured NDVE
movie-enhancement kernels with:

```sh
DM_VK_SKIP_CS=d15370a382630d4e,e6be82d835fbe4d3,9442573ee4ace011,0a56002cdc7e1fce
```

The movie chain is incomplete under that diagnostic. Each selected key is
logged once, and `DM_VK_OPS=1` reports the cumulative number of deliberately
skipped dispatches. This is containment for a bounded investigation, not a
general performance switch or a rendering fallback. The root launcher leaves
the variable empty so the real NDVE movie and NDSR gameplay chains run. The
parser and disabled default have a device-free check:

```sh
make -C dm/linux GAME=0400486024DEC000 t_skipcs
```

## Vulkan compute lowering and timing

The Linux SPIR-V build lowers the translator's named
`dmsh_hmma16816_f16` helper only when its current 18-pointer ABI, private
workgroup scratch layout, barriers and unused TensorPass facts all match. The
lowering gathers A/B fragments through vector workgroup loads, reuses equal
fragments within a basic block, preserves the helper's ordered FP32 FMA and
FP16 packing, and removes the scalar helper. A named helper with a changed ABI
fails the key instead of silently retaining the slow path. Every changed
module requires `spirv-val --target-env vulkan1.3` before it is published.

A second conservative pass coalesces aligned pairs or quads of scalar loads
from a reflected read-only storage buffer. It admits only a 32-bit,
stride-four descriptor with a complete no-write/no-alias proof, keeps original
SSA result IDs at their original program points, and closes its candidate run
at every unclassified operation. Existing SPIR-V vector and pointer types are
reused. The canonical module and its `.vkb` table are published as one
fail-closed unit, and both passes participate in `DMSH_CHAIN`.

The portable fixtures cover ABI drift, helper semantics, SPIR-V validation,
pointer aliasing, write detection, alignment and type reuse:

```sh
SPIRV_AS="$(command -v spirv-as)" \
SPIRV_VAL="$(command -v spirv-val)" \
python3 dm/gfx/tests/test_dmsh_spv_hmma.py

DMSH_FIXTURE_TOOLS="$(dirname "$(command -v spirv-val)")" \
python3 dm/gfx/tests/test_dmsh_spv_vector.py
```

`DM_VK_CS_GPU=1` adds timestamp pairs around existing Vulkan compute
dispatches without splitting command buffers or adding barriers or waits.
Completed submissions log shader key, frame, group dimensions and GPU time as
`[vk-cs-gpu]`; `DM_VK_CS_GPU_LIMIT` bounds the total (4,096 by default,
65,536 maximum), and each command-buffer slot holds at most 512 samples. The
publication/reuse lifecycle has a device-free check:

```sh
make -C dm/linux GAME=0400486024DEC000 t_vk_cs_gpu_lifecycle
```

## Compatible 32-bit image views

NVN shaders may address a compatible 32-bit colour image through an R32UI or
R32I view. Ocarina's NDSR chain depends on this: `d17eefaff4f0bc23` performs
integer image atomics on an RGBA8 target. The Vulkan replay therefore declares
the supported R32 integer views in the image's `VkImageFormatListCreateInfo`,
uses mutable/extended usage at image creation, and binds the guest-requested
integer view to the same image allocation. It does not substitute a buffer or
copy the pixels into a title-specific side image.

Only uncompressed, non-depth members of Vulkan's 32-bit format compatibility
class are admitted, and only when the device reports storage-image support for
the integer view. The format selection boundary has a device-free check:

```sh
make -C dm/linux GAME=0400486024DEC000 t_vk_format
```

## Focused Vulkan diagnostics

`DM_VK_GMEM_TRACE=1` extends a translated raw-global store bitmap with a
bounded list of full 64-bit guest page numbers. After the owning timeline fence
completes, the host invalidates only resident images whose guest storage
overlaps one of those pages. It is disabled by default; without it the original
bitmap ABI and behavior are unchanged. This is a page-overlap discriminator
and a future-sample coherence aid, not same-submission store-to-texture
coherence. The SPIR-V rewrite and consumer have a device-free check:

```sh
make -C dm/linux GAME=0400486024DEC000 t_gmem_trace
```

`DM_VK_MISSING_LOG=<file>` appends each untranslated shader key once as
`<16 hex>\t<program index>`. It does not enable `DM_VK_DRAWLOG`; the process
deduplicates keys and stops appending after 4096 unique entries. Use a fresh
file for each route when its complete finite key set is needed. Its bounded set
and line format have a device-free check:

```sh
make -C dm/linux GAME=0400486024DEC000 t_missing_log
```
