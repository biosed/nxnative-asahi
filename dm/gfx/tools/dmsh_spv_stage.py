"""Native Vulkan shader-stage metadata used by the Linux SPIR-V tool.

Task and mesh inputs are deliberately external to the Metal translation chain:
the current pinned translator lowers those jobs to compute.  A native Vulkan
translator may place ``<key>.{spv,res.json,job.json}`` in a separate directory;
``import_native_translation`` admits it only when the job, reflection and
SPIR-V execution model all name the same TaskEXT or MeshEXT stage.
"""
import json
import os
import shutil
import struct

SPV_MAGIC = 0x07230203
OP_ENTRY_POINT = 15
OP_EXTENSION = 10
OP_CAPABILITY = 17
OP_EMIT_MESH_TASKS_EXT = 5294
OP_SET_MESH_OUTPUTS_EXT = 5295
CAP_MESH_SHADING_EXT = 5283
EXECUTION_MODEL = {0: 'vs', 1: 'hs', 2: 'ds', 3: 'gs', 4: 'ps', 5: 'cs', 5364: 'ts', 5365: 'ms'}
# Linux NVN2 remaps task/mesh bound state from guest stage bits 6/7 into
# replay slots 3/4 (hle_nvn.c st_host).  Descriptor sets follow those host
# slots; unlike vk_android's eight-slot state, mesh must therefore use set 4.
STAGE_SET = {'vs': 0, 'ps': 1, 'gs': 2, 'hs': 3, 'ds': 4, 'cs': 0, 'ts': 3, 'ms': 4}
NATIVE_MESH_STAGES = {'ts', 'ms'}


class StageError(ValueError):
    pass


def module_stage(data):
    """Return the single entry point's short stage name, or fail closed."""
    if len(data) < 20 or len(data) % 4:
        raise StageError('not a SPIR-V module (length)')
    words = struct.unpack('<%dI' % (len(data) // 4), data)
    if words[0] != SPV_MAGIC:
        raise StageError('not a SPIR-V module (magic)')
    models, caps, exts, ops, i = [], set(), set(), set(), 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if not n or i + n > len(words):
            raise StageError('malformed SPIR-V instruction at word %d' % i)
        if op == OP_ENTRY_POINT:
            if n < 3:
                raise StageError('malformed OpEntryPoint')
            models.append(words[i + 1])
        elif op == OP_CAPABILITY and n >= 2:
            caps.add(words[i + 1])
        elif op == OP_EXTENSION:
            raw = struct.pack('<%dI' % (n - 1), *words[i + 1:i + n])
            exts.add(raw.split(b'\0', 1)[0].decode('ascii', 'replace'))
        ops.add(op)
        i += n
    if len(models) != 1:
        raise StageError('expected one SPIR-V entry point, found %d' % len(models))
    stage = EXECUTION_MODEL.get(models[0])
    if stage is None:
        raise StageError('unsupported SPIR-V execution model %d' % models[0])
    if stage in NATIVE_MESH_STAGES:
        required_op = OP_EMIT_MESH_TASKS_EXT if stage == 'ts' else OP_SET_MESH_OUTPUTS_EXT
        if CAP_MESH_SHADING_EXT not in caps or 'SPV_EXT_mesh_shader' not in exts or required_op not in ops:
            raise StageError('%s module lacks the SPV_EXT_mesh_shader capability, extension, or stage operation' % stage)
    return stage


def require_stage(path, expected):
    if expected not in STAGE_SET:
        raise StageError('unsupported requested stage %r' % expected)
    with open(path, 'rb') as f:
        actual = module_stage(f.read())
    if actual != expected:
        raise StageError('%s is %s, expected %s' % (path, actual, expected))
    return actual


def emit_loader_pair(key, stage, spv_path, out_dir):
    """Emit a descriptor-free synthetic ``.vk.spv/.vkb`` loader fixture."""
    if len(key) != 16 or any(c not in '0123456789abcdefABCDEF' for c in key):
        raise StageError('key must be 16 hexadecimal digits')
    require_stage(spv_path, stage)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, key.lower() + '.vk.spv')
    shutil.copyfile(spv_path, dst)
    with open(os.path.join(out_dir, key.lower() + '.vkb'), 'w') as f:
        f.write('# %s %s set %d\n' % (key.lower(), stage, STAGE_SET[stage]))
    return dst


def import_native_translation(key, current_job, source_dir, out_dir):
    """Import one externally translated native task/mesh job.

    Missing files return ``False`` so the caller counts the stage as omitted.
    Present-but-inconsistent files raise ``StageError`` instead of silently
    relabelling a compute shader as task or mesh.
    """
    expected = (current_job.get('mesh_stage') or {}).get('kind')
    if expected not in NATIVE_MESH_STAGES:
        raise StageError('%s is not a current task/mesh job' % key)
    paths = {ext: os.path.join(source_dir, key + ext) for ext in ('.spv', '.res.json', '.job.json')}
    if any(not os.path.isfile(p) for p in paths.values()):
        return False
    require_stage(paths['.spv'], expected)
    with open(paths['.res.json']) as f:
        res = json.load(f)
    with open(paths['.job.json']) as f:
        job = json.load(f)
    if res.get('stage') != expected or job.get('stage') != expected:
        raise StageError('%s metadata disagrees with expected stage %s' % (key, expected))
    # Preserve the control-section ownership fact that the external job format
    # does not carry.  The MIT postprocessor uses it to patch task and only
    # taskless mesh, never a task-fed mesh whose WorkgroupId is already
    # relative to its parent task.
    job['mesh_has_task'] = expected == 'ms' and bool((current_job.get('mesh_stage') or {}).get('task', True))
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(paths['.spv'], os.path.join(out_dir, key + '.spv'))
    shutil.copyfile(paths['.res.json'], os.path.join(out_dir, key + '.res.json'))
    with open(os.path.join(out_dir, key + '.job.json'), 'w') as f:
        json.dump(job, f, indent=2, sort_keys=True)
    reader = os.path.join(source_dir, key + '.reader.spv')
    if os.path.isfile(reader):
        shutil.copyfile(reader, os.path.join(out_dir, key + '.reader.spv'))
    return True
