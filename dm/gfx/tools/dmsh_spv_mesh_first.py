#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add the NVN mesh ``first`` base to native TaskEXT/taskless-MeshEXT SPIR-V.

The external GPL translator remains untouched.  This deterministic MIT-side
postprocessor recognizes the Vulkan WorkgroupId input, computes one effective
``x = WorkgroupId.x + mesh_first`` at the entry function, and redirects every
vector-load or constant-component-x load to that single value.  Task-fed mesh
modules are audited but remain byte-for-byte unchanged: their WorkgroupId is
task-relative and must not receive the draw base again.
"""
import argparse
import json
import os
import struct
import subprocess

from dmsh_spv_bindings import MESH_FIRST_BINDING, VK_BINDING_LIMIT
from dmsh_spv_stage import StageError, module_stage, STAGE_SET

OP_ENTRY_POINT = 15
OP_TYPE_INT = 21
OP_TYPE_VECTOR = 23
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_FUNCTION_CALL = 57
OP_VARIABLE = 59
OP_LOAD = 61
OP_STORE = 62
OP_ACCESS_CHAIN = 65
OP_IN_BOUNDS_ACCESS_CHAIN = 66
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_COMPOSITE_EXTRACT = 81
OP_COMPOSITE_INSERT = 82
OP_COPY_OBJECT = 83
OP_BITCAST = 124
OP_IADD = 128
OP_SELECT = 169
OP_PHI = 245
OP_LABEL = 248

DEC_BLOCK = 2
DEC_BUILTIN = 11
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
DEC_OFFSET = 35
BUILTIN_WORKGROUP_ID = 26
SC_INPUT = 1
SC_UNIFORM = 2


class MeshFirstError(StageError):
    pass


def _inst(op, *args):
    return [((len(args) + 1) << 16) | op, *args]


def _decode(data):
    if len(data) < 20 or len(data) % 4:
        raise MeshFirstError('not a SPIR-V module (length)')
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    if words[0] != 0x07230203:
        raise MeshFirstError('not a SPIR-V module (magic)')
    out, at = [], 5
    while at < len(words):
        n, op = words[at] >> 16, words[at] & 0xffff
        if not n or at + n > len(words):
            raise MeshFirstError('malformed instruction at word %d' % at)
        out.append([op, words[at + 1:at + n]])
        at += n
    return words[:5], out


def _encode(header, ins):
    words = list(header)
    for op, args in ins:
        words.extend(_inst(op, *args))
    return struct.pack('<%dI' % len(words), *words)


def _analyze(data, expected_stage=None):
    header, ins = _decode(data)
    stage = module_stage(data)
    if expected_stage is not None and stage != expected_stage:
        raise MeshFirstError('module stage %s, expected %s' % (stage, expected_stage))
    if stage not in ('ts', 'ms'):
        return header, ins, {'stage': stage, 'native': False, 'vector_loads': 0,
                             'component_loads': 0}

    ptr, vec, ints, const = {}, {}, {}, {}
    variables, wg = {}, []
    entry = []
    for op, a in ins:
        if op == OP_TYPE_POINTER and len(a) == 3:
            ptr[a[0]] = (a[1], a[2])
        elif op == OP_TYPE_VECTOR and len(a) == 3:
            vec[a[0]] = (a[1], a[2])
        elif op == OP_TYPE_INT and len(a) == 3:
            ints[a[0]] = (a[1], a[2])
        elif op == OP_CONSTANT and len(a) >= 3:
            const[a[1]] = (a[0], a[2:])
        elif op == OP_VARIABLE and len(a) >= 3:
            variables[a[1]] = (a[0], a[2])
        elif op == OP_DECORATE and len(a) == 3 and a[1] == DEC_BUILTIN and a[2] == BUILTIN_WORKGROUP_ID:
            wg.append(a[0])
        elif op == OP_ENTRY_POINT and len(a) >= 2:
            entry.append(a[1])
    if len(entry) != 1:
        raise MeshFirstError('expected one entry point')
    if len(wg) > 1 or (wg and wg[0] not in variables):
        raise MeshFirstError('expected at most one WorkgroupId input variable')
    if not wg:
        utys = sorted(i for i, shape in ints.items() if shape == (32, 0))
        if not utys:
            raise MeshFirstError('module has no u32 type for a synthesized WorkgroupId')
        uty = utys[0]
        vtys = sorted(i for i, shape in vec.items() if shape == (uty, 3))
        return header, ins, {'stage': stage, 'native': True, 'entry': entry[0],
            'wg': None, 'vector_type': vtys[0] if vtys else None,
            'uint_type': uty, 'zero_ids': [], 'vector_loads': [],
            'component_loads': [], 'load_function': entry[0],
            'synthesized_workgroup_id': True}
    wg = wg[0]
    pty, storage = variables[wg]
    if pty not in ptr or ptr[pty][0] != SC_INPUT:
        raise MeshFirstError('WorkgroupId is not an Input pointer')
    vty = ptr[pty][1]
    if vty not in vec or vec[vty][1] != 3:
        raise MeshFirstError('WorkgroupId is not a three-component vector')
    uty = vec[vty][0]
    if ints.get(uty) != (32, 0):
        raise MeshFirstError('WorkgroupId component is not u32')

    zero = [i for i, (t, v) in const.items() if t == uty and v == [0]]
    access, vector_loads, component_loads, load_functions = {}, [], [], set()
    current_fn = None
    for ix, (op, a) in enumerate(ins):
        if op == OP_FUNCTION:
            if len(a) < 2: raise MeshFirstError('malformed OpFunction')
            current_fn = a[1]
        elif op == OP_FUNCTION_END:
            current_fn = None
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(a) >= 4 and a[2] == wg:
            index = const.get(a[3]) if len(a) == 4 else None
            if not index or index[1] != [0] or ints.get(index[0], (0, 0))[0] != 32:
                raise MeshFirstError('WorkgroupId component access is not constant x')
            if a[0] not in ptr or ptr[a[0]] != (SC_INPUT, uty):
                raise MeshFirstError('WorkgroupId.x access has the wrong pointer type')
            access[a[1]] = (ix, current_fn)
        elif op == OP_LOAD and len(a) >= 3 and a[2] == wg:
            if a[0] != vty or current_fn is None:
                raise MeshFirstError('WorkgroupId vector load is outside a function or has the wrong type')
            vector_loads.append(ix); load_functions.add(current_fn)
        elif op == OP_LOAD and len(a) >= 3 and a[2] in access:
            if a[0] != uty or current_fn is None or access[a[2]][1] != current_fn:
                raise MeshFirstError('WorkgroupId.x load is outside its access function or has the wrong type')
            component_loads.append(ix); load_functions.add(current_fn)

    # The input variable and its x pointers may only feed the recognized load
    # forms.  This catches pointer copies, calls, dynamic indexing and a second
    # semantic path before any binary is changed.
    def pointer_operands(op, a):
        # Only positions whose SPIR-V grammar is an IdRef are considered.  A
        # raw numeric search is invalid because literal enum values can equal
        # an unrelated result id (seen in real modules' OpExtInst operands).
        if op == OP_LOAD and len(a) >= 3: return a[2:3]
        if op == OP_STORE and len(a) >= 2: return a[:2]
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(a) >= 3: return a[2:3]
        if op in (OP_COPY_OBJECT, OP_BITCAST) and len(a) >= 3: return a[2:3]
        if op == OP_FUNCTION_CALL and len(a) >= 3: return a[3:]
        if op == OP_SELECT and len(a) >= 5: return a[3:5]
        if op == OP_PHI and len(a) >= 4: return a[2::2]
        return ()

    in_function = False
    for ix, (op, a) in enumerate(ins):
        if op == OP_FUNCTION: in_function = True
        if not in_function: continue
        refs = pointer_operands(op, a)
        if wg in refs:
            allowed = ((op == OP_LOAD and len(a) >= 3 and a[2] == wg) or
                       (op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(a) >= 4 and a[2] == wg))
            if not allowed:
                raise MeshFirstError('unsupported WorkgroupId use at instruction %d' % ix)
        for p in access:
            if p in refs and not (op == OP_LOAD and len(a) >= 3 and a[2] == p):
                raise MeshFirstError('unsupported WorkgroupId.x pointer use at instruction %d' % ix)
        if op == OP_FUNCTION_END: in_function = False
    if len(load_functions) > 1:
        raise MeshFirstError('WorkgroupId loads span %d functions' % len(load_functions))
    return header, ins, {'stage': stage, 'native': True, 'entry': entry[0], 'wg': wg,
        'vector_type': vty, 'uint_type': uty, 'zero_ids': zero,
        'vector_loads': vector_loads, 'component_loads': component_loads,
        'load_function': next(iter(load_functions), entry[0]),
        'synthesized_workgroup_id': False}


def transform_mesh_first(data, expected_stage=None, mesh_has_task=False, descriptor_set=None):
    """Return ``(bytes, report)``; task-fed mesh is audited but unchanged."""
    header, ins, r = _analyze(data, expected_stage)
    if not r['native']:
        r.update(owner=False, semantic_adds=0, binding=None)
        return data, r
    owner = r['stage'] == 'ts' or (r['stage'] == 'ms' and not mesh_has_task)
    r.update(owner=owner, owner_no_use=False, semantic_adds=0, binding=None)
    if not owner:
        r['vector_loads'] = len(r['vector_loads'])
        r['component_loads'] = len(r['component_loads'])
        return data, r
    if r['wg'] is None or not (r['vector_loads'] or r['component_loads']):
        r.update(owner_no_use=True, vector_loads=len(r['vector_loads']),
                 component_loads=len(r['component_loads']))
        return data, r
    if descriptor_set is None:
        descriptor_set = STAGE_SET[r['stage']]
    if descriptor_set != STAGE_SET[r['stage']]:
        raise MeshFirstError('descriptor set %d disagrees with %s set %d' %
                             (descriptor_set, r['stage'], STAGE_SET[r['stage']]))
    if not 0 <= MESH_FIRST_BINDING < VK_BINDING_LIMIT:
        raise MeshFirstError('mesh_first binding outside the Linux limit')
    for op, a in ins:
        if op == OP_DECORATE and len(a) == 3 and a[1] == DEC_BINDING and a[2] == MESH_FIRST_BINDING:
            raise MeshFirstError('binding %d already exists' % MESH_FIRST_BINDING)

    next_id = header[3]
    def new_id():
        nonlocal next_id
        out = next_id; next_id += 1; return out
    struct_id, ptr_struct, ptr_uint, first_var = new_id(), new_id(), new_id(), new_id()
    zero = r['zero_ids'][0] if r['zero_ids'] else new_id()
    raw_vec, raw_x, first_ptr, first_value, effective_x, effective_vec = (new_id() for _ in range(6))

    decorations = [
        [OP_DECORATE, [struct_id, DEC_BLOCK]],
        [OP_MEMBER_DECORATE, [struct_id, 0, DEC_OFFSET, 0]],
        [OP_DECORATE, [first_var, DEC_DESCRIPTOR_SET, descriptor_set]],
        [OP_DECORATE, [first_var, DEC_BINDING, MESH_FIRST_BINDING]],
    ]
    globals_ = [
        [OP_TYPE_STRUCT, [struct_id, r['uint_type']]],
        [OP_TYPE_POINTER, [ptr_struct, SC_UNIFORM, struct_id]],
        [OP_TYPE_POINTER, [ptr_uint, SC_UNIFORM, r['uint_type']]],
    ]
    if not r['zero_ids']:
        globals_.append([OP_CONSTANT, [r['uint_type'], zero, 0]])
    globals_.append([OP_VARIABLE, [ptr_struct, first_var, SC_UNIFORM]])
    body = [
        [OP_LOAD, [r['vector_type'], raw_vec, r['wg']]],
        [OP_COMPOSITE_EXTRACT, [r['uint_type'], raw_x, raw_vec, 0]],
        [OP_ACCESS_CHAIN, [ptr_uint, first_ptr, first_var, zero]],
        [OP_LOAD, [r['uint_type'], first_value, first_ptr]],
        [OP_IADD, [r['uint_type'], effective_x, raw_x, first_value]],
        [OP_COMPOSITE_INSERT, [r['vector_type'], effective_vec, effective_x, raw_vec, 0]],
    ]

    type_at = next((i for i, (op, _) in enumerate(ins) if 19 <= op <= 39), None)
    fn_at = next((i for i, (op, _) in enumerate(ins) if op == OP_FUNCTION), None)
    if type_at is None or fn_at is None:
        raise MeshFirstError('module has no type/function section')
    label_at = None
    in_entry = False
    for i, (op, a) in enumerate(ins):
        if op == OP_FUNCTION: in_entry = len(a) >= 2 and a[1] == r['load_function']
        elif in_entry and op == OP_LABEL: label_at = i; break
    if label_at is None:
        raise MeshFirstError('entry function has no label')
    body_at = label_at + 1
    while body_at < len(ins) and ins[body_at][0] == OP_VARIABLE:
        body_at += 1

    rewrite = {i: [OP_COPY_OBJECT, [r['vector_type'], ins[i][1][1], effective_vec]] for i in r['vector_loads']}
    rewrite.update({i: [OP_COPY_OBJECT, [r['uint_type'], ins[i][1][1], effective_x]] for i in r['component_loads']})
    out = []
    for i, inst in enumerate(ins):
        if i == type_at: out.extend(decorations)
        if i == fn_at: out.extend(globals_)
        if i == body_at: out.extend(body)
        emitted = rewrite.get(i, inst)
        if emitted[0] == OP_ENTRY_POINT and emitted[1][1] == r['entry']:
            emitted = [OP_ENTRY_POINT, emitted[1] + [first_var]]
        out.append(emitted)
    header[3] = next_id
    encoded = _encode(header, out)
    # Re-audit the untouched built-in path.  Exactly one injected IAdd exists;
    # all original x loads are now copies of its result.
    _, check = _decode(encoded)
    adds = sum(op == OP_IADD and len(a) == 4 and a[1] == effective_x and
               a[2] == raw_x and a[3] == first_value for op, a in check)
    if adds != 1:
        raise MeshFirstError('postprocess did not produce exactly one semantic add')
    r.update(semantic_adds=1, binding=MESH_FIRST_BINDING,
             descriptor_set=descriptor_set, rewritten_loads=len(rewrite),
             vector_loads=len(r['vector_loads']),
             component_loads=len(r['component_loads']))
    return encoded, r


def _owners(package_dir, map_path):
    """Derive task-fed mesh ownership from MIT control-section facts."""
    from dmsh_spv_native import _locations, _package_records, split_record, nvn2_key
    import dmsh_meshstage
    loc, packages, out = _locations(map_path), {}, {}
    for key, (name, index, stage) in loc.items():
        if stage not in ('ts', 'ms'): continue
        if name not in packages:
            packages[name] = _package_records(os.path.join(package_dir, name))
        if index >= len(packages[name]):
            raise MeshFirstError('%s: package index outside %s' % (key, name))
        control, area = split_record(packages[name][index])
        if nvn2_key(control, area) != key:
            raise MeshFirstError('%s: package key mismatch' % key)
        info = dmsh_meshstage.info(control, area)
        if info.get('kind') != stage:
            raise MeshFirstError('%s: package stage mismatch' % key)
        out[key] = {'stage': stage, 'mesh_has_task': bool(info.get('task', False))}
    return out


def patch_tree(source, out_dir, package_dir, map_path, spirv_val):
    owners = _owners(package_dir, map_path)
    os.makedirs(out_dir, exist_ok=True)
    totals = {'ts': 0, 'ms': 0, 'owner': 0, 'owner_no_use': 0,
              'patched_owner': 0, 'paired_mesh': 0, 'vector_loads': 0,
              'component_loads': 0, 'rewritten_loads': 0, 'semantic_adds': 0}
    keys = sorted(p[:-7] for p in os.listdir(source) if p.endswith('.vk.spv'))
    for key in keys:
        if key not in owners:
            raise MeshFirstError('%s: no package ownership fact' % key)
        fact = owners[key]; stage = fact['stage']
        with open(os.path.join(source, key + '.vk.spv'), 'rb') as f: data = f.read()
        try:
            patched, report = transform_mesh_first(data, stage, fact['mesh_has_task'], STAGE_SET[stage])
        except MeshFirstError as e:
            raise MeshFirstError('%s: %s' % (key, e)) from e
        with open(os.path.join(source, key + '.vkb')) as f: vkb = f.read()
        header = '# %s %s set %d\n' % (key, stage, STAGE_SET[stage])
        if not vkb.startswith(header) or '\tmesh_first\t' in vkb:
            raise MeshFirstError('%s: invalid or already-patched .vkb' % key)
        if report['binding'] is not None:
            row = '\t'.join(str(x) for x in (MESH_FIRST_BINDING, 'u', 'mesh_first', '', '', '', '', '', 0, 0, 0, 0, 1)) + '\n'
            vkb += row
        with open(os.path.join(source, key + '.job.json')) as f: job = json.load(f)
        if job.get('stage') != stage:
            raise MeshFirstError('%s: job stage mismatch' % key)
        job['mesh_has_task'] = fact['mesh_has_task']
        job['mesh_first'] = report['binding'] is not None
        job['mesh_first_owner'] = bool(report['owner'])
        job['mesh_first_owner_no_use'] = bool(report.get('owner_no_use', False))
        for suffix, content, binary in (('.vk.spv', patched, True), ('.vkb', vkb, False),
                                        ('.job.json', json.dumps(job, indent=2, sort_keys=True) + '\n', False)):
            path = os.path.join(out_dir, key + suffix)
            with open(path, 'wb' if binary else 'w') as f: f.write(content)
        r = subprocess.run([spirv_val, '--target-env', 'vulkan1.3', os.path.join(out_dir, key + '.vk.spv')],
                           capture_output=True, text=True)
        if r.returncode:
            raise MeshFirstError('%s: spirv-val: %s' % (key, (r.stderr or r.stdout).strip()[:300]))
        totals[stage] += 1; totals['owner'] += report['owner']
        totals['owner_no_use'] += report.get('owner_no_use', False)
        totals['patched_owner'] += report['binding'] is not None
        totals['paired_mesh'] += stage == 'ms' and fact['mesh_has_task']
        for name in ('vector_loads', 'component_loads', 'rewritten_loads', 'semantic_adds'):
            totals[name] += report.get(name, 0)
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', action='store_true', help='patch a validated canonical native-mesh directory')
    p.add_argument('--source'); p.add_argument('--output'); p.add_argument('--package-dir'); p.add_argument('--map')
    p.add_argument('--spirv-val', default=os.environ.get('SPIRV_VAL', 'spirv-val'))
    a = p.parse_args()
    if not a.tree or not all((a.source, a.output, a.package_dir, a.map)):
        p.error('--tree requires --source, --output, --package-dir and --map')
    totals = patch_tree(a.source, a.output, a.package_dir, a.map, a.spirv_val)
    print('mesh_first: ' + ' '.join('%s=%s' % item for item in totals.items()))


if __name__ == '__main__':
    main()
