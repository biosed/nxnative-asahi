#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lower large uniform-buffer handle tables to the persistent bindless set.

Ryujinx represents a dynamically indexed handle table as an array of combined
image samplers.  The array index ``k`` names the 32-bit NVN handle at
``cbuf[off + 8*k]``; materialising all 8,192 descriptors in every draw set is
both wasteful and, on Honeykrisp, several GiB per command-ring high water.

This pass preserves the index calculation, loads that handle from the already
bound constant buffer, adds the currently bound texture/sampler pool bases
from the tail of driver bank c2, and indexes the persistent texture/sampler
pools.  Descriptor indices and the values loaded through them are decorated
NonUniform.  Unsupported shapes or handle layouts fail closed.
"""
import struct


OP_CAPABILITY = 17
OP_EXTENSION = 10
OP_ENTRY_POINT = 15
OP_NAME = 5
OP_TYPE_BOOL = 20
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_IMAGE = 25
OP_TYPE_SAMPLER = 26
OP_TYPE_SAMPLED_IMAGE = 27
OP_TYPE_ARRAY = 28
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_LOAD = 61
OP_ACCESS_CHAIN = 65
OP_IN_BOUNDS_ACCESS_CHAIN = 66
OP_VECTOR_EXTRACT_DYNAMIC = 77
OP_COPY_OBJECT = 83
OP_SAMPLED_IMAGE = 86
OP_BITCAST = 124
OP_IADD = 128
OP_LOGICAL_AND = 167
OP_SELECT = 169
OP_I_NOT_EQUAL = 171
OP_U_LESS_THAN = 176
OP_SHIFT_RIGHT_LOGICAL = 194
OP_SHIFT_LEFT_LOGICAL = 196
OP_BITWISE_AND = 199
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_FUNCTION = 54

DEC_BLOCK = 2
DEC_ARRAY_STRIDE = 6
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
DEC_OFFSET = 35
DEC_NON_UNIFORM = 5300

CAP_SHADER_NON_UNIFORM = 5301
CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING = 5307

SC_UNIFORM_CONSTANT = 0
SC_UNIFORM = 2

POOL_TEXTURE_COUNT = 1 << 20       # spvr_bindless bounds this to the device table
POOL_SAMPLER_COUNT = 1 << 12
POOL_SAMPLER_BINDING = 660
POOL_BASE_BINDING = 17             # canonical driver-bank-c2 UBO

# Translator-original pool bindings.  spvr_bindless canonicalises them to the
# runtime set: 2D=0, 2D-array=1, depth-2D=2, sampler=3, 3D=4.
POOL_SHAPES = {
    (1, 0, 0, 0): 564,             # SpvDim2D
    (1, 0, 1, 0): 565,
    (1, 1, 0, 0): 566,
    (2, 0, 0, 0): 567,             # SpvDim3D
}


class HtabError(ValueError):
    pass


def _inst(op, *args):
    return [op, list(args)]


def _decode(data):
    if len(data) < 20 or len(data) % 4:
        raise HtabError('not a SPIR-V module (length)')
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    if words[0] != 0x07230203:
        raise HtabError('not a SPIR-V module (magic)')
    out, at = [], 5
    while at < len(words):
        n, op = words[at] >> 16, words[at] & 0xffff
        if not n or at + n > len(words):
            raise HtabError('malformed instruction at word %d' % at)
        out.append([op, words[at + 1:at + n]])
        at += n
    return words[:5], out


def _encode(header, ins):
    words = list(header)
    for op, args in ins:
        words.extend([((len(args) + 1) << 16) | op, *args])
    return struct.pack('<%dI' % len(words), *words)


def _string_words(text):
    raw = text.encode('utf-8') + b'\0'
    raw += b'\0' * (-len(raw) % 4)
    return list(struct.unpack('<%dI' % (len(raw) // 4), raw))


def _result_id(op, a):
    if op in (OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
              OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE,
              OP_TYPE_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER):
        return a[0] if a else None
    if op in (OP_CONSTANT, OP_VARIABLE, OP_LOAD, OP_ACCESS_CHAIN,
              OP_VECTOR_EXTRACT_DYNAMIC, OP_COPY_OBJECT, OP_SAMPLED_IMAGE,
              OP_BITCAST, OP_IADD, OP_LOGICAL_AND, OP_SELECT,
              OP_I_NOT_EQUAL, OP_U_LESS_THAN, OP_SHIFT_RIGHT_LOGICAL,
              OP_SHIFT_LEFT_LOGICAL, OP_BITWISE_AND):
        return a[1] if len(a) >= 2 else None
    return None


def _find_constant(ins, type_id, value):
    for op, a in ins:
        if op == OP_CONSTANT and len(a) >= 3 and a[0] == type_id and a[2] == value:
            return a[1]
    return None


def transform_tex_ubo_tables(data, descriptor_set, tables,
                             tex_base_offset, samp_base_offset):
    """Return ``(SPIR-V bytes, report)`` for large combined handle tables.

    Each table is ``{'binding', 'cbuf_binding', 'off', 'count'}``.  The bound
    variable supplies its image shape; only a combined texture+sampler handle
    in the same cbuf word is accepted.  ``off + 8*k`` is the exact NVN table
    rule shared with the Metal replay's htab_gpu path.
    """
    if not tables:
        return data, {'tables': 0, 'loads': 0, 'nonuniform': 0}
    if tex_base_offset < 0 or samp_base_offset < 0:
        raise HtabError('negative pool-base offset')
    header, ins = _decode(data)
    bound = header[3]
    if not bound:
        raise HtabError('module has no id bound')

    types, variables, constants, id_type = {}, {}, {}, {}
    binding, dset, decorations = {}, {}, {}
    capabilities, extensions = set(), set()
    for ix, (op, a) in enumerate(ins):
        rid = _result_id(op, a)
        if rid is not None:
            id_type[rid] = a[0] if op not in (
                OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
                OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE,
                OP_TYPE_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER) else rid
        if op in (OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
                  OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE,
                  OP_TYPE_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER):
            types[a[0]] = (op, a)
        elif op == OP_VARIABLE and len(a) >= 3:
            variables[a[1]] = (a[0], a[2])
        elif op == OP_CONSTANT and len(a) >= 3:
            constants[a[1]] = (a[0], a[2:])
        elif op == OP_DECORATE and len(a) >= 2:
            decorations.setdefault(a[0], []).append((ix, a[1:]))
            if len(a) == 3 and a[1] == DEC_BINDING:
                binding[a[0]] = a[2]
            elif len(a) == 3 and a[1] == DEC_DESCRIPTOR_SET:
                dset[a[0]] = a[2]
        elif op == OP_CAPABILITY and a:
            capabilities.add(a[0])
        elif op == OP_EXTENSION:
            raw = b''.join(struct.pack('<I', w) for w in a).split(b'\0', 1)[0]
            extensions.add(raw.decode('ascii', 'replace'))

    uint_types = [i for i, (op, a) in types.items()
                  if op == OP_TYPE_INT and a[1:] == [32, 0]]
    bool_types = [i for i, (op, _a) in types.items() if op == OP_TYPE_BOOL]
    if not uint_types:
        raise HtabError('module has no u32 type')
    uint_t = uint_types[0]

    next_id = bound
    def new_id():
        nonlocal next_id
        out = next_id
        next_id += 1
        return out

    bool_t = bool_types[0] if bool_types else new_id()
    const_ids = {}
    def const(value):
        if value not in const_ids:
            const_ids[value] = _find_constant(ins, uint_t, value) or new_id()
        return const_ids[value]

    c0, c1, c2, c3 = (const(v) for v in (0, 1, 2, 3))
    c20, cmask = const(20), const(0xfffff)
    ctex, csamp = const(POOL_TEXTURE_COUNT), const(POOL_SAMPLER_COUNT)

    by_location = {}
    for var in variables:
        if var in binding and var in dset:
            by_location.setdefault((dset[var], binding[var]), []).append(var)

    if by_location.get((descriptor_set, POOL_BASE_BINDING)):
        raise HtabError('driver-bank-c2 binding %d already exists' % POOL_BASE_BINDING)

    old_vars, pointer_access, table_info = set(), {}, {}
    cbuf_layout = {}
    for spec in tables:
        try:
            old_binding = int(spec['binding'])
            cbuf_binding = int(spec['cbuf_binding'])
            off = int(spec['off'])
            count = int(spec['count'])
        except (KeyError, TypeError, ValueError) as e:
            raise HtabError('invalid handle-table specification') from e
        if off < 0 or off % 4 or count < 33:
            raise HtabError('binding %d has invalid off/count %d/%d' %
                            (old_binding, off, count))
        # Intern the word offset before global constants are emitted below.
        # Keeping all declarations in the module's global section is required
        # even when a table starts after word zero.
        const(off // 4)
        ov = by_location.get((descriptor_set, old_binding), [])
        cv = by_location.get((descriptor_set, cbuf_binding), [])
        if len(ov) != 1 or len(cv) != 1:
            raise HtabError('binding %d or cbuf binding %d is not unique' %
                            (old_binding, cbuf_binding))
        old_var, cbuf_var = ov[0], cv[0]
        if old_var in old_vars:
            raise HtabError('duplicate handle-table binding %d' % old_binding)

        pty, sc = variables[old_var]
        if sc != SC_UNIFORM_CONSTANT or pty not in types or types[pty][0] != OP_TYPE_POINTER:
            raise HtabError('binding %d is not a UniformConstant pointer' % old_binding)
        arr_t = types[pty][1][2]
        if arr_t not in types or types[arr_t][0] != OP_TYPE_ARRAY:
            raise HtabError('binding %d is not a fixed descriptor array' % old_binding)
        sampled_t, len_id = types[arr_t][1][1:3]
        if sampled_t not in types or types[sampled_t][0] != OP_TYPE_SAMPLED_IMAGE:
            raise HtabError('binding %d is not a combined sampled-image array' % old_binding)
        if len_id not in constants or constants[len_id][1] != [count]:
            raise HtabError('binding %d array count disagrees with metadata' % old_binding)
        image_t = types[sampled_t][1][1]
        if image_t not in types or types[image_t][0] != OP_TYPE_IMAGE:
            raise HtabError('binding %d has no image type' % old_binding)
        ia = types[image_t][1]
        if len(ia) < 8 or ia[6] != 1:
            raise HtabError('binding %d image is not sampled' % old_binding)
        shape = (ia[2], ia[3], ia[4], ia[5])
        pool_binding = POOL_SHAPES.get(shape)
        if pool_binding is None:
            raise HtabError('binding %d image shape %r has no bindless pool' %
                            (old_binding, shape))

        if cbuf_var not in cbuf_layout:
            cp, csc = variables[cbuf_var]
            if csc != SC_UNIFORM or cp not in types or types[cp][0] != OP_TYPE_POINTER:
                raise HtabError('cbuf binding %d is not a Uniform block' % cbuf_binding)
            struct_t = types[cp][1][2]
            if struct_t not in types or types[struct_t][0] != OP_TYPE_STRUCT or not types[struct_t][1][1:]:
                raise HtabError('cbuf binding %d has no block struct' % cbuf_binding)
            arr = types[struct_t][1][1]
            if arr not in types or types[arr][0] != OP_TYPE_ARRAY:
                raise HtabError('cbuf binding %d member zero is not an array' % cbuf_binding)
            vec_t = types[arr][1][1]
            if vec_t not in types or types[vec_t][0] != OP_TYPE_VECTOR:
                raise HtabError('cbuf binding %d is not a vector array' % cbuf_binding)
            scalar_t, width = types[vec_t][1][1:3]
            scalar = types.get(scalar_t)
            if width != 4 or not scalar or scalar[0] not in (OP_TYPE_INT, OP_TYPE_FLOAT):
                raise HtabError('cbuf binding %d is not vec4/uvec4/ivec4' % cbuf_binding)
            if scalar[0] == OP_TYPE_FLOAT and scalar[1][1:] != [32]:
                raise HtabError('cbuf binding %d scalar is not 32-bit' % cbuf_binding)
            if scalar[0] == OP_TYPE_INT and scalar[1][1] != 32:
                raise HtabError('cbuf binding %d scalar is not 32-bit' % cbuf_binding)
            stride = [a[1] for _ix, a in decorations.get(arr, [])
                      if len(a) == 2 and a[0] == DEC_ARRAY_STRIDE]
            member_off = [a[3] for op, a in ins if op == OP_MEMBER_DECORATE and
                          len(a) == 4 and a[0] == struct_t and a[1] == 0 and a[2] == DEC_OFFSET]
            if stride != [16] or member_off != [0]:
                raise HtabError('cbuf binding %d does not have 16-byte rows at offset zero' % cbuf_binding)
            ptr_vec = next((i for i, (op, a) in types.items()
                            if op == OP_TYPE_POINTER and a[1:] == [SC_UNIFORM, vec_t]), None)
            if ptr_vec is None:
                ptr_vec = new_id()
            cbuf_layout[cbuf_var] = (vec_t, scalar_t, ptr_vec)

        table_info[old_var] = {
            'binding': old_binding, 'cbuf': cbuf_var, 'off': off,
            'count': count, 'sampled_t': sampled_t, 'image_t': image_t,
            'pool_binding': pool_binding,
        }
        old_vars.add(old_var)

    # Recognize every old-array access and its single combined-image load.
    drop_ids, replacement = set(), {}
    for ix, (op, a) in enumerate(ins):
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(a) == 4 and a[2] in old_vars:
            ptr_id, index = a[1], a[3]
            if ptr_id in pointer_access:
                raise HtabError('descriptor pointer id reused')
            pointer_access[ptr_id] = (ix, a[2], index)
            drop_ids.add(ptr_id)
    if not pointer_access:
        raise HtabError('large handle-table variables have no accesses')
    loads_by_ptr = {}
    for ix, (op, a) in enumerate(ins):
        if op == OP_LOAD and len(a) >= 3 and a[2] in pointer_access:
            loads_by_ptr.setdefault(a[2], []).append(ix)
    for ptr, (access_ix, old_var, index) in pointer_access.items():
        ls = loads_by_ptr.get(ptr, [])
        if len(ls) != 1:
            raise HtabError('descriptor pointer %d has %d loads' % (ptr, len(ls)))
        load_ix = ls[0]
        if ins[load_ix][1][0] != table_info[old_var]['sampled_t']:
            raise HtabError('descriptor load type disagrees with table')
        replacement[load_ix] = (old_var, index, ins[load_ix][1][1])
        replacement[access_ix] = None

    # Old descriptor variables and their access pointers must have no other
    # semantic use. Entry-point interface and annotations are handled below.
    for ix, (op, a) in enumerate(ins):
        if ix in replacement or op in (OP_ENTRY_POINT, OP_NAME, OP_DECORATE):
            continue
        if op == OP_VARIABLE and len(a) >= 2 and a[1] in old_vars:
            continue
        # Variable and pointer IDs are operands (never result/type IDs) in the
        # operations that may consume them here.  Do not scan every integer
        # operand: constants, line numbers and enumerants can legitimately
        # equal an ID and would make a numeric scan reject a valid module.
        consumed = ((op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN,
                            OP_LOAD, OP_COPY_OBJECT) and
                     len(a) >= 3 and a[2] in old_vars | drop_ids))
        if consumed:
            raise HtabError('unsupported handle-table id use at instruction %d' % ix)

    # Global declarations shared by every rewritten table.
    pool_base_struct, pool_base_ptr, pool_base_uint_ptr, pool_base_var = (
        new_id() for _ in range(4))
    sampler_t = next((i for i, (op, _a) in types.items() if op == OP_TYPE_SAMPLER), None) or new_id()
    sampler_arr, sampler_arr_ptr, sampler_elem_ptr, sampler_var = (
        new_id() for _ in range(4))
    image_pool = {}
    for info in table_info.values():
        key = (info['image_t'], info['pool_binding'])
        if key not in image_pool:
            image_pool[key] = tuple(new_id() for _ in range(4))  # array, array ptr, elem ptr, var
    for pool_binding in {POOL_SAMPLER_BINDING} | {k[1] for k in image_pool}:
        if by_location.get((descriptor_set, pool_binding)):
            raise HtabError('bindless source binding %d already exists' % pool_binding)

    annotations = [
        _inst(OP_DECORATE, pool_base_struct, DEC_BLOCK),
        _inst(OP_MEMBER_DECORATE, pool_base_struct, 0, DEC_OFFSET, tex_base_offset),
        _inst(OP_MEMBER_DECORATE, pool_base_struct, 1, DEC_OFFSET, samp_base_offset),
        _inst(OP_DECORATE, pool_base_var, DEC_DESCRIPTOR_SET, descriptor_set),
        _inst(OP_DECORATE, pool_base_var, DEC_BINDING, POOL_BASE_BINDING),
        _inst(OP_DECORATE, sampler_var, DEC_DESCRIPTOR_SET, descriptor_set),
        _inst(OP_DECORATE, sampler_var, DEC_BINDING, POOL_SAMPLER_BINDING),
    ]
    for (_image_t, pool_binding), (_arr, _aptr, _eptr, var) in image_pool.items():
        annotations += [
            _inst(OP_DECORATE, var, DEC_DESCRIPTOR_SET, descriptor_set),
            _inst(OP_DECORATE, var, DEC_BINDING, pool_binding),
        ]

    globals_ = []
    if not bool_types:
        globals_.append(_inst(OP_TYPE_BOOL, bool_t))
    existing_ptrs = set(types)
    for cbuf_var, (vec_t, _scalar_t, ptr_vec) in cbuf_layout.items():
        if ptr_vec not in existing_ptrs:
            globals_.append(_inst(OP_TYPE_POINTER, ptr_vec, SC_UNIFORM, vec_t))
    for value, cid in const_ids.items():
        if _find_constant(ins, uint_t, value) is None:
            globals_.append(_inst(OP_CONSTANT, uint_t, cid, value))
    globals_ += [
        _inst(OP_TYPE_STRUCT, pool_base_struct, uint_t, uint_t),
        _inst(OP_TYPE_POINTER, pool_base_ptr, SC_UNIFORM, pool_base_struct),
        _inst(OP_TYPE_POINTER, pool_base_uint_ptr, SC_UNIFORM, uint_t),
        _inst(OP_VARIABLE, pool_base_ptr, pool_base_var, SC_UNIFORM),
    ]
    if not any(op == OP_TYPE_SAMPLER for op, _a in types.values()):
        globals_.append(_inst(OP_TYPE_SAMPLER, sampler_t))
    globals_ += [
        _inst(OP_TYPE_ARRAY, sampler_arr, sampler_t, csamp),
        _inst(OP_TYPE_POINTER, sampler_arr_ptr, SC_UNIFORM_CONSTANT, sampler_arr),
        _inst(OP_TYPE_POINTER, sampler_elem_ptr, SC_UNIFORM_CONSTANT, sampler_t),
        _inst(OP_VARIABLE, sampler_arr_ptr, sampler_var, SC_UNIFORM_CONSTANT),
    ]
    for (image_t, _pool_binding), (arr, aptr, eptr, var) in image_pool.items():
        globals_ += [
            _inst(OP_TYPE_ARRAY, arr, image_t, ctex),
            _inst(OP_TYPE_POINTER, aptr, SC_UNIFORM_CONSTANT, arr),
            _inst(OP_TYPE_POINTER, eptr, SC_UNIFORM_CONSTANT, image_t),
            _inst(OP_VARIABLE, aptr, var, SC_UNIFORM_CONSTANT),
        ]

    nonuniform_ids, bodies = [], {}
    for load_ix, item in list(replacement.items()):
        if item is None:
            continue
        old_var, index, result = item
        info = table_info[old_var]
        idx_type = id_type.get(index)
        idx_u = index
        body = []
        if idx_type != uint_t:
            it = types.get(idx_type)
            if not it or it[0] != OP_TYPE_INT or it[1][1] != 32:
                raise HtabError('table index %d is not a 32-bit integer' % index)
            idx_u = new_id(); body.append(_inst(OP_BITCAST, uint_t, idx_u, index))
        word = new_id(); body.append(_inst(OP_SHIFT_LEFT_LOGICAL, uint_t, word, idx_u, c1))
        base_words = info['off'] // 4
        if base_words:
            added = new_id(); body.append(_inst(OP_IADD, uint_t, added, word, const(base_words))); word = added
        vec_index = new_id(); body.append(_inst(OP_SHIFT_RIGHT_LOGICAL, uint_t, vec_index, word, c2))
        component = new_id(); body.append(_inst(OP_BITWISE_AND, uint_t, component, word, c3))
        vec_t, scalar_t, ptr_vec = cbuf_layout[info['cbuf']]
        cptr = new_id(); body.append(_inst(OP_ACCESS_CHAIN, ptr_vec, cptr, info['cbuf'], c0, vec_index))
        cvec = new_id(); body.append(_inst(OP_LOAD, vec_t, cvec, cptr))
        scalar = new_id(); body.append(_inst(OP_VECTOR_EXTRACT_DYNAMIC, scalar_t, scalar, cvec, component))
        handle = new_id()
        if scalar_t == uint_t:
            body.append(_inst(OP_COPY_OBJECT, uint_t, handle, scalar))
        else:
            body.append(_inst(OP_BITCAST, uint_t, handle, scalar))

        tex_local = new_id(); body.append(_inst(OP_BITWISE_AND, uint_t, tex_local, handle, cmask))
        tex_nz = new_id(); body.append(_inst(OP_I_NOT_EQUAL, bool_t, tex_nz, tex_local, c0))
        tbptr = new_id(); body.append(_inst(OP_ACCESS_CHAIN, pool_base_uint_ptr, tbptr, pool_base_var, c0))
        tex_base = new_id(); body.append(_inst(OP_LOAD, uint_t, tex_base, tbptr))
        tex_global = new_id(); body.append(_inst(OP_IADD, uint_t, tex_global, tex_local, tex_base))
        tex_in = new_id(); body.append(_inst(OP_U_LESS_THAN, bool_t, tex_in, tex_global, ctex))
        tex_ok = new_id(); body.append(_inst(OP_LOGICAL_AND, bool_t, tex_ok, tex_nz, tex_in))
        tex_id = new_id(); body.append(_inst(OP_SELECT, uint_t, tex_id, tex_ok, tex_global, c0))

        samp_local = new_id(); body.append(_inst(OP_SHIFT_RIGHT_LOGICAL, uint_t, samp_local, handle, c20))
        samp_nz = new_id(); body.append(_inst(OP_I_NOT_EQUAL, bool_t, samp_nz, samp_local, c0))
        sbptr = new_id(); body.append(_inst(OP_ACCESS_CHAIN, pool_base_uint_ptr, sbptr, pool_base_var, c1))
        samp_base = new_id(); body.append(_inst(OP_LOAD, uint_t, samp_base, sbptr))
        samp_global = new_id(); body.append(_inst(OP_IADD, uint_t, samp_global, samp_local, samp_base))
        samp_in = new_id(); body.append(_inst(OP_U_LESS_THAN, bool_t, samp_in, samp_global, csamp))
        samp_ok = new_id(); body.append(_inst(OP_LOGICAL_AND, bool_t, samp_ok, samp_nz, samp_in))
        samp_id = new_id(); body.append(_inst(OP_SELECT, uint_t, samp_id, samp_ok, samp_global, c0))

        arr, _aptr, image_ptr_t, image_var = image_pool[(info['image_t'], info['pool_binding'])]
        del arr
        image_ptr = new_id(); body.append(_inst(OP_ACCESS_CHAIN, image_ptr_t, image_ptr, image_var, tex_id))
        image = new_id(); body.append(_inst(OP_LOAD, info['image_t'], image, image_ptr))
        samp_ptr = new_id(); body.append(_inst(OP_ACCESS_CHAIN, sampler_elem_ptr, samp_ptr, sampler_var, samp_id))
        sampler = new_id(); body.append(_inst(OP_LOAD, sampler_t, sampler, samp_ptr))
        body.append(_inst(OP_SAMPLED_IMAGE, info['sampled_t'], result, image, sampler))
        nonuniform_ids += [tex_id, image_ptr, image, samp_id, samp_ptr, sampler, result]
        bodies[load_ix] = body

    for nid in nonuniform_ids:
        annotations.append(_inst(OP_DECORATE, nid, DEC_NON_UNIFORM))

    type_at = next((i for i, (op, _a) in enumerate(ins) if 19 <= op <= 39), None)
    fn_at = next((i for i, (op, _a) in enumerate(ins) if op == OP_FUNCTION), None)
    if type_at is None or fn_at is None:
        raise HtabError('module has no type/function section')

    # SPIR-V 1.4 requires every used global in the entry-point interface;
    # earlier versions permit only Input/Output globals there.
    new_interfaces = ([pool_base_var, sampler_var] +
                      [x[3] for x in image_pool.values()]) if header[1] >= 0x00010400 else []
    out = []
    added_caps = []
    if CAP_SHADER_NON_UNIFORM not in capabilities:
        added_caps.append(_inst(OP_CAPABILITY, CAP_SHADER_NON_UNIFORM))
    if CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING not in capabilities:
        added_caps.append(_inst(OP_CAPABILITY, CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING))
    add_extension = 'SPV_EXT_descriptor_indexing' not in extensions
    inserted_header = False
    for ix, (op, a) in enumerate(ins):
        if not inserted_header and op != OP_CAPABILITY:
            out.extend(added_caps)
            if add_extension:
                out.append(_inst(OP_EXTENSION, *_string_words('SPV_EXT_descriptor_indexing')))
            inserted_header = True
        if ix == type_at:
            out.extend(annotations)
        if ix == fn_at:
            out.extend(globals_)
        if op == OP_ENTRY_POINT:
            # Literal name words end at the first NUL; every later word is an
            # interface id. Remove old descriptor variables and add the new
            # resource variables exactly once.
            name_end = 2
            while name_end < len(a):
                if b'\0' in struct.pack('<I', a[name_end]):
                    name_end += 1
                    break
                name_end += 1
            iface = [x for x in a[name_end:] if x not in old_vars]
            for var in new_interfaces:
                if var not in iface:
                    iface.append(var)
            out.append([op, a[:name_end] + iface])
            continue
        if op == OP_NAME and a and (a[0] in old_vars or a[0] in drop_ids):
            continue
        if op == OP_VARIABLE and len(a) >= 2 and a[1] in old_vars:
            continue
        if op == OP_DECORATE and a and (a[0] in old_vars or a[0] in drop_ids):
            continue
        if ix in replacement:
            if replacement[ix] is not None:
                out.extend(bodies[ix])
            continue
        out.append([op, list(a)])
    if not inserted_header:
        raise HtabError('module has no instruction after capabilities')
    header[3] = next_id
    encoded = _encode(header, out)

    # Structural post-audit: all old descriptor variables are gone; every
    # combined load became one sampled-image construction and each result is
    # explicitly NonUniform.
    _h, check = _decode(encoded)
    live_vars = {a[1] for op, a in check if op == OP_VARIABLE and len(a) >= 2}
    if live_vars & old_vars:
        raise HtabError('old handle-table descriptor survived rewrite')
    sampled = sum(op == OP_SAMPLED_IMAGE and len(a) >= 2 and a[1] in
                  {item[2] for item in replacement.values() if item is not None}
                  for op, a in check)
    if sampled != len(bodies):
        raise HtabError('postprocess sampled-image count mismatch')
    return encoded, {'tables': len(table_info), 'loads': len(bodies),
                     'nonuniform': len(nonuniform_ids),
                     'pool_base_binding': POOL_BASE_BINDING}


__all__ = ['HtabError', 'POOL_BASE_BINDING', 'POOL_SHAPES',
           'transform_tex_ubo_tables']
