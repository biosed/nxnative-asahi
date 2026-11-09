#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Coalesce provably contiguous read-only SPIR-V storage-buffer loads.

The NVIDIA translator represents a storage buffer as a runtime array of u32
and commonly emits two or four scalar loads for one aligned source word group.
This pass adds a NonWritable alias of that descriptor whose array element is
uvec2/uvec4, loads it once, then defines every original SSA result at its old
program point with OpCompositeExtract.

The proof is deliberately narrow: reflection must mark the sbuf read-only;
the variable must be a 4-byte scalar runtime array and have no write; loads
must have no memory operands; candidates must be consecutive external-memory
operations in one basic block; and affine indices must cover one aligned 2- or
4-word interval. No generic optimizer is run and descriptor ABI is unchanged.
"""
import argparse
import json
import struct

SPV_MAGIC = 0x07230203
OP_NAME = 5
OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR = 21, 22, 23
OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER = 29, 30, 32
OP_CONSTANT = 43
OP_FUNCTION, OP_FUNCTION_END, OP_FUNCTION_CALL = 54, 56, 57
OP_VARIABLE, OP_LOAD, OP_STORE = 59, 61, 62
OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED = 63, 64
OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN = 65, 66
OP_PTR_ACCESS_CHAIN, OP_ARRAY_LENGTH, OP_IN_BOUNDS_PTR_ACCESS_CHAIN = 67, 68, 70
OP_DECORATE, OP_MEMBER_DECORATE = 71, 72
OP_COMPOSITE_EXTRACT, OP_COPY_OBJECT = 81, 83
OP_CONVERT_PTR_TO_U, OP_CONVERT_U_TO_PTR = 117, 120
OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR = 121, 122
OP_GENERIC_CAST_TO_PTR_EXPLICIT = 123
OP_BITCAST = 124
OP_IADD, OP_ISUB, OP_IMUL = 128, 130, 132
OP_SELECT = 169
OP_SHIFT_RIGHT_LOGICAL, OP_SHIFT_LEFT_LOGICAL = 194, 196
OP_CONTROL_BARRIER, OP_MEMORY_BARRIER = 224, 225
OP_ATOMIC_LOAD, OP_ATOMIC_STORE, OP_ATOMIC_LAST = 227, 228, 242
OP_GROUP_ASYNC_COPY, OP_GROUP_WAIT_EVENTS = 259, 260
OP_IMAGE_SPARSE_FIRST, OP_IMAGE_SPARSE_LAST = 305, 320
OP_ATOMIC_FLAG_TEST_AND_SET, OP_ATOMIC_FLAG_CLEAR = 318, 319
OP_MEMORY_NAMED_BARRIER = 329
OP_COPY_LOGICAL = 400
OP_PTR_EQUAL, OP_PTR_NOT_EQUAL, OP_PTR_DIFF = 401, 402, 403
OP_COOPERATIVE_MATRIX_LOAD_NV, OP_COOPERATIVE_MATRIX_STORE_NV = 5359, 5360
OP_ATOMIC_FADD_EXT = 6035
OP_PHI = 245
OP_LABEL = 248

DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_ARRAY_STRIDE = 2, 3, 6
DEC_RESTRICT, DEC_ALIASED, DEC_VOLATILE = 19, 20, 21
DEC_COHERENT, DEC_NON_WRITABLE = 23, 24
DEC_BINDING, DEC_DESCRIPTOR_SET, DEC_OFFSET = 33, 34, 35
SC_UNIFORM, SC_PRIVATE, SC_FUNCTION, SC_STORAGE_BUFFER = 2, 6, 7, 12


class VectorLoadError(ValueError):
    pass


def _inst(op, *args):
    return [((len(args) + 1) << 16) | op, *args]


def _decode(data):
    if len(data) < 20 or len(data) % 4:
        raise VectorLoadError('not a SPIR-V module (length)')
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    if words[0] != SPV_MAGIC:
        raise VectorLoadError('not a SPIR-V module (magic)')
    out, at = [], 5
    while at < len(words):
        n, op = words[at] >> 16, words[at] & 0xffff
        if not n or at + n > len(words):
            raise VectorLoadError('malformed instruction at word %d' % at)
        out.append([op, words[at + 1:at + n]])
        at += n
    return words[:5], out


def _encode(header, ins):
    words = list(header)
    for op, args in ins:
        words.extend(_inst(op, *args))
    return struct.pack('<%dI' % len(words), *words)


def _string(words):
    raw = struct.pack('<%dI' % len(words), *words)
    return raw.split(b'\0', 1)[0].decode('utf-8', 'replace')


def _string_words(text):
    raw = text.encode() + b'\0'; raw += b'\0' * (-len(raw) % 4)
    return list(struct.unpack('<%dI' % (len(raw) // 4), raw))


def _readonly_names(reflection):
    bindings = reflection.get('bindings', reflection) if isinstance(reflection, dict) else reflection
    return {b.get('name') for b in bindings
            if b.get('kind') == 'sbuf' and b.get('rw') is False and b.get('name')}


# Affine values are (sorted terms, unsigned constant, bit width). Leaf keys are
# tuples, so a function-local value unknown at a label remains stable across
# repeated loads of that same variable.
def _leaf(identity, bits=32):
    return (((identity, 1),), 0, bits)


def _constant(value, bits=32):
    return ((), value & ((1 << bits) - 1), bits)


def _add(a, b, subtract=False):
    if a is None or b is None or a[2] != b[2]: return None
    bits, mask = a[2], (1 << a[2]) - 1
    terms, sign = dict(a[0]), -1 if subtract else 1
    for key, value in b[0]:
        terms[key] = (terms.get(key, 0) + sign * value) & mask
        if not terms[key]: del terms[key]
    return (tuple(sorted(terms.items(), key=lambda x: repr(x[0]))),
            (a[1] + sign * b[1]) & mask, bits)


def _mul(a, b):
    if a is None or b is None or a[2] != b[2] or (a[0] and b[0]): return None
    variable, constant = (a, b) if a[0] else (b, a)
    bits, mask, scale = a[2], (1 << a[2]) - 1, constant[1]
    return (tuple((key, value * scale & mask) for key, value in variable[0]),
            variable[1] * scale & mask, bits)


def _literal(words, width):
    if width <= 32 and words: return words[0] & ((1 << width) - 1)
    if width == 64 and len(words) >= 2: return words[0] | words[1] << 32
    return None


def _result(op, a):
    if op in (OP_LOAD, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN,
              OP_PTR_ACCESS_CHAIN, OP_IN_BOUNDS_PTR_ACCESS_CHAIN,
              OP_COPY_OBJECT, OP_COPY_LOGICAL, OP_BITCAST,
              OP_CONVERT_PTR_TO_U, OP_CONVERT_U_TO_PTR,
              OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR,
              OP_GENERIC_CAST_TO_PTR_EXPLICIT, OP_SELECT, OP_PHI,
              OP_IADD, OP_ISUB, OP_IMUL,
              OP_SHIFT_RIGHT_LOGICAL, OP_SHIFT_LEFT_LOGICAL,
              OP_COMPOSITE_EXTRACT) and len(a) >= 2:
        return a[0], a[1]
    if op == OP_VARIABLE and len(a) >= 2: return a[0], a[1]
    return None


_POINTER_COPIES = {
    OP_COPY_OBJECT, OP_COPY_LOGICAL, OP_BITCAST,
    OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR,
    OP_GENERIC_CAST_TO_PTR_EXPLICIT,
}
_POINTER_CHAINS = {
    OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN,
    OP_PTR_ACCESS_CHAIN, OP_IN_BOUNDS_PTR_ACCESS_CHAIN,
}
_ATOMIC_OPS = set(range(OP_ATOMIC_LOAD, OP_ATOMIC_LAST + 1)) | {
    OP_ATOMIC_FLAG_TEST_AND_SET, OP_ATOMIC_FLAG_CLEAR, OP_ATOMIC_FADD_EXT,
}


def _merge_roots(root, result, sources):
    merged = set().union(*(root.get(source, ()) for source in sources))
    if not merged or merged <= root.get(result, set()): return False
    root.setdefault(result, set()).update(merged)
    return True


def _written_roots(ins, variables):
    """Roots written through, escaped, or used by an unproved pointer operation."""
    root, written = {v: {v} for v in variables}, set()
    # Pointer Phi operands can come from later blocks. Reach a fixed point before
    # auditing consumers so a backedge alias cannot hide a write.
    changed = True
    while changed:
        changed = False
        for op, a in ins:
            if op in _POINTER_CHAINS and len(a) >= 3:
                changed |= _merge_roots(root, a[1], (a[2],))
            elif op in _POINTER_COPIES and len(a) >= 3:
                changed |= _merge_roots(root, a[1], (a[2],))
            elif op == OP_SELECT and len(a) >= 5:
                changed |= _merge_roots(root, a[1], a[3:5])
            elif op == OP_PHI and len(a) >= 4:
                changed |= _merge_roots(root, a[1], a[2::2])
    for op, a in ins:
        if op == OP_STORE and len(a) >= 2:
            written.update(root.get(a[0], ()))
            # Storing a rooted pointer value makes its later consumers opaque.
            written.update(root.get(a[1], ()))
        elif op in (OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED) and a:
            written.update(root.get(a[0], ()))
        elif op == OP_GROUP_ASYNC_COPY and len(a) >= 4:
            written.update(root.get(a[3], ()))
        elif op == OP_COOPERATIVE_MATRIX_STORE_NV and a:
            written.update(root.get(a[0], ()))
        elif op in _ATOMIC_OPS:
            pointer = a[0] if op in (OP_ATOMIC_STORE, OP_ATOMIC_FLAG_CLEAR) else (a[2] if len(a) >= 3 else None)
            written.update(root.get(pointer, ()))
        elif op == OP_FUNCTION_CALL and len(a) >= 3:
            for value in a[3:]:
                written.update(root.get(value, ()))
        elif op in (OP_CONVERT_PTR_TO_U, OP_CONVERT_U_TO_PTR) and len(a) >= 3:
            written.update(root.get(a[2], ()))
        elif op == 254 and a:   # OpReturnValue: a rooted pointer escapes the proof.
            written.update(root.get(a[0], ()))
    # Reject any rooted pointer operand used by an operation whose semantics
    # this proof does not model. Infer and skip conventional result type/id
    # operands so a definition is not mistaken for a use.
    safe = (_POINTER_CHAINS | _POINTER_COPIES | _ATOMIC_OPS |
            {OP_LOAD, OP_STORE, OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED,
             OP_FUNCTION_CALL, OP_ARRAY_LENGTH, OP_SELECT, OP_PHI,
             OP_CONVERT_PTR_TO_U, OP_CONVERT_U_TO_PTR,
             OP_PTR_EQUAL, OP_PTR_NOT_EQUAL, OP_PTR_DIFF,
             OP_GROUP_ASYNC_COPY, OP_COOPERATIVE_MATRIX_LOAD_NV,
             OP_COOPERATIVE_MATRIX_STORE_NV, 254})
    type_ids = {a[0] for op, a in ins if 19 <= op <= 39 and a}
    in_function = False
    for op, a in ins:
        if op == OP_FUNCTION: in_function = True
        if in_function and op not in safe:
            operands = a[2:] if len(a) >= 2 and a[0] in type_ids else a
            for value in operands:
                written.update(root.get(value, ()))
        if op == OP_FUNCTION_END: in_function = False
    return written


def _memory_fence(op):
    """Whether moving a buffer load across ``op`` is not proved safe."""
    return (op in (OP_FUNCTION_CALL, OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED,
                   OP_CONTROL_BARRIER, OP_MEMORY_BARRIER,
                   OP_GROUP_ASYNC_COPY, OP_GROUP_WAIT_EVENTS,
                   OP_MEMORY_NAMED_BARRIER,
                   OP_COOPERATIVE_MATRIX_LOAD_NV, OP_COOPERATIVE_MATRIX_STORE_NV)
            or 87 <= op <= 100
            or OP_IMAGE_SPARSE_FIRST <= op <= OP_IMAGE_SPARSE_LAST
            or op in _ATOMIC_OPS)


# Only these fully understood, side-effect-free/local operations may occur
# between scalar buffer loads that become one earlier vector load. Everything
# else closes the run, including future SPIR-V memory/tensor extensions.
_SEGMENT_SAFE_OPS = {
    0, 8, 317,   # OpNop, OpLine, OpNoLine
    OP_VARIABLE,
    OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN,
    OP_COPY_OBJECT, OP_BITCAST,
    OP_IADD, OP_ISUB, OP_IMUL,
    OP_SHIFT_RIGHT_LOGICAL, OP_SHIFT_LEFT_LOGICAL,
}


def _segment_safe(op, operands=None):
    if op == OP_LOAD: return operands is not None and len(operands) == 3
    if op == OP_STORE: return operands is not None and len(operands) == 2
    return op in _SEGMENT_SAFE_OPS


def _entrypoint_aliases(version, aliases):
    """Global descriptor aliases required in OpEntryPoint for this version."""
    return list(aliases) if version >= 0x00010400 else []


_UNPROVEN_POINTER_DERIVATIONS = {
    OP_COPY_LOGICAL, OP_CONVERT_PTR_TO_U, OP_CONVERT_U_TO_PTR,
    OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR,
    OP_GENERIC_CAST_TO_PTR_EXPLICIT, OP_SELECT, OP_PHI,
}


def materialize_nonwritable(data, reflection):
    """Materialize translator ``rw:false`` facts as NonWritable decorations.

    This is separate so ``transform_vector_loads`` itself can require the
    source SPIR-V descriptor to carry the memory-semantics fact. A reflection
    entry contradicted by a store, atomic or pointer-passing call is rejected.
    """
    header, ins = _decode(data)
    requested = _readonly_names(reflection)
    names, variables, decorated = {}, {}, set()
    for op, a in ins:
        if op == OP_NAME and len(a) >= 2: names[a[0]] = _string(a[1:])
        elif op == OP_VARIABLE and len(a) >= 3: variables[a[1]] = (a[0], a[2])
        elif op == OP_DECORATE and len(a) >= 2 and a[1] == DEC_NON_WRITABLE: decorated.add(a[0])
    written = _written_roots(ins, variables)
    add = [v for v in variables if names.get(v) in requested and v not in decorated and v not in written]
    conflicts = sum(names.get(v) in requested for v in written)
    if not add:
        return data, {'materialized_nonwritable': 0,
                      'reflection_write_conflicts': conflicts,
                      '_materialized_variables': []}
    annotation_at = next((i for i, (op, _) in enumerate(ins)
                          if op in (OP_DECORATE, OP_MEMBER_DECORATE)), None)
    if annotation_at is None:
        annotation_at = next((i for i, (op, _) in enumerate(ins) if 19 <= op <= 39), None)
    if annotation_at is None: raise VectorLoadError('module has no annotation/type section')
    out = ins[:annotation_at] + [[OP_DECORATE, [v, DEC_NON_WRITABLE]] for v in add] + ins[annotation_at:]
    return _encode(header, out), {'materialized_nonwritable': len(add),
                                  'reflection_write_conflicts': conflicts,
                                  '_materialized_variables': add}


def transform_vector_loads(data, reflection):
    """Return ``(patched_bytes, report)`` for one canonical Vulkan module."""
    header, ins = _decode(data)
    names, decorations, member_decorations = {}, {}, {}
    ints, floats, vectors, arrays, structs, pointers = {}, {}, {}, {}, {}, {}
    variables, constants, type_of = {}, {}, {}
    for op, a in ins:
        if op == OP_NAME and len(a) >= 2: names[a[0]] = _string(a[1:])
        elif op == OP_DECORATE and len(a) >= 2: decorations.setdefault(a[0], []).append(tuple(a[1:]))
        elif op == OP_MEMBER_DECORATE and len(a) >= 3: member_decorations.setdefault((a[0], a[1]), []).append(tuple(a[2:]))
        elif op == OP_TYPE_INT and len(a) == 3: ints[a[0]] = (a[1], a[2])
        elif op == OP_TYPE_FLOAT and len(a) == 2: floats[a[0]] = a[1]
        elif op == OP_TYPE_VECTOR and len(a) == 3: vectors[(a[1], a[2])] = a[0]
        elif op == OP_TYPE_RUNTIME_ARRAY and len(a) == 2: arrays[a[0]] = a[1]
        elif op == OP_TYPE_STRUCT and len(a) >= 2: structs[a[0]] = tuple(a[1:])
        elif op == OP_TYPE_POINTER and len(a) == 3: pointers[a[0]] = (a[1], a[2])
        elif op == OP_VARIABLE and len(a) >= 3:
            variables[a[1]] = (a[0], a[2]); type_of[a[1]] = a[0]
        elif op == OP_CONSTANT and len(a) >= 3:
            constants[a[1]] = (a[0], tuple(a[2:])); type_of[a[1]] = a[0]
        else:
            r = _result(op, a)
            if r: type_of[r[1]] = r[0]

    requested, written = _readonly_names(reflection), _written_roots(ins, variables)
    locations = {}
    for var in variables:
        ds = decorations.get(var, ())
        binding = next((d[1] for d in ds if len(d) == 2 and d[0] == DEC_BINDING), None)
        dset = next((d[1] for d in ds if len(d) == 2 and d[0] == DEC_DESCRIPTOR_SET), None)
        if binding is not None and dset is not None:
            locations.setdefault((dset, binding), []).append(var)
    eligible = {}; rejected = {'shape': 0, 'written': 0, 'volatile': 0,
                                'aliasing': 0, 'nonwritable': 0}
    for var, (ptr_type, storage) in variables.items():
        if names.get(var) not in requested: continue
        if var in written: rejected['written'] += 1; continue
        vdecs = {d[0] for d in decorations.get(var, [])}
        if DEC_NON_WRITABLE not in vdecs: rejected['nonwritable'] += 1; continue
        if ptr_type not in pointers or pointers[ptr_type][0] != storage or storage not in (SC_UNIFORM, SC_STORAGE_BUFFER):
            rejected['shape'] += 1; continue
        struct_id = pointers[ptr_type][1]
        if len(structs.get(struct_id, ())) != 1: rejected['shape'] += 1; continue
        mdecs = {d[0] for d in member_decorations.get((struct_id, 0), [])}
        if (vdecs | mdecs) & {DEC_VOLATILE, DEC_COHERENT}:
            rejected['volatile'] += 1; continue
        if DEC_RESTRICT in vdecs:
            rejected['aliasing'] += 1; continue
        array_id = structs[struct_id][0]; scalar = arrays.get(array_id)
        scalar_bits = ints.get(scalar, (floats.get(scalar), 0))[0]
        stride = next((d[1] for d in decorations.get(array_id, []) if len(d) == 2 and d[0] == DEC_ARRAY_STRIDE), None)
        block = next((d[0] for d in decorations.get(struct_id, []) if d and d[0] in (DEC_BLOCK, DEC_BUFFER_BLOCK)), None)
        offset = next((d[1] for d in member_decorations.get((struct_id, 0), []) if len(d) == 2 and d[0] == DEC_OFFSET), None)
        binding = next((d[1] for d in decorations.get(var, []) if len(d) == 2 and d[0] == DEC_BINDING), None)
        dset = next((d[1] for d in decorations.get(var, []) if len(d) == 2 and d[0] == DEC_DESCRIPTOR_SET), None)
        if scalar_bits != 32 or stride != 4 or block is None or offset != 0 or binding is None or dset is None:
            rejected['shape'] += 1; continue
        if len(locations.get((dset, binding), ())) != 1:
            rejected['aliasing'] += 1; continue
        eligible[var] = {'name': names[var], 'storage': storage, 'scalar': scalar,
                         'block': block, 'binding': binding, 'set': dset,
                         'decorations': vdecs}

    const_expr = {}
    for result, (ty, words) in constants.items():
        if ty in ints:
            value = _literal(words, ints[ty][0])
            if value is not None: const_expr[result] = _constant(value, ints[ty][0])

    # Segments contain consecutive external-memory reads only. Local/private
    # loads, stores and integer arithmetic do not break them.
    access, expr, local_value = {}, dict(const_expr), {}
    segments, current, current_fn = [], [], None
    def close():
        nonlocal current
        if current: segments.append(current); current = []
    for ix, (op, a) in enumerate(ins):
        if op == OP_FUNCTION:
            close(); current_fn = a[1] if len(a) >= 2 else None; expr = dict(const_expr); local_value = {}; access = {}
        elif op == OP_FUNCTION_END:
            close(); current_fn = None
        elif op == OP_LABEL:
            close(); expr = dict(const_expr); local_value = {}; access = {}

        get = expr.get
        if op == OP_VARIABLE and len(a) >= 3 and a[2] in (SC_FUNCTION, SC_PRIVATE):
            local_value[a[1]] = None
        elif op == OP_STORE and len(a) >= 2:
            pty = type_of.get(a[0]); storage = pointers.get(pty, (None, None))[0]
            if storage in (SC_FUNCTION, SC_PRIVATE) and a[0] in variables:
                local_value[a[0]] = get(a[1])
            else:
                close()
                if storage in (SC_FUNCTION, SC_PRIVATE):
                    # An aliasing local pointer may have changed any tracked
                    # value. Keep constants only; do not reuse a stale proof.
                    expr = dict(const_expr); local_value = {}
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(a) == 5:
            index_ty = type_of.get(a[4])
            if a[2] in eligible and index_ty in ints and get(a[4]) is not None:
                access[a[1]] = {'var': a[2], 'index': a[4], 'expr': get(a[4]),
                                'index_type': index_ty, 'member': a[3]}
        elif op == OP_LOAD and len(a) == 3:
            pty = type_of.get(a[2]); storage = pointers.get(pty, (None, None))[0]
            if storage in (SC_FUNCTION, SC_PRIVATE) and a[2] in variables:
                expr[a[1]] = local_value.get(a[2]) or _leaf(('var', a[2]), ints.get(a[0], (32, 0))[0])
            elif a[2] in access and a[0] == eligible[access[a[2]]['var']]['scalar']:
                ac = access[a[2]]
                current.append({'ix': ix, 'result': a[1], 'type': a[0],
                                'var': ac['var'], 'index': ac['index'], 'expr': ac['expr'],
                                'index_type': ac['index_type'], 'member': ac['member'],
                                'function': current_fn})
                expr[a[1]] = _leaf(('load', a[1]), ints.get(a[0], (32, 0))[0])
            else:
                if storage not in (SC_FUNCTION, SC_PRIVATE): close()
                if a[0] in ints: expr[a[1]] = _leaf(('load', a[1]), ints[a[0]][0])
        elif op == OP_BITCAST and len(a) == 3:
            src, dst = get(a[2]), ints.get(a[0])
            expr[a[1]] = src if src is not None and dst and src[2] == dst[0] else None
        elif op == OP_COPY_OBJECT and len(a) == 3: expr[a[1]] = get(a[2])
        elif op in (OP_IADD, OP_ISUB) and len(a) == 4: expr[a[1]] = _add(get(a[2]), get(a[3]), op == OP_ISUB)
        elif op == OP_IMUL and len(a) == 4: expr[a[1]] = _mul(get(a[2]), get(a[3]))
        elif op == OP_SHIFT_LEFT_LOGICAL and len(a) == 4 and get(a[3]) and not get(a[3])[0]:
            shift = get(a[3])[1]
            expr[a[1]] = _mul(get(a[2]), _constant(1 << shift, get(a[3])[2])) if shift < get(a[3])[2] else None
        else:
            r = _result(op, a)
            if r and r[0] in ints and r[1] not in expr: expr[r[1]] = _leaf(('ssa', r[1]), ints[r[0]][0])

        r = _result(op, a)
        if (op in _UNPROVEN_POINTER_DERIVATIONS and r and
                (r[0] in pointers or op == OP_CONVERT_PTR_TO_U)):
            close(); expr = dict(const_expr); local_value = {}

        # Image sampling/read/write is an observable memory operation too. Do
        # not move a storage read across it even when a binding appears
        # unrelated: Vulkan descriptors may alias the same allocation.
        if _memory_fence(op):
            close(); expr = dict(const_expr); local_value = {}; access = {}
        elif current_fn is not None and not _segment_safe(op, a):
            # Unknown operations are unsafe even before the first candidate:
            # they may mutate a Function/Private pointer used by a later index.
            close(); expr = dict(const_expr); local_value = {}; access = {}
    close()

    groups = []
    for segment in segments:
        i = 0
        while i < len(segment):
            made = None
            for width in (4, 2):
                part = segment[i:i + width]
                if len(part) != width or len({x['var'] for x in part}) != 1 or len({x['type'] for x in part}) != 1: continue
                ex = [x['expr'] for x in part]
                if any(v is None for v in ex) or len({(v[0], v[2]) for v in ex}) != 1: continue
                values, mask = [v[1] for v in ex], (1 << ex[0][2]) - 1
                low = min(values)
                if sorted(values) != list(range(low, low + width)) or low + width - 1 > mask: continue
                if low % width or any(coef % width for _, coef in ex[0][0]): continue
                made = {'width': width, 'loads': part, 'low': low}; break
            if made: groups.append(made); i += made['width']
            else: i += 1

    if not groups:
        return data, {'readonly_variables': len(eligible), 'groups2': 0, 'groups4': 0,
                      'scalar_loads': 0, 'aliases': 0, 'rejected': rejected,
                      '_transformed_variables': []}

    next_id = header[3]
    def new_id():
        nonlocal next_id
        result = next_id; next_id += 1; return result

    const_ids = {}
    for cid, (ty, words) in constants.items():
        if ty in ints:
            value = _literal(words, ints[ty][0])
            if value is not None: const_ids.setdefault((ty, value), cid)
    new_constants = []
    def const_id(ty, value):
        bits = ints[ty][0]; value &= (1 << bits) - 1; key = (ty, value)
        if key not in const_ids:
            cid = new_id(); const_ids[key] = cid
            words = [value & 0xffffffff] + ([value >> 32] if bits == 64 else [])
            new_constants.append([OP_CONSTANT, [ty, cid, *words]])
        return const_ids[key]

    new_names, new_decorations, new_types, new_variables = [], [], [], []
    shape_cache, aliases = {}, {}
    pointer_ids = {(storage, pointee): ident
                   for ident, (storage, pointee) in pointers.items()}
    def vector_type(scalar, width):
        key = (scalar, width)
        if key not in vectors:
            vectors[key] = new_id()
            new_types.append([OP_TYPE_VECTOR, [vectors[key], scalar, width]])
        return vectors[key]
    def pointer_type(storage, pointee):
        key = (storage, pointee)
        if key not in pointer_ids:
            pointer_ids[key] = new_id()
            new_types.append([OP_TYPE_POINTER, [pointer_ids[key], storage, pointee]])
        return pointer_ids[key]
    for group in groups:
        var, width = group['loads'][0]['var'], group['width']
        if (var, width) in aliases: continue
        info = eligible[var]; shape = (info['scalar'], info['storage'], info['block'], width)
        if shape not in shape_cache:
            vec = vector_type(info['scalar'], width)
            arr, struct_id = new_id(), new_id()
            new_types += [[OP_TYPE_RUNTIME_ARRAY, [arr, vec]],
                          [OP_TYPE_STRUCT, [struct_id, arr]]]
            pstruct = pointer_type(info['storage'], struct_id)
            pvec = pointer_type(info['storage'], vec)
            shape_cache[shape] = (vec, arr, struct_id, pstruct, pvec)
            new_decorations += [[OP_DECORATE, [arr, DEC_ARRAY_STRIDE, 4 * width]],
                                [OP_DECORATE, [struct_id, info['block']]],
                                [OP_MEMBER_DECORATE, [struct_id, 0, DEC_OFFSET, 0]]]
        vec, _, _, pstruct, pvec = shape_cache[shape]
        alias = new_id(); aliases[(var, width)] = (alias, vec, pvec)
        new_variables.append([OP_VARIABLE, [pstruct, alias, info['storage']]])
        new_names.append([OP_NAME, [alias, *_string_words(info['name'] + '_v%d' % width)]])
        new_decorations += [[OP_DECORATE, [alias, DEC_DESCRIPTOR_SET, info['set']]],
                            [OP_DECORATE, [alias, DEC_BINDING, info['binding']]],
                            [OP_DECORATE, [alias, DEC_NON_WRITABLE]]]
        for dec in (DEC_ALIASED,):
            if dec in info['decorations']: new_decorations.append([OP_DECORATE, [alias, dec]])

    prefix, rewrite = {}, {}
    for group in groups:
        width, loads = group['width'], group['loads']; first = loads[0]
        alias, vec_type, ptr_vec = aliases[(first['var'], width)]
        idx = first['index']; mask = (1 << first['expr'][2]) - 1
        delta = (first['expr'][1] - group['low']) & mask; code = []
        if delta:
            adjusted = new_id()
            code.append([OP_ISUB, [first['index_type'], adjusted, idx, const_id(first['index_type'], delta)]])
            idx = adjusted
        vector_index = new_id()
        code.append([OP_SHIFT_RIGHT_LOGICAL, [first['index_type'], vector_index, idx,
                                              const_id(first['index_type'], 1 if width == 2 else 2)]])
        vector_ptr, vector_value = new_id(), new_id()
        code.append([OP_ACCESS_CHAIN, [ptr_vec, vector_ptr, alias, first['member'], vector_index]])
        code.append([OP_LOAD, [vec_type, vector_value, vector_ptr]])
        prefix.setdefault(first['ix'], []).extend(code)
        for load in loads:
            component = (load['expr'][1] - group['low']) & mask
            rewrite[load['ix']] = [OP_COMPOSITE_EXTRACT, [load['type'], load['result'], vector_value, component]]

    annotation_at = next((i for i, (op, _) in enumerate(ins) if op in (OP_DECORATE, OP_MEMBER_DECORATE)), None)
    type_at = next((i for i, (op, _) in enumerate(ins) if 19 <= op <= 39), None)
    fn_at = next((i for i, (op, _) in enumerate(ins) if op == OP_FUNCTION), None)
    if type_at is None or fn_at is None: raise VectorLoadError('module has no type/function section')
    if annotation_at is None: annotation_at = type_at
    out = []
    # Before SPIR-V 1.4 the interface is restricted to Input/Output variables.
    # From 1.4 onward every statically used global must be listed, regardless of
    # whether the descriptor uses StorageBuffer or legacy Uniform+BufferBlock.
    global_interfaces = _entrypoint_aliases(
        header[1], (alias for alias, _, _ in aliases.values()))
    for ix, inst in enumerate(ins):
        if ix == annotation_at: out.extend(new_names); out.extend(new_decorations)
        if ix == fn_at: out.extend(new_types); out.extend(new_constants); out.extend(new_variables)
        out.extend(prefix.get(ix, ()))
        emitted = rewrite.get(ix, inst)
        # SPIR-V 1.4+ lists every used global in the entry-point interface;
        # older modules list only Input/Output variables.
        if emitted[0] == 15 and global_interfaces:
            emitted = [emitted[0], emitted[1] + global_interfaces]
        out.append(emitted)
    header[3] = next_id
    return _encode(header, out), {
        'readonly_variables': len(eligible),
        'groups2': sum(g['width'] == 2 for g in groups),
        'groups4': sum(g['width'] == 4 for g in groups),
        'scalar_loads': sum(g['width'] for g in groups),
        'aliases': len(aliases), 'rejected': rejected,
        '_transformed_variables': sorted({g['loads'][0]['var'] for g in groups})}


def optimize_vector_loads(data, reflection):
    """Materialize read-only facts, then run the strictly gated coalescer."""
    marked, facts = materialize_nonwritable(data, reflection)
    patched, report = transform_vector_loads(marked, reflection)
    materialized = set(facts.pop('_materialized_variables'))
    transformed = set(report.pop('_transformed_variables'))
    unused = materialized - transformed
    if unused and report['scalar_loads']:
        header, instructions = _decode(patched)
        instructions = [
            (op, a) for op, a in instructions
            if not (op == OP_DECORATE and len(a) >= 2 and
                    a[0] in unused and a[1] == DEC_NON_WRITABLE)
        ]
        patched = _encode(header, instructions)
    facts['materialized_nonwritable'] = len(materialized & transformed)
    report.update(facts)
    # NonWritable is materialized only as a proof input. Do not churn every
    # shader merely to add it when no scalar load can actually be coalesced.
    return (patched if report['scalar_loads'] else data), report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input'); p.add_argument('reflection'); p.add_argument('output')
    a = p.parse_args(argv)
    with open(a.input, 'rb') as f: data = f.read()
    with open(a.reflection) as f: reflection = json.load(f)
    patched, report = optimize_vector_loads(data, reflection)
    with open(a.output, 'wb') as f: f.write(patched)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__': main()
