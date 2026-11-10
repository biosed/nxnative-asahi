"""Vulkan SPIR-V lowering for the translator's Ampere HMMA.16816 helper.

The translator deliberately keeps one named scalar helper in SPIR-V.  Metal
replaces it with ``simdgroup_matrix`` code after SPIRV-Cross; Vulkan used to
execute the helper literally, moving every fragment through workgroup memory
and loading every matrix word for every output.  This module recognizes the
current TensorPass ABI (17 input pointers plus D1) and replaces each call with
the same ordered f32 FMA calculation while sharing gathered A/B fragments
between calls in a basic block.

The small SPIR-V reader and the lowering are derived from BananaNative's
``vk_android_spv.py`` at commit 45d2b3424204b803b549202cc7e33bdf540f2ea6,
used under that repository's MIT licence.  The current-ABI recognition,
literal-safe scratch ownership check and Linux integration are local changes.
"""

import os
import struct
import subprocess


MAGIC = 0x07230203

# SPIR-V opcodes used by this transform.
OP_NAME, OP_EXT_INST_IMPORT, OP_EXT_INST, OP_CAPABILITY = 5, 11, 12, 17
OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR = 21, 22, 23
OP_TYPE_ARRAY, OP_TYPE_POINTER = 28, 32
OP_CONSTANT, OP_FUNCTION, OP_FUNCTION_PARAMETER = 43, 54, 55
OP_FUNCTION_END, OP_FUNCTION_CALL, OP_VARIABLE = 56, 57, 59
OP_LOAD, OP_STORE = 61, 62
OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN = 65, 66
OP_PTR_ACCESS_CHAIN, OP_ARRAY_LENGTH, OP_IN_BOUNDS_PTR_ACCESS_CHAIN = 67, 68, 70
OP_DECORATE = 71
OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT = 80, 81
OP_COPY_OBJECT, OP_BITCAST = 83, 124
OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR = 121, 122
OP_GENERIC_CAST_TO_PTR_EXPLICIT = 123
OP_SELECT = 169
OP_IADD, OP_ISUB, OP_IMUL = 128, 130, 132
OP_SHR_LOGICAL, OP_SHR_ARITH, OP_SHL_LOGICAL = 194, 195, 196
OP_BITWISE_OR, OP_BITWISE_XOR, OP_BITWISE_AND = 197, 198, 199
OP_CONTROL_BARRIER, OP_MEMORY_BARRIER = 224, 225
OP_PHI, OP_LABEL, OP_RETURN, OP_RETURN_VALUE = 245, 248, 253, 254
OP_COPY_LOGICAL, OP_PTR_EQUAL, OP_PTR_NOT_EQUAL, OP_PTR_DIFF = 400, 401, 402, 403

SC_INPUT, SC_WORKGROUP, SC_FUNCTION = 1, 4, 7
GLSL_FMA, GLSL_PACK_HALF_2X16, GLSL_UNPACK_HALF_2X16 = 50, 58, 62
VN_OPS = {
    OP_IADD, OP_ISUB, OP_IMUL, OP_SHR_LOGICAL, OP_SHR_ARITH,
    OP_SHL_LOGICAL, OP_BITWISE_OR, OP_BITWISE_XOR, OP_BITWISE_AND,
}
NO_RESULT_OPS = {
    OP_STORE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, OP_RETURN,
    OP_RETURN_VALUE, 246, 247, 249, 250, 251, 252, OP_FUNCTION_END,
}

HMMA_NAME = 'dmsh_hmma16816_f16'
HMMA_INPUTS = 17
HMMA_PARAMETERS = HMMA_INPUTS + 1       # D1 is the final out pointer
HMMA_OUT_ARGUMENT = HMMA_INPUTS


def string_at(words, i):
    """Return a SPIR-V literal string and the word index following it."""
    out = bytearray()
    while i < len(words):
        b = struct.pack('<I', words[i]); i += 1
        if 0 in b:
            out += b[:b.index(0)]
            break
        out += b
    return out.decode('utf-8', 'replace'), i


def lit_string(text):
    b = text.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack('<%dI' % (len(b) // 4), b))


class Module:
    """The subset of a SPIR-V module needed by :func:`hmma_lds`."""

    def __init__(self, data):
        if len(data) % 4 or len(data) < 20:
            raise ValueError('not a SPIR-V module (length)')
        words = list(struct.unpack('<%dI' % (len(data) // 4), data))
        if words[0] != MAGIC:
            raise ValueError('not a SPIR-V module (magic)')
        self.header, self.ins, i = words[:5], [], 5
        while i < len(words):
            n = words[i] >> 16
            if not n or i + n > len(words):
                raise ValueError('malformed SPIR-V instruction at word %d' % i)
            self.ins.append(words[i:i + n]); i += n
        self.index()

    def index(self):
        self.names, self.types, self.consts, self.vars = {}, {}, {}, {}
        for x in self.ins:
            op = x[0] & 0xffff
            if op == OP_NAME:
                self.names[x[1]] = string_at(x, 2)[0]
            elif 19 <= op <= 39:
                self.types[x[1]] = x
            elif op == OP_CONSTANT:
                self.consts[x[2]] = (x[1], x[3:])
            elif op == OP_VARIABLE and len(x) >= 4:
                self.vars[x[2]] = (x[1], x[3])

    def data(self):
        return struct.pack(
            '<%dI' % (5 + sum(len(x) for x in self.ins)),
            *self.header, *(v for x in self.ins for v in x),
        )

    def new_id(self):
        ident = self.header[3]
        self.header[3] += 1
        return ident

    def op(self, ident):
        return self.types[ident][0] & 0xffff if ident in self.types else None

    def pointee(self, ptr):
        t = self.types[ptr]
        return t[2], t[3]

    def int_type(self, ty):
        x = self.types.get(ty)
        return (x[2], x[3]) if x and x[0] & 0xffff == OP_TYPE_INT else None

    def find_type(self, op, *operands):
        for ident, x in self.types.items():
            if x[0] & 0xffff == op and x[2:] == list(operands):
                return ident
        return None

    def add_type(self, op, *operands):
        ident = self.find_type(op, *operands)
        if ident is not None:
            return ident
        ident = self.new_id()
        x = [(2 + len(operands)) << 16 | op, ident, *operands]
        self._insert_decl(x); self.types[ident] = x
        return ident

    def add_const(self, ty, value):
        for ident, (ctype, vals) in self.consts.items():
            if ctype == ty and vals == [value]:
                return ident
        ident = self.new_id()
        x = [4 << 16 | OP_CONSTANT, ty, ident, value]
        self._insert_decl(x); self.consts[ident] = (ty, [value])
        return ident

    def _insert_decl(self, x):
        at = next(k for k, y in enumerate(self.ins)
                  if y[0] & 0xffff == OP_FUNCTION)
        self.ins.insert(at, x)

    def function_range(self, fid):
        first = next(k for k, x in enumerate(self.ins)
                     if x[0] & 0xffff == OP_FUNCTION and x[2] == fid)
        last = next(k for k in range(first, len(self.ins))
                    if self.ins[k][0] & 0xffff == OP_FUNCTION_END)
        return first, last


def functions_named(module):
    return {
        module.names.get(x[2], ''): x[2] for x in module.ins
        if x[0] & 0xffff == OP_FUNCTION
    }


def ensure_glsl(module):
    """Return the GLSL.std.450 import, inserting it before OpMemoryModel."""
    for x in module.ins:
        if (x[0] & 0xffff == OP_EXT_INST_IMPORT and
                string_at(x, 2)[0] == 'GLSL.std.450'):
            return x[1]
    ident = module.new_id(); text = lit_string('GLSL.std.450')
    at = next(k for k, x in enumerate(module.ins) if x[0] & 0xffff == 14)
    module.ins.insert(
        at, [(2 + len(text)) << 16 | OP_EXT_INST_IMPORT, ident, *text],
    )
    return ident


class BlockValues:
    """Value numbers for translator registers represented by Function variables."""

    def __init__(self, module):
        self.m = module
        self.inptr = {}
        self.function()

    def function(self):
        self.fvars, self.fbase = set(), {}
        self.ver, self.stored, self.vn = {}, {}, {}

    def block(self):
        self.stored = {}

    def __call__(self, ident):
        return self.vn.get(ident, ident)

    def key_of_var(self, ptr):
        base = self.fbase.get(ptr, ptr)
        if ptr in self.stored:
            return self(self.stored[ptr])
        return ('ld', base, self.ver.get(base, 0))

    def bump(self, ptr):
        base = self.fbase.get(ptr, ptr)
        self.ver[base] = self.ver.get(base, 0) + 1
        self.stored.pop(base, None)

    def see(self, x):
        m, op = self.m, x[0] & 0xffff
        if op == OP_FUNCTION:
            self.function()
        elif op == OP_LABEL:
            self.block()
        elif ((op == OP_VARIABLE and len(x) > 3 and x[3] == SC_FUNCTION) or
              (op == OP_FUNCTION_PARAMETER and m.op(x[1]) == OP_TYPE_POINTER
               and m.types[x[1]][2] == SC_FUNCTION)):
            self.fvars.add(x[2])
        elif (op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and
              (x[3] in self.fvars or x[3] in self.fbase)):
            self.fbase[x[2]] = self.fbase.get(x[3], x[3])
        elif op == OP_STORE and (x[1] in self.fvars or x[1] in self.fbase):
            self.bump(x[1])
            if x[1] in self.fvars:
                self.stored[x[1]] = x[2]
        elif op == OP_LOAD and x[3] in self.fvars:
            self.vn[x[2]] = self.key_of_var(x[3])
        elif (op == OP_LOAD and
              (x[3] in self.inptr or
               m.vars.get(x[3], (0, 0))[1] == SC_INPUT)):
            self.vn[x[2]] = ('in', self.inptr.get(x[3], x[3]))
        elif (op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and
              m.vars.get(x[3], (0, 0))[1] == SC_INPUT and
              all(ident in m.consts for ident in x[4:])):
            self.inptr[x[2]] = (x[3], *x[4:])
        elif op in (OP_BITCAST, OP_COPY_OBJECT):
            self.vn[x[2]] = self(x[3])
        elif op in VN_OPS:
            self.vn[x[2]] = (op, self(x[3]), self(x[4]))
        elif op == OP_FUNCTION_CALL:
            for ptr in x[4:]:
                if ptr in self.fvars or ptr in self.fbase:
                    self.bump(ptr)


def _scratch_used_outside(module, scratch, first, last):
    """Whether the scratch variable is read/written outside the helper.

    SPIR-V words mix ids and literals.  Searching every operand for ``scratch``
    rejects valid modules when, for example, a capability number happens to
    equal its result id.  Only pointer-bearing operand positions count here.
    """
    for k, x in enumerate(module.ins):
        if first <= k <= last:
            continue
        if _pointer_uses((x,), scratch):
            return True
    return False


def _pointer_uses(instructions, pointer):
    """Pointer-bearing uses of ``pointer`` as ``(opcode, instruction)``."""
    out = []
    for x in instructions:
        op = x[0] & 0xffff
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN,
                  OP_PTR_ACCESS_CHAIN, OP_IN_BOUNDS_PTR_ACCESS_CHAIN,
                  OP_ARRAY_LENGTH):
            used = len(x) > 3 and x[3] == pointer
        elif op == OP_LOAD:
            used = len(x) > 3 and x[3] == pointer
        elif op == OP_STORE:
            used = len(x) > 2 and pointer in x[1:3]
        elif op in (OP_COPY_OBJECT, OP_BITCAST, OP_COPY_LOGICAL,
                    OP_PTR_CAST_TO_GENERIC, OP_GENERIC_CAST_TO_PTR,
                    OP_GENERIC_CAST_TO_PTR_EXPLICIT):
            used = len(x) > 3 and x[3] == pointer
        elif op == OP_SELECT:
            used = len(x) > 5 and pointer in x[4:6]
        elif op == OP_PHI:
            used = pointer in x[3::2]
        elif op in (OP_PTR_EQUAL, OP_PTR_NOT_EQUAL, OP_PTR_DIFF):
            used = len(x) > 4 and pointer in x[3:5]
        elif op == OP_FUNCTION_CALL:
            used = pointer in x[4:]
        elif op == OP_RETURN_VALUE:
            used = len(x) > 1 and x[1] == pointer
        else:
            used = False
        if used:
            out.append((op, x))
    return out


def _current_scratch_array(module, array, u32):
    """Whether ``array`` can hold the current 256-word scratch ABI.

    The translator declares one helper-private Workgroup array for the whole
    module.  Its size follows the module's shared-memory requirement and is
    therefore sometimes larger than the 256 words addressed by HMMA.  The
    lowering preserves its total byte size while retyping it to uvec4, so a
    larger multiple-of-four word count is safe; a smaller or partial vector is
    not.
    """
    declaration = module.types.get(array)
    if not declaration or declaration[0] & 0xffff != OP_TYPE_ARRAY:
        return False
    length = module.consts.get(declaration[3])
    if declaration[2] != u32 or not length or length[0] != u32:
        return False
    words = length[1]
    return len(words) == 1 and words[0] >= 256 and words[0] % 4 == 0


def hmma_helper(module):
    """Recognize the current 18-pointer TensorPass helper, or return ``None``."""
    fid = functions_named(module).get(HMMA_NAME)
    if fid is None:
        return None
    first, last = module.function_range(fid)
    i32 = module.ins[first][1]
    if module.int_type(i32) != (32, 1):
        return None
    params = [x for x in module.ins[first + 1:last]
              if x[0] & 0xffff == OP_FUNCTION_PARAMETER]
    if (len(params) != HMMA_PARAMETERS or
            any(module.types.get(p[1], [0, 0, 0, 0])[2:4] !=
                [SC_FUNCTION, i32] for p in params)):
        return None
    body = module.ins[first + 1 + len(params):last]
    if sum(1 for x in body if x[0] & 0xffff == OP_LABEL) != 1:
        return None
    # TensorPass's seven facts are promises used by target postprocessors, not
    # by the scalar reference body.  Fail closed if a future translator starts
    # consuming one.  D1 is still exactly one final pointer store.
    if any(_pointer_uses(body, p[2]) for p in params[10:17]):
        return None
    out_uses = _pointer_uses(body, params[HMMA_OUT_ARGUMENT][2])
    if len(out_uses) != 1 or out_uses[0][0] != OP_STORE:
        return None
    chains = [x for x in body if x[0] & 0xffff in
              (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN)]
    if len(chains) != 8 + 64 or len({x[3] for x in chains}) != 1:
        return None
    scratch, pword = chains[0][3], chains[0][1]
    if (module.vars.get(scratch, (0, 0))[1] != SC_WORKGROUP or
            module.types.get(pword, [0, 0, 0])[2] != SC_WORKGROUP):
        return None
    u32 = module.pointee(pword)[1]
    array = module.pointee(module.vars[scratch][0])[1]
    if (module.int_type(u32) != (32, 0) or
            not _current_scratch_array(module, array, u32)):
        return None
    if _scratch_used_outside(module, scratch, first, last):
        return None
    groups, current = [], None
    for x in body:
        if x[0] & 0xffff in (OP_CONTROL_BARRIER, OP_MEMORY_BARRIER):
            if current is None:
                current = []
                groups.append(current)
            current.append(list(x))
        else:
            current = None
    if len(groups) != 2 or groups[0] != groups[1]:
        return None
    return {
        'fid': fid, 'i32': i32, 'u32': u32,
        'scratch': scratch, 'barrier': groups[0],
    }


# Gathered A / B fragments kept for reuse within a basic block, most recently
# used first.  Every kept fragment is 32 half-float values that stay live until
# its last reuse: an unbounded cache kept ten B matrices live across thousands
# of instructions in the NDSR kernel d99fbeeb53d38847 (five convolution layers
# in one block), more than the AGX register file holds, and the driver spilled
# 852 values and filled 877 per thread (36 / 31 before this lowering existed).
# Two of each keeps the adjacent reuse the kernels actually have.
# DMSH_HMMA_CACHE=<a>,<b> overrides it for measurements (0 = unbounded).
HMMA_CACHE_A, HMMA_CACHE_B = 2, 2


def _cache_limits():
    text = os.environ.get('DMSH_HMMA_CACHE')
    if not text:
        return HMMA_CACHE_A, HMMA_CACHE_B
    a, b = (int(part) for part in text.split(','))
    if a < 0 or b < 0:
        raise ValueError('DMSH_HMMA_CACHE=%s: limits are not negative' % text)
    return a, b


def _cache_keep(cache, key, limits):
    """Mark key most recently used and drop the oldest of its kind past the limit."""
    cache[key] = cache.pop(key)
    limit = limits[0] if key[0] == 'A' else limits[1]
    if not limit:
        return
    same = [k for k in cache if k[0] == key[0]]
    for old in same[:max(0, len(same) - limit)]:
        del cache[old]


def hmma_lds(module):
    """Lower every recognized HMMA call, returning the number replaced.

    A and B fragments with the same block value number are gathered once and
    reused.  C remains lane-local.  The FMA order and final f16 rounding are
    identical to the translator helper.  TensorPass's seven appended facts are
    deliberately not consumed yet; they do not alter the helper's result.
    """
    helper = hmma_helper(module)
    if helper is None:
        return 0
    fid = helper['fid']
    i32, u32, scratch = helper['i32'], helper['u32'], helper['scratch']
    glsl = ensure_glsl(module)
    f32 = module.add_type(OP_TYPE_FLOAT, 32)
    v2f = module.add_type(OP_TYPE_VECTOR, f32, 2)
    uvec4 = module.add_type(OP_TYPE_VECTOR, u32, 4)
    array = module.pointee(module.vars[scratch][0])[1]
    words = module.consts[module.types[array][3]][1][0]
    array4 = module.add_type(
        OP_TYPE_ARRAY, uvec4, module.add_const(u32, words // 4),
    )
    p_array4 = module.add_type(OP_TYPE_POINTER, SC_WORKGROUP, array4)
    p_uvec4 = module.add_type(OP_TYPE_POINTER, SC_WORKGROUP, uvec4)
    p_u32 = module.add_type(OP_TYPE_POINTER, SC_WORKGROUP, u32)
    at = next(k for k, x in enumerate(module.ins)
              if x[0] & 0xffff == OP_VARIABLE and x[2] == scratch)
    var = module.ins.pop(at); var[1] = p_array4
    module._insert_decl(var)
    ci = {v: module.add_const(i32, v) for v in (1, 2, 3, 8, 32)}
    cu = {v: module.add_const(u32, v) for v in (0, 1)}
    module.index()
    calls = 0
    functions = [x[2] for x in module.ins
                 if x[0] & 0xffff == OP_FUNCTION and x[2] != fid]
    for function in functions:
        first, last = module.function_range(function)
        out, values, cache = [], BlockValues(module), {}
        limits = _cache_limits()

        def emit(op, ty, *operands):
            result = module.new_id()
            out.append([(3 + len(operands)) << 16 | op,
                        ty, result, *operands])
            return result

        def integer(op, lhs, rhs):
            return emit(op, i32, lhs, rhs)

        for x in module.ins[first:last + 1]:
            op = x[0] & 0xffff
            if op == OP_LABEL:
                cache = {}
            if not (op == OP_FUNCTION_CALL and x[3] == fid):
                values.see(x); out.append(x)
                continue
            args = x[4:4 + HMMA_PARAMETERS]
            if len(args) != HMMA_PARAMETERS:
                raise ValueError('%s call has %d arguments, expected %d' %
                                 (HMMA_NAME, len(args), HMMA_PARAMETERS))
            keys = [values.key_of_var(ptr) for ptr in args[:10]]
            key_a = ('A', *keys[:6])
            key_b = ('B', *keys[:2], *keys[6:8])
            needed = [key for key in (key_a, key_b) if key not in cache]
            if needed:
                load_indices = {0, 1}
                if key_a in needed:
                    load_indices.update((2, 3, 4, 5))
                if key_b in needed:
                    load_indices.update((6, 7))
                vals = {
                    i: emit(OP_LOAD, i32, args[i])
                    for i in sorted(load_indices)
                }
                lane, base = vals[0], vals[1]
                group = integer(OP_SHR_LOGICAL, lane, ci[2])
                thread = integer(OP_BITWISE_AND, lane, ci[3])
                base4 = integer(OP_SHR_LOGICAL, base, ci[2])
                vi = integer(OP_SHR_LOGICAL, thread, ci[1])
                comp0 = emit(
                    OP_BITCAST, u32,
                    integer(
                        OP_SHL_LOGICAL,
                        integer(OP_BITWISE_AND, thread, ci[1]), ci[1],
                    ),
                )
                comp1 = emit(OP_IADD, u32, comp0, cu[1])

                def row_a(row):
                    return integer(
                        OP_IADD, base4,
                        integer(OP_SHL_LOGICAL, row, ci[1]),
                    )

                def row_b(col):
                    return integer(
                        OP_IADD, base4,
                        integer(
                            OP_IADD, ci[32],
                            integer(OP_SHL_LOGICAL, col, ci[1]),
                        ),
                    )

                def store(row, word0, word1):
                    vec = integer(OP_IADD, row, vi)
                    for component, word in ((comp0, word0), (comp1, word1)):
                        pointer = emit(
                            OP_ACCESS_CHAIN, p_u32, scratch, vec, component,
                        )
                        out.append([
                            3 << 16 | OP_STORE, pointer,
                            emit(OP_BITCAST, u32, word),
                        ])

                group8 = integer(OP_IADD, group, ci[8])
                if key_a in needed:
                    store(row_a(group), vals[2], vals[4])
                    store(row_a(group8), vals[3], vals[5])
                if key_b in needed:
                    store(row_b(group), vals[6], vals[7])
                out.extend(list(barrier) for barrier in helper['barrier'])

                def load_row(row):
                    packed = []
                    for part in range(2):
                        pointer = emit(
                            OP_ACCESS_CHAIN, p_uvec4, scratch,
                            integer(OP_IADD, row, ci[1]) if part else row,
                        )
                        vec = emit(OP_LOAD, uvec4, pointer)
                        packed += [emit(OP_COMPOSITE_EXTRACT, u32, vec, c)
                                   for c in range(4)]
                    unpacked = [
                        emit(
                            OP_EXT_INST, v2f, glsl, GLSL_UNPACK_HALF_2X16,
                            packed[2 * (kw % 4) + kw // 4],
                        ) for kw in range(8)
                    ]
                    return [
                        [emit(OP_COMPOSITE_EXTRACT, f32, pair, half)
                         for half in range(2)]
                        for pair in unpacked
                    ]

                if key_a in needed:
                    cache[key_a] = [
                        load_row(row_a(group)), load_row(row_a(group8)),
                    ]
                if key_b in needed:
                    thread2 = integer(OP_SHL_LOGICAL, thread, ci[1])
                    cache[key_b] = [
                        load_row(row_b(thread2)),
                        load_row(row_b(integer(OP_IADD, thread2, ci[1]))),
                    ]
                out.extend(list(barrier) for barrier in helper['barrier'])
            a_fragments, b_fragments = cache[key_a], cache[key_b]
            _cache_keep(cache, key_a, limits)
            _cache_keep(cache, key_b, limits)
            cvals = [emit(OP_LOAD, i32, args[i]) for i in (8, 9)]
            result_words = []
            for row in range(2):
                cvec = emit(
                    OP_EXT_INST, v2f, glsl, GLSL_UNPACK_HALF_2X16,
                    emit(OP_BITCAST, u32, cvals[row]),
                )
                accum = [
                    emit(OP_COMPOSITE_EXTRACT, f32, cvec, 0),
                    emit(OP_COMPOSITE_EXTRACT, f32, cvec, 1),
                ]
                for col in range(2):
                    for kw in range(8):
                        for half in range(2):
                            accum[col] = emit(
                                OP_EXT_INST, f32, glsl, GLSL_FMA,
                                a_fragments[row][kw][half],
                                b_fragments[col][kw][half],
                                accum[col],
                            )
                pair = emit(OP_COMPOSITE_CONSTRUCT, v2f, *accum)
                packed = emit(
                    OP_EXT_INST, u32, glsl, GLSL_PACK_HALF_2X16, pair,
                )
                result_words.append(emit(OP_BITCAST, i32, packed))
            store_out = [
                3 << 16 | OP_STORE,
                args[HMMA_OUT_ARGUMENT], result_words[1],
            ]
            out.append(store_out); values.see(store_out)
            out.append([
                4 << 16 | OP_COPY_OBJECT,
                i32, x[2], result_words[0],
            ])
            calls += 1
        module.ins[first:last + 1] = out
    first, last = module.function_range(fid)
    removed = {
        x[2] for x in module.ins[first:last + 1]
        if len(x) > 2 and (x[0] & 0xffff) not in NO_RESULT_OPS
    } | {
        x[1] for x in module.ins[first:last + 1]
        if x[0] & 0xffff == OP_LABEL
    }
    del module.ins[first:last + 1]
    module.ins = [
        x for x in module.ins
        if not (x[0] & 0xffff in (OP_NAME, OP_DECORATE) and x[1] in removed)
    ]
    module.index()
    return calls


DERIVED_SUFFIXES = (
    '.spv', '.res.json', '.reader.spv', '.vk.spv', '.vk.json', '.vkb',
    '.job.json', '.spv.hmma.tmp',
)


def remove_outputs(folder, key):
    """Remove every source and derived file for a failed shader key."""
    for suffix in DERIVED_SUFFIXES:
        try:
            os.unlink(os.path.join(folder, key + suffix))
        except FileNotFoundError:
            pass


def lower_file(path, validator=None, run=subprocess.run):
    """Lower one file atomically and validate it for Vulkan 1.3.

    A named helper that does not match the current closed recognizer is ABI
    drift, not a shader without HMMA: fail loudly instead of retaining the
    scalar implementation under a successful build.
    """
    with open(path, 'rb') as source:
        module = Module(source.read())
    named = HMMA_NAME in functions_named(module)
    calls = hmma_lds(module)
    if named and not calls:
        raise ValueError('%s is present but does not match the current ABI' %
                         HMMA_NAME)
    if not calls:
        return 0
    if not validator:
        raise ValueError('HMMA lowering needs SPIRV_VAL or spirv-val on PATH')
    temp = path + '.hmma.tmp'
    try:
        with open(temp, 'wb') as output:
            output.write(module.data())
        result = run(
            [validator, '--target-env', 'vulkan1.3', temp],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise ValueError(
                (result.stderr or result.stdout or 'spirv-val failed')
                .strip()[:300]
            )
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
    return calls
