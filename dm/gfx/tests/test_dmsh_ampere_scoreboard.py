#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Swap2-Native contributors
# Adapted from https://github.com/mamcjr/Swap2-Native/blob/main/dm/gfx/tests/test_dmsh_ampere_scoreboard.py
"""Deferred register writes under a guard (dmsh_xlate, arch sm86): a variable-latency instruction's result goes to a shadow register that is copied to its register at the
first wait on its scoreboard or at the end of its block (Translator.EmitOpsAmpere). Under a guard the instruction runs on some lanes only, and every copy must still run on
every lane (docs/reference/shaders.md#deferred-register-writes-under-a-guard-ampere-scoreboards). Zelda: Ocarina of Time's glyph quads lost their corners' y that way
(b19e1bdd247555ec: "@!P0 ALD R85" then "@P0 FFMA R85" at the end of its block), and so drew no text. Three hand-encoded compute programs, each run for both values of the
guard: a register written by a fixed-latency instruction while its deferred write is pending, a register written by two deferred loads under opposite guards, and a guarded
instruction at the end of a block while a deferred write is pending. The checks run the main function of the SPIR-V on one lane with a small interpreter (the operations the
translator makes of these programs: function variables, workgroup memory, structured branches), so they need no GPU. Needs DMSH_TEST_XLATE (the translator) like the other
translator tests; with SPIRV_VAL (default spirv-val on PATH) every module must also pass spirv-val --target-env vulkan1.3."""
import json, os, shutil, struct, subprocess, tempfile, unittest
from pathlib import Path

VAL = os.environ.get('SPIRV_VAL') or shutil.which('spirv-val')
PT = 7


def inst(hi, lo): return struct.pack('<QQ', lo, hi)


def ctl(wb=7, wait=0):
    """Control bits (105..125, the high word's 41..61): stall 1, write barrier wb (7 = none), no read barrier, wait mask"""
    return 1 << 41 | wb << 46 | 7 << 49 | wait << 52


def guard(pred, inv): return (pred | 8 * inv) << 12   # bits 12..15: @P / @!P


def mov(rd, value, pred=PT, inv=False): return inst(ctl() | 0xf00, value << 32 | rd << 16 | guard(pred, inv) | 0x802)   # MOV Rd, imm32
def lds(rd, offset, pred=PT, inv=False, wb=7): return inst(ctl(wb) | 0x800, offset << 40 | 0xff << 24 | rd << 16 | guard(pred, inv) | 0x984)   # LDS.U32 Rd, [RZ + offset]
def sts(offset, rb, wait=0): return inst(ctl(wait=wait) | 0x800, offset << 40 | rb << 32 | 0xff << 24 | 0x7388)   # STS.U32 [RZ + offset], Rb
def isetp_ge(pd, ra, imm): return inst(ctl() | 0x3f06270 | pd << 17, imm << 32 | ra << 24 | 0x780c)   # ISETP.GE.AND Pd, PT, Ra, imm32, PT
def bra(words, pred=PT, inv=False): return inst(ctl() | 0x3800000, words << 34 | guard(pred, inv) | 0x947)   # BRA: words (4 bytes) after the next instruction


EXIT = inst(0x000fea0003800000, 0x000000000000794d)
SHARED_BASE = 0x5000   # workgroup word i starts as SHARED_BASE + i


def prologue(p0): return mov(0, p0) + isetp_ge(0, 0, 1)   # P0 = (R0 >= 1), R0 = p0


# SPIR-V opcodes the interpreter knows
OP_NAME = 5
OP_ENTRY_POINT, OP_CONSTANT_TRUE, OP_CONSTANT_FALSE, OP_CONSTANT, OP_FUNCTION, OP_FUNCTION_END, OP_VARIABLE, OP_LOAD, OP_STORE = 15, 41, 42, 43, 54, 56, 59, 61, 62
OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_BITCAST, OP_COPY_OBJECT, OP_IADD = 65, 66, 124, 83, 128
OP_LOGICAL_EQUAL, OP_LOGICAL_NOT_EQUAL, OP_LOGICAL_OR, OP_LOGICAL_AND, OP_LOGICAL_NOT, OP_SELECT = 164, 165, 166, 167, 168, 169
OP_IEQUAL, OP_INOT_EQUAL = 170, 171
OP_PHI, OP_LOOP_MERGE, OP_SELECTION_MERGE, OP_LABEL, OP_BRANCH, OP_BRANCH_CONDITIONAL, OP_RETURN = 245, 246, 247, 248, 249, 250, 253
OP_CONTROL_BARRIER, OP_MEMORY_BARRIER = 224, 225
OP_TYPES = range(19, 40)   # OpTypeVoid .. OpTypeForwardPointer
OP_DECORATIONS = (71, 72, 73, 74, 75, 332)
WORKGROUP, FUNCTION = 4, 7


def s32(x): return x - (1 << 32) if x & 1 << 31 else x


COMPARE = {170: lambda a, b: a == b, 171: lambda a, b: a != b, 172: lambda a, b: a > b, 173: lambda a, b: s32(a) > s32(b), 174: lambda a, b: a >= b,
           175: lambda a, b: s32(a) >= s32(b), 176: lambda a, b: a < b, 177: lambda a, b: s32(a) < s32(b), 178: lambda a, b: a <= b, 179: lambda a, b: s32(a) <= s32(b)}


def run(spv):
    """Runs the entry point of a compute module on one lane: → {workgroup word index: value} written. Every Workgroup word reads SHARED_BASE + its index until written;
    an operation the interpreter does not know fails the test"""
    w = struct.unpack(f'<{len(spv) // 4}I', spv)
    insts, i = [], 5
    while i < len(w):
        n = w[i] >> 16
        insts.append((w[i] & 0xffff, list(w[i + 1:i + n])))
        i += n
    vals, storage, entry, funcs = {}, {}, None, {}
    for k, (op, a) in enumerate(insts):
        if op == OP_ENTRY_POINT: entry = a[1]
        elif op == OP_CONSTANT: vals[a[1]] = a[2]
        elif op in (OP_CONSTANT_TRUE, OP_CONSTANT_FALSE): vals[a[1]] = op == OP_CONSTANT_TRUE
        elif op == OP_VARIABLE: storage[a[1]] = a[2]
        elif op == OP_FUNCTION: funcs[a[1]] = k
    body, labels, k = [], {}, funcs[entry] + 1
    while insts[k][0] != OP_FUNCTION_END:
        if insts[k][0] == OP_LABEL: labels[insts[k][1][0]] = len(body)
        body.append(insts[k]); k += 1
    mem, shared, written = {}, {}, {}   # function variables (by id), workgroup words, words written

    def load(ptr):
        if isinstance(ptr, tuple): return shared.get(ptr[1], SHARED_BASE + ptr[1])
        return mem.get(ptr, 0)

    pc, prev, cur, steps = 0, None, None, 0
    while True:
        steps += 1
        if steps > 10000: raise AssertionError('the main function does not return')
        op, a = body[pc]; pc += 1
        if op == OP_LABEL: prev, cur = cur, a[0]
        elif op == OP_VARIABLE: mem[a[1]] = vals[a[3]] if len(a) > 3 else 0
        elif op in (OP_SELECTION_MERGE, OP_LOOP_MERGE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, OP_NAME) or op in OP_DECORATIONS: pass
        elif op == OP_LOAD: vals[a[1]] = load(vals.get(a[2], a[2]))
        elif op == OP_STORE:
            ptr, v = vals.get(a[0], a[0]), vals[a[1]]
            if isinstance(ptr, tuple): shared[ptr[1]] = written[ptr[1]] = v
            else: mem[ptr] = v
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN):
            if storage.get(a[2]) != WORKGROUP or len(a) != 4: raise AssertionError(f'access chain {a} into storage class {storage.get(a[2])}')
            vals[a[1]] = ('wg', vals[a[3]])
        elif op in (OP_BITCAST, OP_COPY_OBJECT): vals[a[1]] = vals[a[2]]
        elif op == OP_IADD: vals[a[1]] = (vals[a[2]] + vals[a[3]]) & 0xffffffff
        elif op in COMPARE: vals[a[1]] = COMPARE[op](vals[a[2]], vals[a[3]])
        elif op == OP_LOGICAL_NOT: vals[a[1]] = not vals[a[2]]
        elif op == OP_LOGICAL_AND: vals[a[1]] = vals[a[2]] and vals[a[3]]
        elif op == OP_LOGICAL_OR: vals[a[1]] = vals[a[2]] or vals[a[3]]
        elif op == OP_LOGICAL_EQUAL: vals[a[1]] = vals[a[2]] == vals[a[3]]
        elif op == OP_LOGICAL_NOT_EQUAL: vals[a[1]] = vals[a[2]] != vals[a[3]]
        elif op == OP_SELECT: vals[a[1]] = vals[a[3]] if vals[a[2]] else vals[a[4]]
        elif op == OP_PHI: vals[a[1]] = next(vals[a[j]] for j in range(2, len(a), 2) if a[j + 1] == prev)
        elif op == OP_BRANCH: pc = labels[a[0]]
        elif op == OP_BRANCH_CONDITIONAL: pc = labels[a[1] if vals[a[0]] else a[2]]
        elif op == OP_RETURN: return written
        else: raise AssertionError(f'the interpreter does not know SPIR-V opcode {op} ({a})')


@unittest.skipUnless(os.environ.get('DMSH_TEST_XLATE'), 'set DMSH_TEST_XLATE to the offline translator')
class GuardedDeferredWrites(unittest.TestCase):
    def translate(self, code):
        d = tempfile.mkdtemp(prefix='ampere_sb_'); self.addCleanup(shutil.rmtree, d, True)
        work = Path(d); (work / 'k.bin').write_bytes(code + EXIT)
        # c0 layout (QueryAmpereComputeC0Layout): no uniform or storage buffers, 3 texture handles from 0x10, image handles from 0x28, 0x40 bytes
        job = [dict(key='k', stage='cs', arch='sm86', code=str(work / 'k.bin'), tex={}, img={}, attr={}, fout={}, local=[32, 1, 1], debug=True, nvn2_cs=[0, 0, 3, 0x40, 0],
                    shared=0x400)]
        (work / 'jobs.json').write_text(json.dumps(job))
        r = subprocess.run([os.environ['DMSH_TEST_XLATE'], str(work / 'jobs.json'), d], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((work / 'k.spv').exists(), r.stdout + r.stderr)
        if VAL:
            v = subprocess.run([VAL, '--target-env', 'vulkan1.3', str(work / 'k.spv')], capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        return run((work / 'k.spv').read_bytes())

    def test_fixed_latency_write_while_pending(self):
        # "@!P0 LDS R5" (scoreboard 0) then "@P0 MOV R5" before the wait (b19e1bdd247555ec's "@!P0 ALD R85" / "@P0 FFMA R85"): the MOV's lanes keep its value; the
        # pending copy at the wait used to overwrite it with the shadow, which on those lanes held R5's old value (and the MOV was removed as dead)
        for p0, want in ((1, 0x12345), (0, SHARED_BASE + 0x40)):
            with self.subTest(p0=p0):
                m = self.translate(prologue(p0) + mov(5, 0x77) + lds(5, 0x100, 0, True, wb=0) + mov(5, 0x12345, 0) + sts(0x300, 5, wait=1))
                self.assertEqual(m, {0xc0: want})

    def test_two_deferred_writes_under_opposite_guards(self):
        # "@P0 LDS R6" (scoreboard 0) then "@!P0 LDS R6" (scoreboard 1), one wait on both (9b310b3349825e9a's "@P1 TEX R6" / "@!P1 TEX R6"): each load's lanes keep its
        # word; the second deferred write used to drop the first, so P0's lanes kept R6's old value
        for p0, want in ((1, SHARED_BASE + 0x40), (0, SHARED_BASE + 0x80)):
            with self.subTest(p0=p0):
                m = self.translate(prologue(p0) + mov(6, 0x66) + lds(6, 0x100, 0, False, wb=0) + lds(6, 0x200, 0, True, wb=1) + sts(0x300, 6, wait=3))
                self.assertEqual(m, {0xc0: want})

    def test_guarded_end_of_block_while_pending(self):
        # "@!P0 LDS R5" (scoreboard 0) then "@P0 MOV R6", the last instruction of its block (an untaken "@P3 BRA" makes the next instruction a branch target): the lanes
        # that skip the MOV used to jump straight to the next block, past the copy of R5 at the end of the block, and kept R5's old value
        for p0, r5, r6 in ((0, SHARED_BASE + 0x40, 0x66), (1, 0x55, 0x777)):
            with self.subTest(p0=p0):
                m = self.translate(prologue(p0) + isetp_ge(3, 0, 100) + mov(5, 0x55) + mov(6, 0x66) + bra(2 * 4, 3) +   # @P3 BRA over the next two instructions
                                   lds(5, 0x100, 0, True, wb=0) + mov(6, 0x777, 0) + sts(0x300, 5, wait=1) + sts(0x304, 6))
                self.assertEqual(m, {0xc0: r5, 0xc1: r6})

if __name__ == '__main__':
    unittest.main()
