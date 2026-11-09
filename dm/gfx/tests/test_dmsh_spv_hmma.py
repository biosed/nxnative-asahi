#!/usr/bin/env python3
"""Current TensorPass HMMA ABI and the Vulkan call-site lowering."""

import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dmsh_chain
from dmsh_spv_hmma import (DERIVED_SUFFIXES, HMMA_NAME, Module, OP_ACCESS_CHAIN, OP_BITCAST,
                           OP_CAPABILITY, OP_COMPOSITE_EXTRACT, OP_COPY_OBJECT,
                           OP_FUNCTION_PARAMETER, OP_LOAD, _current_scratch_array,
                           _pointer_uses, _scratch_used_outside, functions_named,
                           hmma_helper, hmma_lds, lower_file, remove_outputs)


def half_word(a, b):
    return int.from_bytes(struct.pack('<ee', a, b), 'little')


def halves(word):
    return struct.unpack('<ee', word.to_bytes(4, 'little'))


def scalar_helper(a_words, b_words, c_words):
    """The translator helper's literal 256-word scratch implementation."""
    scratch = [0] * 256
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        a, b = a_words[lane], b_words[lane]
        scratch[8 * group + thread] = a[0]
        scratch[8 * (group + 8) + thread] = a[1]
        scratch[8 * group + thread + 4] = a[2]
        scratch[8 * (group + 8) + thread + 4] = a[3]
        scratch[128 + 8 * group + thread] = b[0]
        scratch[128 + 8 * group + thread + 4] = b[1]
    out = []
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        words = []
        for q in range(2):
            row = group + 8 * q
            acc = list(halves(c_words[lane][q]))
            for col in range(2):
                for kw in range(8):
                    av = halves(scratch[8 * row + kw])
                    bv = halves(scratch[128 + 8 * (2 * thread + col) + kw])
                    for half in range(2):
                        acc[col] = av[half] * bv[half] + acc[col]
            words.append(half_word(*acc))
        out.append(words)
    return out


def gathered_lowering(a_words, b_words, c_words):
    """The uvec4 gather layout emitted by dmsh_spv_hmma.hmma_lds."""
    vectors = [[0] * 4 for _ in range(64)]
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        vi, component = thread >> 1, 2 * (thread & 1)

        def store(row, word0, word1):
            vectors[row + vi][component:component + 2] = [word0, word1]

        store(2 * group, a_words[lane][0], a_words[lane][2])
        store(2 * (group + 8), a_words[lane][1], a_words[lane][3])
        store(32 + 2 * group, b_words[lane][0], b_words[lane][1])

    def load_row(row):
        packed = vectors[row] + vectors[row + 1]
        return [halves(packed[2 * (kw % 4) + kw // 4]) for kw in range(8)]

    out = []
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        a = [load_row(2 * group), load_row(2 * (group + 8))]
        b = [load_row(32 + 4 * thread), load_row(32 + 4 * thread + 2)]
        words = []
        for row in range(2):
            acc = list(halves(c_words[lane][row]))
            for col in range(2):
                for kw in range(8):
                    for half in range(2):
                        acc[col] = a[row][kw][half] * b[col][kw][half] + acc[col]
            words.append(half_word(*acc))
        out.append(words)
    return out


def fixture_assembly(call_count=1):
    params = '\n'.join('%%%s = OpFunctionParameter %%fn_int' % ('p%d' % i)
                       for i in range(18))
    variables = '\n'.join('%%%s = OpVariable %%fn_int Function' % ('v%d' % i)
                          for i in range(18))
    stores = '\n'.join('OpStore %%v%d %%zero_i' % i for i in range(18))
    arguments = ' '.join('%%v%d' % i for i in range(18))
    calls = '\n'.join(
        '%s = OpFunctionCall %%int %%hmma %s' %
        ('%call' if i == 0 else '%%call%d' % i, arguments)
        for i in range(call_count)
    )
    writes = '\n'.join(
        '%%store_ptr%d = OpAccessChain %%wg_uint %%scratch %%zero_u\n'
        'OpStore %%store_ptr%d %%zero_u' % (i, i) for i in range(8)
    )
    reads = '\n'.join(
        '%%load_ptr%d = OpAccessChain %%wg_uint %%scratch %%zero_u\n'
        '%%loaded%d = OpLoad %%uint %%load_ptr%d' % (i, i, i)
        for i in range(64)
    )
    param_types = ' '.join(['%fn_int'] * 18)
    return f'''OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %scratch
OpExecutionMode %main LocalSize 32 1 1
OpName %hmma "dmsh_hmma16816_f16"
%void = OpTypeVoid
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%zero_i = OpConstant %int 0
%zero_u = OpConstant %uint 0
%scope_subgroup = OpConstant %uint 3
%sem_workgroup = OpConstant %uint 264
%words = OpConstant %uint 256
%scratch_array = OpTypeArray %uint %words
%wg_array = OpTypePointer Workgroup %scratch_array
%wg_uint = OpTypePointer Workgroup %uint
%fn_int = OpTypePointer Function %int
%main_type = OpTypeFunction %void
%hmma_type = OpTypeFunction %int {param_types}
%scratch = OpVariable %wg_array Workgroup
%main = OpFunction %void None %main_type
%main_label = OpLabel
{variables}
{stores}
{calls}
OpReturn
OpFunctionEnd
%hmma = OpFunction %int None %hmma_type
{params}
%hmma_label = OpLabel
{writes}
OpControlBarrier %scope_subgroup %scope_subgroup %sem_workgroup
{reads}
OpControlBarrier %scope_subgroup %scope_subgroup %sem_workgroup
OpStore %p17 %zero_i
OpReturnValue %zero_i
OpFunctionEnd
'''


class HmmaSemantics(unittest.TestCase):
    def test_gather_layout_matches_scalar_helper_and_f16_rounding(self):
        rng = random.Random(0x16816)
        values = [-3.0, -1.5, -0.25, -0.0, 0.0999755859375,
                  0.5, 1.0, 1.75, 4.0]
        word = lambda: half_word(rng.choice(values), rng.choice(values))
        a = [[word() for _ in range(4)] for _ in range(32)]
        b = [[word() for _ in range(2)] for _ in range(32)]
        c = [[word() for _ in range(2)] for _ in range(32)]
        self.assertEqual(gathered_lowering(a, b, c), scalar_helper(a, b, c))

    def test_vulkan_sources_are_in_the_chain_identity(self):
        for name in ('tools/dmsh_spv.py', 'tools/dmsh_spv_hmma.py'):
            self.assertIn(name, dmsh_chain.FILES)

    def test_capability_literal_is_not_a_scratch_use(self):
        fake = type('FakeModule', (), {
            'ins': [[2 << 16 | OP_CAPABILITY, 46]],
        })()
        self.assertFalse(_scratch_used_outside(fake, 46, 1, 0))
        fake.ins.append([5 << 16 | OP_ACCESS_CHAIN, 2, 3, 46, 4])
        self.assertTrue(_scratch_used_outside(fake, 46, 1, 0))

    def test_failed_key_removes_source_and_derived_outputs(self):
        with tempfile.TemporaryDirectory(prefix='dmsh_hmma_cleanup_') as td:
            key = '0123456789abcdef'
            for suffix in DERIVED_SUFFIXES:
                (Path(td) / (key + suffix)).write_bytes(b'stale')
            remove_outputs(td, key)
            self.assertFalse(any((Path(td) / (key + suffix)).exists()
                                 for suffix in DERIVED_SUFFIXES))

    def test_current_scratch_shape_accepts_larger_whole_vectors(self):
        u32, array, length = 10, 11, 12
        module = type('FakeModule', (), {
            'types': {array: [4 << 16 | 28, array, u32, length]},
            'consts': {length: (u32, [256])},
        })()
        for words in (256, 512, 1024):
            module.consts[length] = (u32, [words])
            self.assertTrue(_current_scratch_array(module, array, u32))
        for words in (128, 258):
            module.consts[length] = (u32, [words])
            self.assertFalse(_current_scratch_array(module, array, u32))

    def test_fact_pointer_copy_and_bitcast_are_uses(self):
        pointer = 46
        for opcode in (OP_COPY_OBJECT, OP_BITCAST):
            instruction = [4 << 16 | opcode, 2, 3, pointer]
            self.assertEqual(_pointer_uses([instruction], pointer),
                             [(opcode, instruction)])


class HmmaSpirv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spirv_as = os.environ.get('SPIRV_AS') or shutil.which('spirv-as')
        cls.spirv_val = os.environ.get('SPIRV_VAL') or shutil.which('spirv-val')

    def test_current_abi_call_is_lowered_and_valid(self):
        if not self.spirv_as or not self.spirv_val:
            self.skipTest('spirv-as and spirv-val are needed for the SPIR-V fixture')
        with tempfile.TemporaryDirectory(prefix='dmsh_hmma_spv_') as td:
            assembly = Path(td) / 'hmma.spvasm'
            original = Path(td) / 'hmma.spv'
            lowered = Path(td) / 'hmma.lowered.spv'
            assembly.write_text(fixture_assembly())
            r = subprocess.run(
                [self.spirv_as, '--target-env', 'vulkan1.3',
                 str(assembly), '-o', str(original)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            module = Module(original.read_bytes())
            helper = hmma_helper(module)
            self.assertIsNotNone(helper)
            self.assertEqual(hmma_lds(module), 1)
            self.assertNotIn(HMMA_NAME, functions_named(module))
            lowered.write_bytes(module.data())
            r = subprocess.run(
                [self.spirv_val, '--target-env', 'vulkan1.3', str(lowered)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            # The seven TensorPass facts are unused by the reference helper.
            # If a future ABI consumes one, this conservative lowering refuses it.
            module = Module(original.read_bytes())
            fid = functions_named(module)[HMMA_NAME]
            first, last = module.function_range(fid)
            params = [x for x in module.ins[first:last]
                      if x[0] & 0xffff == OP_FUNCTION_PARAMETER]
            label = next(i for i in range(first, last)
                         if module.ins[i][0] & 0xffff == 248)
            result = module.new_id()
            module.ins.insert(label + 1, [4 << 16 | OP_LOAD,
                                          module.ins[first][1], result,
                                          params[10][2]])
            self.assertIsNone(hmma_helper(module))
            broken = Path(td) / 'broken.spv'
            broken.write_bytes(module.data())
            with self.assertRaisesRegex(ValueError, 'current ABI'):
                lower_file(broken)

            # The helper addresses exactly 256 words, while the translator's
            # module-wide private array may be larger.  Lowering must preserve
            # those larger byte sizes and still reject undersized declarations.
            for words in (512, 1024, 128, 258):
                module = Module(original.read_bytes())
                helper = hmma_helper(module)
                array = module.pointee(module.vars[helper['scratch']][0])[1]
                length_id = module.types[array][3]
                module.consts[length_id][1][0] = words
                const = next(x for x in module.ins
                             if x[0] & 0xffff == 43 and x[2] == length_id)
                const[3] = words
                if words in (512, 1024):
                    self.assertIsNotNone(hmma_helper(module))
                    self.assertEqual(hmma_lds(module), 1)
                    sized = Path(td) / ('hmma.%d.lowered.spv' % words)
                    sized.write_bytes(module.data())
                    r = subprocess.run(
                        [self.spirv_val, '--target-env', 'vulkan1.3',
                         str(sized)], capture_output=True, text=True,
                    )
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                else:
                    self.assertIsNone(hmma_helper(module))

            # Facts 10..16 are promised unused. Copies/casts are uses too: they
            # can feed a later load while hiding the original pointer operand.
            for opcode in (OP_COPY_OBJECT, OP_BITCAST):
                module = Module(original.read_bytes())
                fid = functions_named(module)[HMMA_NAME]
                first, last = module.function_range(fid)
                params = [x for x in module.ins[first:last]
                          if x[0] & 0xffff == OP_FUNCTION_PARAMETER]
                label = next(i for i in range(first, last)
                             if module.ins[i][0] & 0xffff == 248)
                result = module.new_id()
                module.ins.insert(label + 1, [4 << 16 | opcode,
                                              params[10][1], result,
                                              params[10][2]])
                self.assertIsNone(hmma_helper(module), opcode)

    def test_cached_fragments_reuse_loads_and_scalar_extracts(self):
        if not self.spirv_as or not self.spirv_val:
            self.skipTest('spirv-as and spirv-val are needed for the SPIR-V fixture')
        with tempfile.TemporaryDirectory(prefix='dmsh_hmma_reuse_') as td:
            assembly = Path(td) / 'hmma.spvasm'
            original = Path(td) / 'hmma.spv'
            lowered = Path(td) / 'hmma.lowered.spv'
            assembly.write_text(fixture_assembly(call_count=2))
            r = subprocess.run(
                [self.spirv_as, '--target-env', 'vulkan1.3',
                 str(assembly), '-o', str(original)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            module = Module(original.read_bytes())
            self.assertEqual(hmma_lds(module), 2)
            opcodes = [x[0] & 0xffff for x in module.ins]
            # First use gathers four rows (eight vector loads); the second
            # call reloads only its two accumulators.  Every unpacked half is
            # extracted once when gathered and then reused by both calls.
            self.assertEqual(opcodes.count(OP_LOAD), 20)
            self.assertEqual(opcodes.count(OP_COMPOSITE_EXTRACT), 104)
            lowered.write_bytes(module.data())
            r = subprocess.run(
                [self.spirv_val, '--target-env', 'vulkan1.3', str(lowered)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
