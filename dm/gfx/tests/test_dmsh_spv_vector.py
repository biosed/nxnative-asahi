#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from dmsh_spv_vector import (OP_ATOMIC_FADD_EXT, OP_ATOMIC_FLAG_CLEAR,
                             OP_ATOMIC_FLAG_TEST_AND_SET, OP_COMPOSITE_EXTRACT,
                             OP_COOPERATIVE_MATRIX_LOAD_NV,
                             OP_COOPERATIVE_MATRIX_STORE_NV,
                             OP_GROUP_ASYNC_COPY, OP_GROUP_WAIT_EVENTS,
                             OP_IMAGE_SPARSE_FIRST, OP_LOAD,
                             OP_MEMORY_NAMED_BARRIER, OP_SELECT,
                             OP_TYPE_POINTER, OP_TYPE_VECTOR, SC_STORAGE_BUFFER,
                             _decode, _encode, _entrypoint_aliases, _memory_fence,
                             _segment_safe,
                             _written_roots, materialize_nonwritable,
                             optimize_vector_loads, transform_vector_loads)
import dmsh_chain


def tool(name):
    root = os.environ.get('DMSH_FIXTURE_TOOLS')
    return shutil.which(name) or (os.path.join(root, name) if root else None)


SPIRV_AS, SPIRV_VAL, SPIRV_DIS = (tool(n) for n in ('spirv-as', 'spirv-val', 'spirv-dis'))

PROLOGUE = '''
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %buf %out
OpExecutionMode %main LocalSize 1 1 1
OpName %buf "cp_s0"
OpName %out "sink"
OpDecorate %ra ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 20
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 21
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ra = OpTypeRuntimeArray %u32
%block = OpTypeStruct %ra
%pblock = OpTypePointer StorageBuffer %block
%pu32 = OpTypePointer StorageBuffer %u32
%pfn_u32 = OpTypePointer Function %u32
%mutator_fn = OpTypeFunction %void %pfn_u32
%buf = OpVariable %pblock StorageBuffer
%out = OpVariable %pblock StorageBuffer
%zero = OpConstant %u32 0
%one = OpConstant %u32 1
%two = OpConstant %u32 2
%three = OpConstant %u32 3
%four = OpConstant %u32 4
%main = OpFunction %void None %fn
%label = OpLabel
'''
EPILOGUE = 'OpReturn\nOpFunctionEnd\n'
REFLECTION = {'bindings': [{'kind': 'sbuf', 'rw': False, 'name': 'cp_s0'}]}


class VectorSafetyLogic(unittest.TestCase):
    def test_vector_postprocessor_is_in_chain_identity(self):
        self.assertIn('tools/dmsh_spv_vector.py', dmsh_chain.FILES)

    def test_pointer_merge_and_modern_atomic_are_a_write_conflict(self):
        root, selected = 10, 11
        ins = [
            [OP_SELECT, [90, selected, 3, root, root]],
            [OP_ATOMIC_FLAG_CLEAR, [selected, 4, 5]],
        ]
        self.assertEqual(_written_roots(ins, {root: (1, 12)}), {root})

    def test_every_known_extended_memory_operation_closes_a_run(self):
        for opcode in (OP_ATOMIC_FLAG_TEST_AND_SET, OP_ATOMIC_FLAG_CLEAR,
                       OP_ATOMIC_FADD_EXT, OP_GROUP_ASYNC_COPY,
                       OP_GROUP_WAIT_EVENTS, OP_IMAGE_SPARSE_FIRST,
                       OP_MEMORY_NAMED_BARRIER,
                       OP_COOPERATIVE_MATRIX_LOAD_NV,
                       OP_COOPERATIVE_MATRIX_STORE_NV):
            self.assertTrue(_memory_fence(opcode), opcode)
        for opcode in (4457, 4458, 9999):   # CooperativeMatrix KHR + future op.
            self.assertFalse(_segment_safe(opcode), opcode)
        self.assertTrue(_segment_safe(OP_LOAD, [1, 2, 3]))
        self.assertFalse(_segment_safe(OP_LOAD, [1, 2, 3, 4]))

    def test_entrypoint_globals_follow_spirv_version(self):
        self.assertEqual(_entrypoint_aliases(0x00010300, (7, 8)), [])
        self.assertEqual(_entrypoint_aliases(0x00010400, (7, 8)), [7, 8])


@unittest.skipUnless(SPIRV_AS and SPIRV_VAL and SPIRV_DIS,
                     'SPIRV-Tools are needed for vector-load fixtures')
class VectorLoads(unittest.TestCase):
    def assemble(self, body, prologue=PROLOGUE, suffix=EPILOGUE):
        with tempfile.TemporaryDirectory(prefix='dmsh_spv_vec_') as td:
            asm, spv = Path(td) / 'fixture.spvasm', Path(td) / 'fixture.spv'
            asm.write_text(prologue + body + suffix)
            r = subprocess.run([SPIRV_AS, '--target-env', 'vulkan1.3', str(asm), '-o', str(spv)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return spv.read_bytes()

    def validate_disassemble(self, data):
        with tempfile.TemporaryDirectory(prefix='dmsh_spv_vec_out_') as td:
            spv = Path(td) / 'out.spv'; spv.write_bytes(data)
            r = subprocess.run([SPIRV_VAL, '--target-env', 'vulkan1.3', str(spv)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return subprocess.run([SPIRV_DIS, str(spv)], check=True,
                                  capture_output=True, text=True).stdout

    def test_aligned_pair_and_quad_validate_with_same_binding_aliases(self):
        data = self.assemble('''
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p2 = OpAccessChain %pu32 %buf %zero %two
%v2 = OpLoad %u32 %p2
%p3 = OpAccessChain %pu32 %buf %zero %three
%v3 = OpLoad %u32 %p3
''')
        patched, report = optimize_vector_loads(data, REFLECTION)
        self.assertEqual((report['groups2'], report['groups4'], report['scalar_loads']),
                         (0, 1, 4))
        text = self.validate_disassemble(patched)
        self.assertIn('"cp_s0_v4"', text)
        self.assertEqual(text.count('OpDecorate %cp_s0 NonWritable'), 1)
        self.assertEqual(text.count('Binding 20'), 2)
        self.assertEqual(text.count('OpCompositeExtract'), 4)

    def test_predeclared_vector_and_pointer_types_are_interned(self):
        prologue = PROLOGUE.replace(
            '%u32 = OpTypeInt 32 0',
            '%u32 = OpTypeInt 32 0\n%pre_v2 = OpTypeVector %u32 2\n%pre_v4 = OpTypeVector %u32 4').replace(
            '%pu32 = OpTypePointer StorageBuffer %u32',
            '%pu32 = OpTypePointer StorageBuffer %u32\n%pre_pv2 = OpTypePointer StorageBuffer %pre_v2\n%pre_pv4 = OpTypePointer StorageBuffer %pre_v4')
        data = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
%p2 = OpAccessChain %pu32 %buf %zero %two
%v2 = OpLoad %u32 %p2
%p3 = OpAccessChain %pu32 %buf %zero %three
%v3 = OpLoad %u32 %p3
%p4 = OpAccessChain %pu32 %buf %zero %zero
%v4 = OpLoad %u32 %p4
%p5 = OpAccessChain %pu32 %buf %zero %one
%v5 = OpLoad %u32 %p5
''', prologue=prologue)
        patched, report = optimize_vector_loads(data, REFLECTION)
        self.assertEqual((report['groups2'], report['groups4']), (1, 1))
        self.validate_disassemble(patched)
        _, instructions = _decode(patched)
        for width in (2, 4):
            vector_ids = [a[0] for op, a in instructions
                          if op == OP_TYPE_VECTOR and a[2] == width]
            self.assertEqual(len(vector_ids), 1)
            pointer_ids = [a[0] for op, a in instructions
                           if op == OP_TYPE_POINTER and
                           a[1:] == [SC_STORAGE_BUFFER, vector_ids[0]]]
            self.assertEqual(len(pointer_ids), 1)

    def test_reverse_pair_keeps_original_result_ids(self):
        data = self.assemble('''
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
''')
        _, before = _decode(data)
        scalar_results = {a[1] for op, a in before if op == OP_LOAD}
        patched, report = optimize_vector_loads(data, REFLECTION)
        self.assertEqual((report['groups2'], report['scalar_loads']), (1, 2))
        self.validate_disassemble(patched)
        # The two scalar load result IDs survive as extract result IDs.
        _, after = _decode(patched)
        extracts = {a[1] for op, a in after if op == OP_COMPOSITE_EXTRACT}
        self.assertEqual(extracts, scalar_results)

    def test_store_or_aliasing_memory_read_stops_a_run(self):
        stored = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%po = OpAccessChain %pu32 %out %zero %zero
OpStore %po %v0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
''')
        unchanged, report = optimize_vector_loads(stored, REFLECTION)
        self.assertEqual(report['scalar_loads'], 0)
        self.assertEqual(report['reflection_write_conflicts'], 0)  # the write is through another binding

        other_read = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%po = OpAccessChain %pu32 %out %zero %zero
%vo = OpLoad %u32 %po
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
''')
        unchanged, report = optimize_vector_loads(other_read, REFLECTION)
        self.assertEqual(report['scalar_loads'], 0)

    def test_alignment_and_reflection_fail_closed(self):
        misaligned = self.assemble('''
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
%p2 = OpAccessChain %pu32 %buf %zero %two
%v2 = OpLoad %u32 %p2
''')
        # The coalescer itself requires an explicit source NonWritable fact.
        unchanged, report = transform_vector_loads(misaligned, REFLECTION)
        self.assertEqual(report['rejected']['nonwritable'], 1)
        self.assertEqual(unchanged, misaligned)
        marked, facts = materialize_nonwritable(misaligned, REFLECTION)
        self.assertEqual(facts['materialized_nonwritable'], 1)
        unchanged, report = transform_vector_loads(marked, REFLECTION)
        self.assertEqual(report['scalar_loads'], 0)

        writable = {'bindings': [{'kind': 'sbuf', 'rw': True, 'name': 'cp_s0'}]}
        unchanged, report = optimize_vector_loads(misaligned, writable)
        self.assertEqual(report['readonly_variables'], 0)
        self.assertEqual(unchanged, misaligned)

        stale = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
OpStore %p0 %v0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
''')
        unchanged, report = optimize_vector_loads(stale, REFLECTION)
        self.assertEqual(report['reflection_write_conflicts'], 1)
        self.assertEqual(report['scalar_loads'], 0)
        self.assertEqual(unchanged, stale)

    def test_indirect_local_store_invalidates_affine_alignment(self):
        data = self.assemble('''
%local = OpVariable %pfn_u32 Function
OpStore %local %zero
%local_alias = OpCopyObject %pfn_u32 %local
OpStore %local_alias %one
%index = OpLoad %u32 %local
%index1 = OpIAdd %u32 %index %one
%p0 = OpAccessChain %pu32 %buf %zero %index
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %index1
%v1 = OpLoad %u32 %p1
''')
        unchanged, report = optimize_vector_loads(data, REFLECTION)
        self.assertEqual(report['scalar_loads'], 0)
        self.assertEqual(unchanged, data)

    def test_pointer_passing_call_invalidates_affine_alignment(self):
        suffix = EPILOGUE + '''
%mutator = OpFunction %void None %mutator_fn
%target = OpFunctionParameter %pfn_u32
%mutator_label = OpLabel
OpStore %target %one
OpReturn
OpFunctionEnd
'''
        data = self.assemble('''
%local = OpVariable %pfn_u32 Function
OpStore %local %zero
%call = OpFunctionCall %void %mutator %local
%index = OpLoad %u32 %local
%index1 = OpIAdd %u32 %index %one
%p0 = OpAccessChain %pu32 %buf %zero %index
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %index1
%v1 = OpLoad %u32 %p1
''', suffix=suffix)
        unchanged, report = optimize_vector_loads(data, REFLECTION)
        self.assertEqual(report['scalar_loads'], 0)
        self.assertEqual(unchanged, data)

    def test_member_memory_semantics_and_restrict_reject_alias(self):
        body = '''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
'''
        coherent = PROLOGUE.replace(
            'OpMemberDecorate %block 0 Offset 0',
            'OpMemberDecorate %block 0 Offset 0\nOpMemberDecorate %block 0 Coherent')
        unchanged, report = optimize_vector_loads(self.assemble(body, coherent), REFLECTION)
        self.assertEqual(report['rejected']['volatile'], 1)
        self.assertEqual(report['scalar_loads'], 0)

        restricted = PROLOGUE.replace(
            'OpDecorate %buf Binding 20',
            'OpDecorate %buf Binding 20\nOpDecorate %buf Restrict')
        unchanged, report = optimize_vector_loads(self.assemble(body, restricted), REFLECTION)
        self.assertEqual(report['rejected']['aliasing'], 1)
        self.assertEqual(report['scalar_loads'], 0)

    def test_same_binding_variable_rejects_unproved_alias(self):
        body = '''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
'''
        duplicate = PROLOGUE.replace(
            'OpName %buf "cp_s0"',
            'OpName %buf "cp_s0"\nOpName %buf2 "other"').replace(
            'OpDecorate %buf Binding 20',
            'OpDecorate %buf Binding 20\nOpDecorate %buf2 DescriptorSet 0\nOpDecorate %buf2 Binding 20').replace(
            '%buf = OpVariable %pblock StorageBuffer',
            '%buf = OpVariable %pblock StorageBuffer\n%buf2 = OpVariable %pblock StorageBuffer')
        unchanged, report = optimize_vector_loads(self.assemble(body, duplicate), REFLECTION)
        self.assertEqual(report['rejected']['aliasing'], 1)
        self.assertEqual(report['scalar_loads'], 0)

    def test_entrypoint_interface_matches_module_version(self):
        data = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
''')
        header, original = _decode(data)
        for version, added in ((0x00010300, 0), (0x00010400, 1)):
            source_header = list(header); source_header[1] = version
            source_instructions = [[op, list(a)] for op, a in original]
            entry = next(a for op, a in source_instructions if op == 15)
            if version < 0x00010400:
                del entry[-2:]   # Pre-1.4 interfaces contain Input/Output only.
            before = len(entry)
            source = _encode(source_header, source_instructions)
            patched, report = optimize_vector_loads(source, REFLECTION)
            self.assertEqual(report['scalar_loads'], 2)
            _, instructions = _decode(patched)
            after = next(a for op, a in instructions if op == 15)
            self.assertEqual(len(after), before + added)
            self.validate_disassemble(patched)

    def test_nonwritable_is_published_only_for_transformed_roots(self):
        data = self.assemble('''
%p0 = OpAccessChain %pu32 %buf %zero %zero
%v0 = OpLoad %u32 %p0
%p1 = OpAccessChain %pu32 %buf %zero %one
%v1 = OpLoad %u32 %p1
''')
        reflection = {'bindings': [
            {'kind': 'sbuf', 'rw': False, 'name': 'cp_s0'},
            {'kind': 'sbuf', 'rw': False, 'name': 'sink'},
        ]}
        patched, report = optimize_vector_loads(data, reflection)
        self.assertEqual(report['scalar_loads'], 2)
        self.assertEqual(report['materialized_nonwritable'], 1)
        text = self.validate_disassemble(patched)
        self.assertEqual(text.count('OpDecorate %cp_s0 NonWritable'), 1)
        self.assertNotIn('OpDecorate %sink NonWritable', text)


if __name__ == '__main__':
    unittest.main()
