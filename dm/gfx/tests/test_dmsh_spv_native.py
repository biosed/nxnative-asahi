#!/usr/bin/env python3
import ast
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from dmsh_spv_native import (NativeError, load_mesh_facts, make_native_job,
                             nvn2_key, split_record)


def record(stage, verts=0, point_size=False):
    control = bytearray(0xd8)
    struct.pack_into('<II', control, 0, 0x98761234, 0xc9)
    struct.pack_into('<I', control, 0x30, len(control))
    struct.pack_into('<I', control, 0x38, 0x80)
    struct.pack_into('<II', control, 0x40, 0x100, 0x110)
    control[0x56] = {'ts': 6, 'ms': 7}[stage]
    control[0x5f] = verts
    struct.pack_into('<HH', control, 0x58, 32, 0x40)
    struct.pack_into('<HH', control, 0x60, 64, 32)
    code = bytearray(0x110)
    if stage == 'ms':
        code[0x35] = 0xf0 | (0x08 if point_size else 0)   # optional point size, then position x/y/z/w
        code[0x36] = 0x03                 # generic 0 x/y
    code[0x100:] = bytes(range(0x10))      # c1 after rounded instructions
    head = bytearray(b'BTCDBTCD' + bytes(0x30))
    struct.pack_into('<II', head, 0x28, len(control), len(code))
    return bytes(head + code + control), bytes(control), bytes(code)


class NativeJob(unittest.TestCase):
    def test_exact_task_and_mesh_jobs(self):
        with tempfile.TemporaryDirectory(prefix='dmsh_native_job_') as td:
            for stage in ('ts', 'ms'):
                rec, control, code = record(stage)
                key = nvn2_key(control, code)
                job = make_native_job(key, stage, rec, td)
                self.assertEqual(job['stage'], stage)
                self.assertEqual(job['local'], [32, 1, 1])
                self.assertEqual(job['c1'], bytes(range(0x10)).hex())
                self.assertEqual(Path(job['code']).read_bytes(), code[:0x80])
                if stage == 'ts':
                    self.assertEqual(job['mesh'], [0, 0, 0x40])
                else:
                    self.assertEqual(job['mesh'], [64, 32, 0x40, -5, 3, 0, 1, 2, 3, 4, 5, -1])

    def test_point_job_carries_topology_and_point_size_plane(self):
        rec, control, code = record('ms', verts=1, point_size=True)
        key = nvn2_key(control, code)
        with tempfile.TemporaryDirectory(prefix='dmsh_native_point_') as td:
            job = make_native_job(key, 'ms', rec, td)
            self.assertEqual(job['mesh'], [64, 32, 0x40, -5, 1, -4,
                                           0, 1, 2, 3, 4, 5, -1])

    def test_record_and_stage_mismatch_fail_closed(self):
        rec, control, code = record('ts')
        self.assertEqual(split_record(rec), (control, code))
        with tempfile.TemporaryDirectory(prefix='dmsh_native_bad_') as td:
            with self.assertRaises(NativeError):
                make_native_job('0' * 16, 'ts', rec, td)
            with self.assertRaises(NativeError):
                make_native_job(nvn2_key(control, code), 'ms', rec, td)
        with self.assertRaises(NativeError):
            split_record(b'bad')

    def test_unknown_topology_fails_closed(self):
        rec, control, code = record('ms', verts=4)
        with tempfile.TemporaryDirectory(prefix='dmsh_native_bad_topology_') as td:
            with self.assertRaises(NativeError):
                make_native_job(nvn2_key(control, code), 'ms', rec, td)

    def test_generated_fact_preserves_primitive_hole_and_padding(self):
        rec, control, code = record('ms')
        key = nvn2_key(control, code)
        fact = {'local': [32, 1, 1], 'shared': 0,
                'mesh': [64, 32, 0x40, -5, 3, 0, 1, 2, 3, 4, 5,
                         -1, -3, -2, 7], 'pair': ['0123456789abcdef']}
        with tempfile.TemporaryDirectory(prefix='dmsh_native_fact_') as td:
            job = make_native_job(key, 'ms', rec, td, fact)
            self.assertEqual(job['mesh'], fact['mesh'])
            facts_path = Path(td) / 'shader_facts.tsv'
            facts_path.write_text('# generated\n%s\t%s\n' %
                                  (key, __import__('json').dumps(fact)))
            self.assertEqual(load_mesh_facts(facts_path), {key: fact})
            no_primitive = dict(fact, mesh=[64, 32, 0x40, -5, 3,
                                            0, 1, 2, 3, 4, 5, -2, 7])
            self.assertEqual(make_native_job(key, 'ms', rec, td,
                                             no_primitive)['mesh'],
                             no_primitive['mesh'])
            bad = dict(fact, local=[16, 1, 1])
            with self.assertRaises(NativeError):
                make_native_job(key, 'ms', rec, td, bad)
            bad_topology = dict(fact, mesh=[64, 32, 0x40, -5, 1,
                                            0, 1, 2, 3, 4, 5, -1, -3])
            with self.assertRaises(NativeError):
                make_native_job(key, 'ms', rec, td, bad_topology)


class ShaderSplitFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((TOOLS / 'shader_split.py').read_text())
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'mesh_facts')
        scope = {'struct': struct}
        exec(compile(ast.Module(body=[node], type_ignores=[]),
                     'shader_split.py', 'exec'), scope)
        cls.mesh_facts = staticmethod(scope['mesh_facts'])

    def test_point_topology_and_size_are_generated(self):
        ctl = bytearray(0x80)
        struct.pack_into('<HH', ctl, 0x58, 32, 128)
        ctl[0x5f] = 1
        struct.pack_into('<HH', ctl, 0x60, 32, 32)
        code = bytearray(0x80)
        code[0x35] = 0xf8
        fact = self.mesh_facts('point', 'ms', ctl, code, {}, set())
        self.assertEqual(fact['mesh'], [32, 32, 128, -5, 1, -4,
                                        0, 1, 2, 3])

    def test_unknown_topology_fails_closed(self):
        ctl = bytearray(0x80)
        struct.pack_into('<HH', ctl, 0x58, 32, 128)
        ctl[0x5f] = 4
        struct.pack_into('<HH', ctl, 0x60, 32, 32)
        fact = self.mesh_facts('bad', 'ms', ctl, bytearray(0x80), {}, set())
        self.assertNotIn('mesh', fact)


if __name__ == '__main__':
    unittest.main()
