#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, '..', 'tools'))
sys.path.insert(0, TOOLS)

from dmsh_spv_mesh_first import (MeshFirstError, OP_DECORATE, _decode,
                                 _encode, transform_mesh_first)
from dmsh_spv_bindings import MESH_FIRST_BINDING

ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', '..'))
SPIRV_VAL = os.path.join(ROOT, '.deps', 'spirv-tools-cli-root', 'usr', 'bin', 'spirv-val')
GLSLC = '/usr/bin/glslc'


class MeshFirst(unittest.TestCase):
    def compile(self, stage, body):
        prologue = '''#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x=1) in;
'''
        if stage == 'vertex':
            prologue = '#version 460\n'
        if stage == 'mesh':
            prologue += 'layout(triangles,max_vertices=3,max_primitives=1) out;\n'
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'fixture.' + stage)
            out = src + '.spv'
            with open(src, 'w') as f:
                f.write(prologue + 'void main(){' + body + '}\n')
            subprocess.run([GLSLC, '--target-env=vulkan1.3', '-O0',
                            '-fshader-stage=' + stage, src, '-o', out], check=True)
            with open(out, 'rb') as f:
                return f.read()

    def validate(self, data):
        with tempfile.NamedTemporaryFile(suffix='.spv') as f:
            f.write(data); f.flush()
            subprocess.run([SPIRV_VAL, '--target-env', 'vulkan1.3', f.name], check=True)

    def test_component_and_vector_forms(self):
        component = self.compile('task',
            'uint x=gl_WorkGroupID.x; EmitMeshTasksEXT((x&0u)+1u,1u,1u);')
        patched, report = transform_mesh_first(component, 'ts', False, 3)
        self.assertGreater(report['component_loads'], 0)
        self.assertEqual((report['semantic_adds'], report['binding'], report['descriptor_set']),
                         (1, MESH_FIRST_BINDING, 3))
        self.validate(patched)

        vector = self.compile('mesh',
            'uvec3 v=gl_WorkGroupID; SetMeshOutputsEXT(0u,(v.x&0u));')
        patched, report = transform_mesh_first(vector, 'ms', False, 4)
        self.assertGreater(report['vector_loads'], 0)
        self.assertEqual(report['semantic_adds'], 1)
        self.validate(patched)

    def test_multiple_recognized_loads_share_one_add(self):
        original = self.compile('task',
            'uvec3 v=gl_WorkGroupID; uint x=gl_WorkGroupID.x; '
            'EmitMeshTasksEXT(((v.x+x)&0u)+1u,1u,1u);')
        patched, report = transform_mesh_first(original, 'ts', False, 3)
        self.assertGreaterEqual(report['rewritten_loads'], 2)
        self.assertEqual(report['semantic_adds'], 1)
        self.validate(patched)

    def test_paired_mesh_nonmesh_and_owner_no_use_untouched(self):
        mesh = self.compile('mesh',
            'uint x=gl_WorkGroupID.x; SetMeshOutputsEXT(0u,x&0u);')
        paired, report = transform_mesh_first(mesh, 'ms', True, 4)
        self.assertEqual(paired, mesh)
        self.assertFalse(report['owner'])
        self.assertIsNone(report['binding'])

        task = self.compile('task', 'EmitMeshTasksEXT(1u,1u,1u);')
        no_use, report = transform_mesh_first(task, 'ts', False, 3)
        self.assertEqual(no_use, task)
        self.assertTrue(report['owner_no_use'])
        self.assertIsNone(report['binding'])

        vertex = self.compile('vertex', 'gl_Position=vec4(0.0);')
        untouched, report = transform_mesh_first(vertex)
        self.assertEqual(untouched, vertex)
        self.assertFalse(report['native'])

    def test_dynamic_multiple_and_collision_fail_closed(self):
        dynamic = self.compile('task',
            'uint i=gl_LocalInvocationIndex&2u; uint x=gl_WorkGroupID[i]; '
            'EmitMeshTasksEXT((x&0u)+1u,1u,1u);')
        with self.assertRaisesRegex(MeshFirstError, 'constant x'):
            transform_mesh_first(dynamic, 'ts', False, 3)

        original = self.compile('task',
            'uint x=gl_WorkGroupID.x; EmitMeshTasksEXT((x&0u)+1u,1u,1u);')
        patched, _ = transform_mesh_first(original, 'ts', False, 3)
        with self.assertRaisesRegex(MeshFirstError, 'binding %d already exists' % MESH_FIRST_BINDING):
            transform_mesh_first(patched, 'ts', False, 3)

        header, ins = _decode(original)
        builtin = next((i for i in ins if i[0] == OP_DECORATE and len(i[1]) == 3 and
                        i[1][1:] == [11, 26]), None)
        self.assertIsNotNone(builtin)
        at = ins.index(builtin)
        ins.insert(at + 1, [builtin[0], list(builtin[1])])
        with self.assertRaisesRegex(MeshFirstError, 'at most one WorkgroupId'):
            transform_mesh_first(_encode(header, ins), 'ts', False, 3)


if __name__ == '__main__':
    unittest.main()
