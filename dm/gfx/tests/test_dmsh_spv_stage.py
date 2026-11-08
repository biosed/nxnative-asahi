#!/usr/bin/env python3
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from dmsh_spv_stage import StageError, STAGE_SET, emit_loader_pair, import_native_translation, module_stage


TASK = '''#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x=1) in;
taskPayloadSharedEXT struct Payload { uint value; } payload;
void main() { payload.value = 1; EmitMeshTasksEXT(1, 1, 1); }
'''

MESH = '''#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x=1) in;
layout(triangles, max_vertices=3, max_primitives=1) out;
void main() {
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[0].gl_Position = vec4(-1, -1, 0, 1);
    gl_MeshVerticesEXT[1].gl_Position = vec4( 1, -1, 0, 1);
    gl_MeshVerticesEXT[2].gl_Position = vec4( 0,  1, 0, 1);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
'''


def minimal(model):
    # Structurally complete enough for the metadata parser; actual mesh modules
    # are compiled by glslc in the tests below.
    ext = struct.unpack('<5I', b'SPV_EXT_mesh_shader\0')
    stage_op = 5294 if model == 5364 else 5295
    words = [0x07230203, 0x00010400, 0, 5, 0,
             (2 << 16) | 17, 5283,
             (6 << 16) | 10, *ext,
             (5 << 16) | 15, model, 1, 0x6e69616d, 0,
             (1 << 16) | stage_op]
    return struct.pack('<%dI' % len(words), *words)


class StageMetadata(unittest.TestCase):
    def test_execution_models_and_sets(self):
        self.assertEqual(module_stage(minimal(5364)), 'ts')
        self.assertEqual(module_stage(minimal(5365)), 'ms')
        self.assertEqual((STAGE_SET['ts'], STAGE_SET['ms']), (3, 4))
        unknown = bytearray(minimal(5364)); struct.pack_into('<I', unknown, 14 * 4, 9999)
        with self.assertRaises(StageError): module_stage(unknown)
        with self.assertRaises(StageError): module_stage(b'bad')

    @unittest.skipUnless(shutil.which('glslc'), 'glslc is needed for actual TaskEXT/MeshEXT fixtures')
    def test_actual_task_mesh_loader_pairs_and_import(self):
        with tempfile.TemporaryDirectory(prefix='dmsh_mesh_stage_') as td:
            root = Path(td); source = root / 'source'; out = root / 'out'; source.mkdir()
            for stage, text, key in (('ts', TASK, '1111111111111111'), ('ms', MESH, '2222222222222222')):
                src = root / ('fixture.' + stage); spv = source / (key + '.spv'); src.write_text(text)
                r = subprocess.run(['glslc', '--target-env=vulkan1.3', '-fshader-stage=' + ('task' if stage == 'ts' else 'mesh'), str(src), '-o', str(spv)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual(module_stage(spv.read_bytes()), stage)
                (source / (key + '.res.json')).write_text(json.dumps({'key': key, 'stage': stage, 'bindings': []}))
                (source / (key + '.job.json')).write_text(json.dumps({'key': key, 'stage': stage, 'c1': ''}))
                current = {'key': key, 'stage': 'cs', 'mesh_stage': {'kind': stage}}
                self.assertTrue(import_native_translation(key, current, source, out))
                fixture = root / ('loader-' + stage); emit_loader_pair(key, stage, out / (key + '.spv'), fixture)
                self.assertEqual((fixture / (key + '.vk.spv')).read_bytes(), spv.read_bytes())
                self.assertEqual((fixture / (key + '.vkb')).read_text(), '# %s %s set %d\n' % (key, stage, STAGE_SET[stage]))

            # A present module with the wrong execution model is never relabelled.
            bad = '3333333333333333'
            shutil.copyfile(source / '1111111111111111.spv', source / (bad + '.spv'))
            (source / (bad + '.res.json')).write_text(json.dumps({'stage': 'ms'}))
            (source / (bad + '.job.json')).write_text(json.dumps({'stage': 'ms'}))
            with self.assertRaises(StageError):
                import_native_translation(bad, {'mesh_stage': {'kind': 'ms'}}, source, out)


if __name__ == '__main__':
    unittest.main()
