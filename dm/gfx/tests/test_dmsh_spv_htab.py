#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))

from dmsh_spv_htab import HtabError, transform_tex_ubo_tables
import dmsh_chain


def tool(name):
    root = os.environ.get('DMSH_FIXTURE_TOOLS')
    return shutil.which(name) or (os.path.join(root, name) if root else None)


GLSLC, SPIRV_VAL, SPIRV_DIS = (tool(n) for n in ('glslc', 'spirv-val', 'spirv-dis'))


@unittest.skipUnless(GLSLC and SPIRV_VAL and SPIRV_DIS,
                     'glslc and SPIRV-Tools are needed for handle-table fixtures')
class HandleTable(unittest.TestCase):
    def test_postprocessor_is_in_chain_identity(self):
        self.assertIn('tools/dmsh_spv_htab.py', dmsh_chain.FILES)

    def compile(self, sampler='sampler3D'):
        source = '''#version 460
layout(set=1,binding=5,std140) uniform Handles { uvec4 row[4096]; } handles;
layout(set=1,binding=320) uniform SAMPLER tex[8192];
layout(location=0) out vec4 colour;
void main() {
    uint i = uint(gl_FragCoord.x) & 8191u;
    colour = texture(tex[i], vec3(0.25));
}
'''.replace('SAMPLER', sampler)
        with tempfile.TemporaryDirectory(prefix='dmsh_htab_') as td:
            src, spv = Path(td) / 'fixture.frag', Path(td) / 'fixture.spv'
            src.write_text(source)
            r = subprocess.run([GLSLC, '--target-env=vulkan1.3', '-O0',
                                str(src), '-o', str(spv)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return spv.read_bytes()

    def validate_disassemble(self, data):
        with tempfile.TemporaryDirectory(prefix='dmsh_htab_out_') as td:
            spv, asm = Path(td) / 'out.spv', Path(td) / 'out.spvasm'
            spv.write_bytes(data)
            r = subprocess.run([SPIRV_VAL, '--target-env', 'vulkan1.3', str(spv)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            subprocess.run([SPIRV_DIS, str(spv), '-o', str(asm)], check=True)
            return asm.read_text()

    def test_3d_table_becomes_separate_nonuniform_pools(self):
        original = self.compile()
        patched, report = transform_tex_ubo_tables(
            original, 1,
            [{'binding': 320, 'cbuf_binding': 5, 'off': 16, 'count': 8192}],
            0x968, 0x96c)
        self.assertEqual((report['tables'], report['loads'],
                          report['pool_base_binding']), (1, 1, 17))
        text = self.validate_disassemble(patched)
        self.assertNotIn('Binding 320', text)
        for binding in (17, 567, 660):
            self.assertIn('Binding %d' % binding, text)
        self.assertIn('OpSampledImage', text)
        self.assertGreaterEqual(text.count('NonUniform'), 7)
        self.assertIn('SampledImageArrayNonUniformIndexing', text)

    def test_metadata_and_shape_mismatches_fail_closed(self):
        original = self.compile()
        with self.assertRaisesRegex(HtabError, 'array count disagrees'):
            transform_tex_ubo_tables(
                original, 1,
                [{'binding': 320, 'cbuf_binding': 5, 'off': 0, 'count': 4096}],
                0x968, 0x96c)
        cube = self.compile('samplerCube')
        with self.assertRaisesRegex(HtabError, 'has no bindless pool'):
            transform_tex_ubo_tables(
                cube, 1,
                [{'binding': 320, 'cbuf_binding': 5, 'off': 0, 'count': 8192}],
                0x968, 0x96c)


if __name__ == '__main__':
    unittest.main()
