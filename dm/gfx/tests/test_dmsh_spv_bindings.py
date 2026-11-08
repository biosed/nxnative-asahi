#!/usr/bin/env python3
"""Device-free regression for dmsh_spv's canonical union-layout ranges."""

import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))

import dmsh_spv_bindings as binding


class CanonicalBindingRanges(unittest.TestCase):
    def test_all_descriptor_ranges_are_disjoint_and_bounded(self):
        ranges = {(group, desc): set(binding.descriptor_range(group, desc))
                  for group in ('fallback', 'duplicate')
                  for desc in binding.DESCRIPTOR_CODES}
        self.assertEqual(len(ranges), 14)
        for key, values in ranges.items():
            self.assertEqual(len(values), 32, key)
            self.assertGreaterEqual(min(values), 0, key)
            self.assertLess(max(values), binding.VK_BINDING_LIMIT, key)
        items = list(ranges.items())
        for i, (left_key, left) in enumerate(items):
            for right_key, right in items[i + 1:]:
                self.assertTrue(left.isdisjoint(right), (left_key, right_key))
        self.assertEqual(min(ranges['fallback', 'u']), 500)
        self.assertEqual(max(ranges['fallback', '?']), 723)
        self.assertEqual(min(ranges['duplicate', 'u']), 736)
        self.assertEqual(max(ranges['duplicate', '?']), 959)

    def test_production_allocator_uses_each_exact_range(self):
        next_index = {}

        def seq(tag, base, count):
            i = next_index.get(tag, 0)
            next_index[tag] = i + 1
            if i >= count:
                raise ValueError(f'more than {count} bindings of kind {tag}')
            return base + i

        allocated = set()
        for group, tag in (('fallback', 'x'), ('duplicate', 'dup')):
            for desc in binding.DESCRIPTOR_CODES:
                expected = tuple(v for v in binding.descriptor_range(group, desc)
                                 if v != binding.MESH_FIRST_BINDING)
                got = [binding.descriptor_binding(seq, group, tag, desc)
                       for _ in expected]
                self.assertEqual(got, list(expected))
                self.assertTrue(allocated.isdisjoint(got))
                allocated.update(got)
                with self.assertRaises(ValueError):
                    binding.descriptor_binding(seq, group, tag, desc)
        self.assertEqual(len(allocated), 14 * binding.DESCRIPTOR_BINDINGS - 1)
        self.assertNotIn(binding.MESH_FIRST_BINDING, allocated)
        self.assertLess(max(allocated), binding.VK_BINDING_LIMIT)

    def test_linux_mesh_first_header_agrees_and_is_reserved(self):
        header = (TOOLS.parent / 'nvn' / 'dmsh_abi_linux_vk.h').read_text()
        self.assertIn('#define DMSH_LINUX_VK_MESH_FIRST_BINDING %d' %
                      binding.MESH_FIRST_BINDING, header)
        self.assertIn('#define DMSH_LINUX_VK_BINDING_LIMIT %d' %
                      binding.VK_BINDING_LIMIT, header)

    def test_unknown_group_or_descriptor_fails_closed(self):
        with self.assertRaises(ValueError):
            binding.descriptor_range('fallback', 'z')
        with self.assertRaises(ValueError):
            binding.descriptor_range('other', 'u')


if __name__ == '__main__':
    unittest.main()
