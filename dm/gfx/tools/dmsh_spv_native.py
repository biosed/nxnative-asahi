#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build native TaskEXT/MeshEXT inputs with a segregated translator.

This adapter contains no donor translator implementation.  It derives the
native NVN2 job from an existing shader package, invokes an explicitly named
external translator, validates every result as Vulkan 1.3 SPIR-V, and writes
the three files consumed by :mod:`dmsh_spv_stage`.

The validated donor for this tree is BananaNative commit 96cc0e8.  Its
``InstEmitAmpereMesh.cs`` is GPL-3.0-or-later, making the built translator a
GPL-3.0-or-later combined work.  Keep that source and binary outside this MIT
tree and pass its path through ``DMSH_VK_MESH_XLATE``.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import dmsh_ampere
import dmsh_chain
import dmsh_meshstage
from dmsh_spv_stage import NATIVE_MESH_STAGES, require_stage

DONOR_URL = 'https://github.com/moceure/BananaNative.git'
DONOR_COMMIT = '96cc0e883c80117e51342c60facfae2f0e600b9a'
DONOR_LICENSE = 'GPL-3.0-or-later combined work'
# Program.cs + dmsh_xlate.csproj embedded by the translator last changed in
# DONOR_COMMIT.  A different build must be named explicitly, never accepted
# merely because it happens to emit a TaskEXT/MeshEXT execution model.
DONOR_SOURCE_SHA = '0f31637014ce4725'


class NativeError(ValueError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def nvn2_key(control, code):
    """The host/shader_split NVN2 key, including its normalized length word."""
    if not dmsh_ampere.is_nvn2(control) or len(control) < 0x48:
        raise NativeError('not an NVN2 control section')
    key_control = control[:0x30] + bytes(4) + control[0x34:0x48]
    return hashlib.sha256(key_control + code).hexdigest()[:16]


def _native_sph_outputs(sph):
    """Output planes needed by the native Vulkan mesh translator.

    Keep this separate from :mod:`dmsh_meshstage`: that module describes the
    Metal wrapper and may only change with a Mac regression run.  The native
    path names point size explicitly so its facts can be checked before the
    segregated translator sees any title input.
    """
    out = ([('psiz', 0)] if sph[0x35] & 0x08 else [])
    out += [(None, component) for component in range(4)
            if sph[0x35] >> (4 + component) & 1]
    for generic in range(32):
        out += [(generic, component) for component in range(4)
                if sph[0x36 + generic // 2] >>
                (4 * (generic % 2) + component) & 1]
    return out


def native_info(control, area):
    """Control/SPH facts for the native Vulkan task/mesh path only."""
    info = dmsh_meshstage.info(control, area)
    if info.get('kind') == 'ms':
        primitive_at = (struct.unpack_from('<I', control, 0x64)[0]
                        if control[0x68] & 0x80 else 0)
        primitive = (_native_sph_outputs(area[primitive_at:primitive_at + 0x80])
                     if primitive_at else [])
        info.update(verts=control[0x5f] or 3,
                    vertex=_native_sph_outputs(area[:0x80]),
                    primitive=[plane for plane in primitive
                               if isinstance(plane[0], int)])
    return info


def _plane_code(plane):
    generic, component = plane
    if generic == 'psiz':
        if component != 0:
            raise NativeError('point-size output has a component other than zero')
        return -4
    if generic is not None and not isinstance(generic, int):
        raise NativeError('unknown mesh output plane %r' % (plane,))
    return component if generic is None else 4 + 4 * generic + component


def native_layout(info):
    """Convert the MIT control/SPH facts into the donor's documented layout."""
    if info['kind'] == 'ts':
        return [0, 0, info['payload']]
    if info['kind'] != 'ms':
        raise NativeError('control section is not task or mesh')
    return [info['max_v'], info['max_p'], info['payload'], -5, info['verts'],
            *(_plane_code(p) for p in info['vertex']), -1,
            *(_plane_code(p) for p in info['primitive'])]


def load_mesh_facts(path):
    """Load the generated task/mesh rows from ``shader_facts.tsv``."""
    facts = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip() or line.startswith('#'):
                continue
            try:
                key, value = line.rstrip('\n').split('\t', 1)
                value = json.loads(value)
            except (ValueError, json.JSONDecodeError) as e:
                raise NativeError('%s:%d: invalid fact row: %s' % (path, lineno, e))
            if isinstance(value, dict) and 'mesh' in value:
                facts[key] = value
    return facts


def checked_fact_layout(key, stage, info, fact):
    """Validate one generated mesh fact against the package control section."""
    if not isinstance(fact, dict):
        raise NativeError('%s: no generated task/mesh fact' % key)
    local, layout = fact.get('local'), fact.get('mesh')
    expected_local = [info['threads'], 1, 1]
    if local != expected_local:
        raise NativeError('%s: fact local %r, control local %r' %
                          (key, local, expected_local))
    if fact.get('shared') != 0:
        raise NativeError('%s: task/mesh fact shared memory is not zero' % key)
    if not isinstance(layout, list) or any(type(v) is not int for v in layout):
        raise NativeError('%s: fact mesh layout is not an integer list' % key)
    head = ([0, 0, info['payload']] if stage == 'ts' else
            [info['max_v'], info['max_p'], info['payload']])
    if layout[:3] != head:
        raise NativeError('%s: fact mesh header %r, control header %r' %
                          (key, layout[:3], head))
    if stage == 'ts' and layout != head:
        raise NativeError('%s: task fact has mesh outputs' % key)
    if stage == 'ms':
        body = layout[3:]
        topology = [-5, info['verts']]
        if body[:2] != topology:
            raise NativeError('%s: fact mesh topology %r, control topology %r' %
                              (key, body[:2], topology))
        body = body[2:]
        if any(v < 0 and v not in (-1, -2, -3, -4) for v in body):
            raise NativeError('%s: unknown mesh layout marker' % key)
        if body.count(-1) > 1 or body.count(-2) > 1:
            raise NativeError('%s: duplicate mesh layout separator' % key)
        prim_at = body.index(-1) if -1 in body else len(body)
        pad_at = body.index(-2) if -2 in body else len(body)
        no_output_at = [i for i, value in enumerate(body) if value == -3]
        if ((-1 in body and prim_at > pad_at) or
                any(-1 not in body or not prim_at < i < pad_at
                    for i in no_output_at)):
            raise NativeError('%s: mesh layout markers are out of order' % key)
        vertex = body[:min(prim_at, pad_at)]
        primitive = ([v for v in body[prim_at + 1:pad_at] if v != -3]
                     if prim_at < len(body) else [])
        expected_vertex = [_plane_code(v) for v in info['vertex']]
        expected_primitive = [_plane_code(v) for v in info['primitive']]
        if vertex != expected_vertex or primitive != expected_primitive:
            raise NativeError('%s: fact output maps disagree with its SPHs' % key)
        if any(v < 0 or v > 31 for v in body[pad_at + 1:]):
            raise NativeError('%s: padded generic location outside 0..31' % key)
    return layout


def split_record(record):
    """Return ``(control, code area)`` from one shader package record."""
    if len(record) < 0x38 or record[:8] != b'BTCDBTCD':
        raise NativeError('not a shader package record')
    nctl, ncode = struct.unpack_from('<2I', record, 0x28)
    end = 0x38 + ncode + nctl
    if end > len(record):
        raise NativeError('shader package record is cut short')
    return record[0x38 + ncode:end], record[0x38:0x38 + ncode]


def make_native_job(key, stage, record, out_dir, fact=None):
    """Write one native translator input and return its JSON job."""
    control, area = split_record(record)
    if nvn2_key(control, area) != key:
        raise NativeError('%s: package record key does not agree with shader_map.tsv' % key)
    info = native_info(control, area)
    if info.get('kind') != stage or stage not in NATIVE_MESH_STAGES:
        raise NativeError('%s: map stage %s, control stage %s' % (key, stage, info.get('kind')))
    if stage == 'ms' and info.get('verts') not in (1, 2, 3):
        raise NativeError('%s: primitive vertex count %r is not 1, 2 or 3' %
                          (key, info.get('verts')))
    ninst = dmsh_ampere.inst_len(control)
    if ninst < 0x80 or ninst > len(area):
        raise NativeError('%s: instruction/SPH length %#x outside code area %#x' % (key, ninst, len(area)))
    os.makedirs(out_dir, exist_ok=True)
    code_path = os.path.join(out_dir, key + '.bin')
    with open(code_path, 'wb') as f:
        f.write(area[:ninst])
    layout = (checked_fact_layout(key, stage, info, fact)
              if fact is not None else native_layout(info))
    job = {
        'key': key, 'stage': stage, 'code': code_path, 'arch': 'sm86',
        'tex': {}, 'img': {}, 'attr': {}, 'fout': {},
        'c1': dmsh_ampere.c1_segment(control, area).hex(),
        'local': [info['threads'], 1, 1], 'shared': 0,
        'mesh': layout,
        'debug': os.environ.get('XLATE_DEBUG', '1') != '0',
        'reduced': os.environ.get('XLATE_REDUCED', '1') == '1',
    }
    with open(os.path.join(out_dir, key + '.job.json'), 'w') as f:
        json.dump(job, f, indent=2, sort_keys=True)
    return job


def _locations(map_path):
    out = {}
    with open(map_path, newline='') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) >= 4 and row[0].endswith('.bin') and row[1].isdigit():
                stage = {'task': 'ts', 'mesh': 'ms'}.get(row[2])
                if stage:
                    out[row[3]] = (row[0], int(row[1]), stage)
    return out


def _package_records(path):
    with open(path, 'rb') as f:
        data = f.read()
    offsets, at = [], 0
    while True:
        at = data.find(b'BTCDBTCD', at)
        if at < 0:
            break
        offsets.append(at)
        at += 8
    offsets.append(len(data))
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _translator_cmd(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'dmsh_xlate.dll')
    if not os.path.isfile(path):
        raise NativeError('external mesh translator does not exist: %s' % path)
    return (['dotnet', path] if path.endswith('.dll') else [path]), os.path.realpath(path)


def _run_checked(argv, what):
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode:
        msg = (r.stderr.strip() or r.stdout.strip() or 'exit %d' % r.returncode).splitlines()[-1]
        raise NativeError('%s: %s' % (what, msg[:300]))
    return r


def _reported_failures(run):
    failures = {}
    for line in ((run.stderr or '') + '\n' + (run.stdout or '')).splitlines():
        if line.startswith('[RED] ') and ':' in line[6:]:
            key, why = line[6:].split(':', 1)
            failures[key.strip()] = why.strip()[:500]
    return failures


def translate_native(package_dir, out_dir, keys, translator, spirv_val,
                     map_path=None, facts_path=None, batch_size=16,
                     allow_failures=False):
    """Translate selected package keys and return the validated successes.

    Translation is deliberately batched so a title-wide inventory cannot make
    one unbounded translator process.  With ``allow_failures``, every failure
    is retained in ``_native_status.json`` and successful modules remain usable
    by the strict importer; the default still fails closed.
    """
    if not translator:
        raise NativeError('DMSH_VK_MESH_XLATE must name the segregated GPL translator')
    if not spirv_val:
        raise NativeError('SPIRV_VAL/spirv-val is required for native mesh output')
    cmd, translator_path = _translator_cmd(translator)
    source = _run_checked(cmd + ['--source-sha'], 'external translator provenance').stdout.strip()
    if source != DONOR_SOURCE_SHA:
        raise NativeError('external translator source %s, expected pinned %s' %
                          (source or 'unknown', DONOR_SOURCE_SHA))
    validator = shutil.which(spirv_val) if os.path.sep not in spirv_val else spirv_val
    if not validator or not os.path.isfile(validator):
        raise NativeError('spirv-val does not exist: %s' % spirv_val)

    game = dmsh_chain.load_game()
    map_path = map_path or str(game.table('shader_map.tsv'))
    locations = _locations(map_path)
    facts = load_mesh_facts(facts_path) if facts_path else None
    chosen = list(dict.fromkeys(keys)) if keys else sorted(locations)
    unknown = [k for k in chosen if k not in locations]
    if unknown:
        raise NativeError('not task/mesh keys in %s: %s' % (map_path, ', '.join(unknown[:5])))
    os.makedirs(out_dir, exist_ok=True)
    packages, jobs, setup_failures = {}, [], {}
    for key in chosen:
        name, index, stage = locations[key]
        try:
            if name not in packages:
                packages[name] = _package_records(os.path.join(package_dir, name))
            if index >= len(packages[name]):
                raise NativeError('%s: record %d outside %s' % (key, index, name))
            fact = facts.get(key) if facts is not None else None
            if facts is not None and fact is None:
                raise NativeError('%s: no generated task/mesh layout fact' % key)
            jobs.append(make_native_job(key, stage, packages[name][index], out_dir, fact))
        except (OSError, ValueError, NativeError) as e:
            setup_failures[key] = str(e)[:500]
    jobs_path = os.path.join(out_dir, '_native_jobs.json')
    with open(jobs_path, 'w') as f:
        json.dump(jobs, f, indent=2, sort_keys=True)

    batch_size = max(1, int(batch_size))
    reported, returncodes = {}, []
    for job in jobs:
        for ext in ('.spv', '.res.json', '.reader.spv'):
            try:
                os.unlink(os.path.join(out_dir, job['key'] + ext))
            except FileNotFoundError:
                pass
    for first in range(0, len(jobs), batch_size):
        batch = jobs[first:first + batch_size]
        batch_path = os.path.join(out_dir, '_native_jobs_%04d.json' %
                                  (first // batch_size))
        with open(batch_path, 'w') as f:
            json.dump(batch, f, indent=2, sort_keys=True)
        run = subprocess.run(cmd + [batch_path, out_dir], capture_output=True, text=True)
        returncodes.append(run.returncode)
        reported.update(_reported_failures(run))

    failures, successes = dict(setup_failures), []
    for job in jobs:
        key, stage = job['key'], job['stage']
        spv = os.path.join(out_dir, key + '.spv')
        res_path = os.path.join(out_dir, key + '.res.json')
        try:
            if key in reported:
                raise NativeError(reported[key])
            require_stage(spv, stage)
            with open(res_path) as f:
                res = json.load(f)
            if res.get('stage') != stage:
                raise NativeError('reflection stage %r, expected %r' % (res.get('stage'), stage))
            _run_checked([validator, '--target-env', 'vulkan1.3', spv], key + ' spirv-val')
        except (OSError, ValueError, NativeError) as e:
            failures[key] = str(e)[:500]
        else:
            successes.append(job)

    files = [translator_path]
    for dll in ('Ryujinx.Graphics.Shader.dll', 'Spv.Generator.dll'):
        p = os.path.join(os.path.dirname(translator_path), dll)
        if os.path.isfile(p):
            files.append(p)
    version = _run_checked([validator, '--version'], 'spirv-val version').stdout.splitlines()[0]
    provenance = {
        'adapter_license': 'MIT', 'translator_license': DONOR_LICENSE,
        'upstream': DONOR_URL, 'implementation_commit': DONOR_COMMIT,
        'translator_source_sha': source,
        'translator_files': {os.path.basename(p): _sha256(p) for p in files},
        'spirv_val': version, 'target_env': 'vulkan1.3',
        'map': os.path.realpath(map_path),
        'facts': ({'path': os.path.realpath(facts_path), 'sha256': _sha256(facts_path)}
                  if facts_path else None),
        'batch_size': batch_size, 'batch_returncodes': returncodes,
        'requested': [{'key': key, 'stage': locations[key][2]} for key in chosen],
        'successful': [{'key': j['key'], 'stage': j['stage']} for j in successes],
        'failures': failures,
    }
    with open(os.path.join(out_dir, '_native_provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, '_native_status.json'), 'w') as f:
        json.dump({'requested': len(chosen), 'translated': len(jobs),
                   'successful': len(successes),
                   'failed': len(failures), 'failures': failures},
                  f, indent=2, sort_keys=True)
    if failures and not allow_failures:
        detail = ['%s: %s' % item for item in sorted(failures.items())[:6]]
        raise NativeError('native mesh translation failed: ' + '; '.join(detail))
    return successes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package_dir')
    p.add_argument('out_dir')
    p.add_argument('keys', nargs='*', help='task/mesh keys; absent means every task/mesh key')
    p.add_argument('--translator', default=os.environ.get('DMSH_VK_MESH_XLATE'))
    p.add_argument('--spirv-val', default=os.environ.get('SPIRV_VAL') or shutil.which('spirv-val'))
    p.add_argument('--map', dest='map_path', default=os.environ.get('DMSH_VK_MESH_MAP'))
    p.add_argument('--facts', dest='facts_path', default=os.environ.get('DMSH_VK_MESH_FACTS'))
    p.add_argument('--batch-size', type=int,
                   default=int(os.environ.get('DMSH_VK_MESH_BATCH', '16')))
    p.add_argument('--allow-failures', action='store_true',
                   default=os.environ.get('DMSH_VK_MESH_ALLOW_FAILURES') == '1')
    a = p.parse_args(argv)
    jobs = translate_native(a.package_dir, a.out_dir, a.keys, a.translator, a.spirv_val,
                            map_path=a.map_path, facts_path=a.facts_path,
                            batch_size=a.batch_size,
                            allow_failures=a.allow_failures)
    print('native TaskEXT/MeshEXT: %d validated job(s) in %s' % (len(jobs), a.out_dir))


if __name__ == '__main__':
    try:
        main()
    except (OSError, NativeError) as e:
        sys.exit('[RED] ' + str(e))
