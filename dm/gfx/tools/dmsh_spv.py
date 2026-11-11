#!/usr/bin/env python3
# Linux (nxasahi): game shaders as SPIR-V for the Vulkan backend, from the same translation as the Metal chain, stopped before SPIRV-Cross.
# Current NVN2 task/mesh jobs are counted and omitted by default. DMSH_VK_MESH_EXT_DIR may name native Vulkan translator output for those keys; its job, reflection and
# TaskEXT/MeshEXT execution model must agree before it is imported. The runtime still refuses these stages until its extension/device/command path is complete.
# Usage: DM_GAME=<id> dmsh_spv.py <Switch package dir> <output dir> [key...]   (as dmsh_build.py; output <output dir>/spv-<chain version>/<key>.spv, .res.json, .job.json)
# It runs the unmodified dmsh_build.py, so the jobs (stage, facts, constant buffer 1 contents, pairs) are exactly the Metal chain's, and changes nothing the chain version
# covers (dmsh_chain.FILES: editing dmsh_build.py or dmsh_chain.py would make every Mac and iOS library stale). Three hooks, all here:
#   - the tool check covers the translator only (its pinned dlls and embedded source, dmsh_chain.lock): spirv-cross and the Metal compilers are not part of this chain
#   - the translator runs through dotnet (its dmsh_xlate apphost is a macOS executable; the dlls are the pinned ones)
#   - each translation's outputs are collected as it returns, and the build stops at its first SPIRV-Cross call, where the Metal-only part begins
import json, os, runpy, shutil, subprocess, sys, threading
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dmsh_chain   # before XLATE is set: XLATE_DIR stays the pinned build in dmsh_xlate_bin
from dmsh_spv_stage import STAGE_SET, StageError, import_native_translation
from dmsh_spv_hmma import lower_file as lower_hmma_file, remove_outputs
from dmsh_spv_htab import HtabError, transform_tex_ubo_tables
from dmsh_spv_vector import optimize_vector_loads
import dmsh_abi as _abi

if len(sys.argv) < 3: sys.exit('usage: DM_GAME=<id> dmsh_spv.py <Switch package dir> <output dir> [key...]')
DLL = os.path.join(dmsh_chain.XLATE_DIR, 'dmsh_xlate.dll')
OUT = os.path.join(sys.argv[2], 'spv-' + dmsh_chain.version()); os.makedirs(OUT, exist_ok=True)
os.makedirs(sys.argv[2], exist_ok=True)

# Native task/mesh semantics live in a segregated GPL translator, not in this
# MIT tree. With an explicit translator, make its validated import directory
# first; DMSH_VK_MESH_EXT_DIR remains the lower-level/manual interface.
if os.environ.get('DMSH_VK_MESH_XLATE') and not os.environ.get('DMSH_VK_MESH_EXT_DIR'):
    from dmsh_spv_native import DONOR_SOURCE_SHA, translate_native
    native_dir = os.path.join(sys.argv[2], 'native-mesh-' + DONOR_SOURCE_SHA)
    translate_native(sys.argv[1], native_dir, sys.argv[3:], os.environ['DMSH_VK_MESH_XLATE'],
                     os.environ.get('SPIRV_VAL') or shutil.which('spirv-val'),
                     map_path=os.environ.get('DMSH_VK_MESH_MAP'),
                     facts_path=os.environ.get('DMSH_VK_MESH_FACTS'),
                     batch_size=int(os.environ.get('DMSH_VK_MESH_BATCH', '16')),
                     allow_failures=os.environ.get('DMSH_VK_MESH_ALLOW_FAILURES') == '1')
    os.environ['DMSH_VK_MESH_EXT_DIR'] = native_dir
wrapper = os.path.join(sys.argv[2], 'dmsh_xlate_dotnet')
with open(wrapper, 'w') as f: f.write(f'#!/bin/sh\nexec dotnet "{DLL}" "$@"\n')
os.chmod(wrapper, 0o755); os.environ['XLATE'] = wrapper

def check_tools():
    lock = os.environ.get('DMSH_XLATE_LOCK', dmsh_chain.LOCK)
    want = dict(l.rstrip('\n').split('\t', 1) for l in open(lock) if '\t' in l and not l.startswith('#'))
    bad = [f'{d}: lock {want.get(d)} / actual {tool_lock.sha16(os.path.join(dmsh_chain.XLATE_DIR, d))}' for d in ('dmsh_xlate.dll', 'Ryujinx.Graphics.Shader.dll', 'Ryujinx.Common.dll')
           if want.get(d) != tool_lock.sha16(os.path.join(dmsh_chain.XLATE_DIR, d))]
    src = subprocess.run(['dotnet', DLL, '--source-sha'], capture_output=True, text=True).stdout.strip()
    if src != want.get('xlate_source'): bad.append(f'translator source: embedded {src} / lock {want.get("xlate_source")}')
    if 'DMSH_XLATE_LOCK' not in os.environ and src != dmsh_chain.xlate_source_sha():
        bad.append(f'translator source: embedded {src} / repository {dmsh_chain.xlate_source_sha()}')
    return bad
dmsh_chain.check_tools = check_tools
import tool_lock
tool_lock.check_tools = check_tools

# Storage images without a declared format (OpTypeImage Sampled 2, Format Unknown) need StorageImageRead/WriteWithoutFormat in Vulkan; the translator's SPIR-V
# leaves them out (Metal never asks; Ryujinx's own Vulkan backend adds them from the device's features). Inserted right after the header, where capabilities go
def fix_caps(path):
    import struct
    b = open(path, 'rb').read(); w = list(struct.unpack('<%dI' % (len(b) // 4), b)); i, caps, unk = 5, set(), False
    while i < len(w):
        op, n = w[i] & 0xFFFF, w[i] >> 16
        if n == 0: break
        if op == 17: caps.add(w[i + 1])                                   # OpCapability
        if op == 25 and w[i + 7] == 2 and w[i + 8] == 0: unk = True         # OpTypeImage, Sampled 2 (storage), ImageFormat Unknown
        i += n
    add = [c for c in (55, 56) if unk and c not in caps]                  # StorageImageReadWithoutFormat, StorageImageWriteWithoutFormat
    if add: open(path, 'wb').write(struct.pack('<%dI' % (len(w) + 2 * len(add)), *(w[:5] + [x for c in add for x in ((2 << 16) | 17, c)] + w[5:])))
    return len(add)

class Translated(Exception): pass
_run, _lock, _state = subprocess.run, threading.Lock(), {'jobs': 0, 'spv': 0, 'native_mesh': 0, 'failed': {}, 'caps': 0, 'hmma': 0, 'hmma_shaders': 0, 'htab_tables': 0, 'htab_loads': 0, 'htab_shaders': 0, 'vec2': 0, 'vec4': 0, 'vec_loads': 0, 'vec_shaders': 0, 'skipped_stage': {}, 'cross': 0}

def lower_hmma(path):
    """Lower the current TensorPass HMMA helper and validate the final module."""
    validator = os.environ.get('SPIRV_VAL') or shutil.which('spirv-val')
    return lower_hmma_file(path, validator, _run)

def run(args, *a, **k):
    if isinstance(args, (list, tuple)) and args and args[0] == wrapper and len(args) == 3 and not args[1].startswith('-'):
        r = _run(args, *a, **k)
        jobs, w = json.load(open(args[1])), args[2]
        with _lock:
            for line in (r.stderr or '').splitlines():
                if line.startswith('[RED] ') and ':' in line: key, msg = line[6:].split(':', 1); _state['failed'][key.strip()] = msg.strip()[:200]
            for job in jobs:
                key = job['key']; _state['jobs'] += 1
                stage = job.get('mesh_stage', {}).get('kind') if job.get('mesh_stage') else job.get('stage')
                if job.get('mesh_stage'):
                    src = os.environ.get('DMSH_VK_MESH_EXT_DIR')
                    try:
                        imported = bool(src) and import_native_translation(key, job, src, OUT)
                    except (OSError, ValueError, StageError) as e:
                        _state['failed'][key] = 'native mesh import: ' + str(e)[:180]
                        imported = False
                    if imported:
                        _state['caps'] += bool(fix_caps(os.path.join(OUT, key + '.spv')))
                        _state['spv'] += 1; _state['native_mesh'] += 1
                    else:
                        _state['skipped_stage'][stage or '?'] = _state['skipped_stage'].get(stage or '?', 0) + 1
                    continue
                if stage not in STAGE_SET:
                    _state['skipped_stage'][stage or '?'] = _state['skipped_stage'].get(stage or '?', 0) + 1
                    continue
                if not os.path.exists(os.path.join(w, key + '.spv')): continue
                for ext in ('.spv', '.res.json', '.reader.spv'):
                    if os.path.exists(os.path.join(w, key + ext)): shutil.copyfile(os.path.join(w, key + ext), os.path.join(OUT, key + ext))
                spv = os.path.join(OUT, key + '.spv')
                caps = bool(fix_caps(spv))
                try:
                    hmma = lower_hmma(spv) if stage == 'cs' else 0
                except (OSError, ValueError, KeyError, StopIteration) as e:
                    _state['failed'][key] = 'HMMA lowering: ' + str(e)[:180]
                    remove_outputs(OUT, key)
                    continue
                _state['caps'] += caps
                _state['hmma'] += hmma
                _state['hmma_shaders'] += bool(hmma)
                json.dump(job, open(os.path.join(OUT, key + '.job.json'), 'w'))
                _state['spv'] += 1
        return r
    if isinstance(args, (list, tuple)) and args and (os.path.realpath(args[0]) == os.path.realpath(dmsh_chain.SPIRV_CROSS) or os.path.basename(args[0]) == 'spirv-cross'):
        if not _state['jobs'] and '--version' in args:
            # dmsh_build fingerprints the Metal-only converter before translating. The pinned binary is a macOS executable; Vulkan output never executes it.
            return subprocess.CompletedProcess(args, 0, 'dmsh_spv: SPIRV-Cross bypassed on Linux\n', '')
        if _state['jobs']:
            # Current dmsh_build pipelines translation in bounded batches. Make the Metal conversion fail cleanly so it advances through every batch; copied SPIR-V survives while its working folder is removed.
            _state['cross'] += 1
            return subprocess.CompletedProcess(args, 1, '', 'dmsh_spv: Metal conversion intentionally skipped')
    if isinstance(args, (list, tuple)) and args and args[0] == 'xcrun':   # the Metal library cache asks for the compiler's version at setup: there is none
        if _state['jobs']: raise Translated()
        return subprocess.CompletedProcess(args, 1, '', 'xcrun: not on Linux')
    return _run(args, *a, **k)
subprocess.run = run

# ---- Vulkan reflection: <key>.vk.json ----
# What each SPIR-V binding is to the replay (a uniform slot, a texture slot, a storage buffer and where its address is, the support buffer …) is decided by the Metal chain's
# reflect() in dmsh_build.py from the resources' names. It is taken from there unchanged: its function definitions and constant tables are lifted out of the parsed source
# (nothing of the script runs), and it is handed a stand-in for SPIRV-Cross's argument buffer struct made from the SPIR-V: one member per resource, a combined image-sampler as
# the texture and its "Smplr" sampler, as SPIRV-Cross declares them. Its answers keep the member ids, which are the SPIR-V bindings
import ast, struct
from dmsh_spv_bindings import descriptor_binding
from dmsh_spv_bindings import MESH_FIRST_BINDING
from dmsh_spv_mesh_first import transform_mesh_first
def _lift_build():
    src = open(os.path.join(HERE, 'dmsh_build.py')).read(); tree = ast.parse(src); keep = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef): keep.append(n)
        elif isinstance(n, ast.Assign) and all(isinstance(t, (ast.Name, ast.Tuple)) for t in n.targets):
            names = [x.id for t in n.targets for x in ([t] if isinstance(t, ast.Name) else t.elts) if isinstance(x, ast.Name)]
            if names and all(x.isupper() for x in names):
                try: ast.literal_eval(n.value); keep.append(n)
                except ValueError: pass
    import re as _re, dmsh_geometry as _geo, dmsh_abi as _abi, collections as _col
    ns = {'re': _re, 'os': os, 'json': json, 'dmsh_geometry': _geo, 'abi': _abi, 'collections': _col, '__name__': 'dmsh_build_lifted',
          'TEX_BASE': _abi.C2_TEX, 'IMG_BASE': _abi.C2_IMG, 'IMG_SLOTS': _abi.C2_IMG_SLOTS,
          'SEPTEX_BASE': _abi.C2_SEPTEX, 'SEPTEX_SLOT0': _abi.SEPTEX_SLOT0, 'SAMP_BASE': _abi.C2_SAMP,
          'SSBO': {'vs': _abi.C0_SSBO_VS, 'cs': _abi.C0_SSBO_CS, 'gs': _abi.C0_SSBO_GS, 'ps': _abi.C0_SSBO_PS,
                   'ts': _abi.C0_SSBO_TS, 'ms': _abi.C0_SSBO_MS}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'dmsh_build.py (lifted)', 'exec'), ns)
    return ns
_B = None
def _spv(path):
    b = open(path, 'rb').read(); w = struct.unpack('<%dI' % (len(b) // 4), b); i, ins = 5, []
    while i < len(w):
        op, n = w[i] & 0xFFFF, w[i] >> 16
        if not n: break
        ins.append((op, w[i + 1:i + n])); i += n
    return ins
def _cstr(ws):
    return b''.join(struct.pack('<I', x) for x in ws).split(b'\0', 1)[0].decode('utf-8', 'replace')
def vk_reflect(key):
    global _B
    if _B is None: _B = _lift_build()
    spv, res = os.path.join(OUT, key + '.spv'), json.load(open(os.path.join(OUT, key + '.res.json')))
    ins = _spv(spv)
    name = {a[0]: _cstr(a[1:]) for op, a in ins if op == 5}
    bind = {a[0]: a[2] for op, a in ins if op == 71 and a[1] == 33}
    dset = {a[0]: a[2] for op, a in ins if op == 71 and a[1] == 34}
    types = {a[0]: (op, a) for op, a in ins if op in (25, 26, 27, 28, 29, 30, 32)}
    const = {a[1]: a[2] for op, a in ins if op == 43 and len(a) > 2}
    members, kinds = [], {}
    for op, a in ins:
        if op != 59 or a[1] not in bind: continue
        var, sc = a[1], a[2]
        t = types.get(a[0], (0, ()))[1]; t = t[2] if len(t) > 2 else None   # pointee
        arr = None
        while t in types and types[t][0] in (28, 29):
            ta = types[t][1]; arr = const.get(ta[2], 0) if types[t][0] == 28 else 0; t = ta[1]
        top = types.get(t, (0, ()))[0]
        nm = name.get(var, '_%d' % var)
        if top in (25, 27):
            img = types[t][1] if top == 25 else types[types[t][1][1]][1]   # OpTypeImage operands: id, sampled type, dim, depth, arrayed, ms, sampled, format
            dim, sampled = img[2], img[6]
            ty = 'texture_buffer<float>' if dim == 5 else ('texture2d<float, access::read_write>' if sampled == 2 else 'texture2d<float>')
            if arr is not None: ty = 'array<%s, %d>' % (ty, arr or 1)
            members.append((ty, nm, bind[var])); kinds[bind[var]] = 'image' if sampled == 2 else 'texel_buffer' if dim == 5 else 'texture'
            if top == 27: members.append(('array<sampler, %d>' % (arr or 1) if arr is not None else 'sampler', nm + 'Smplr', bind[var]))
        elif top == 26: members.append(('sampler', nm, bind[var])); kinds[bind[var]] = 'sampler'
        else: members.append(('device void*', nm, bind[var])); kinds[bind[var]] = 'storage' if sc == 12 or (sc == 2 and any(op2 == 71 and a2[0] == t and a2[1] == 3 for op2, a2 in ins)) else 'uniform'
    msl = 'struct spvDescriptorSetBuffer0\n{\n' + ''.join('    %s %s [[id(%d)]];\n' % m for m in members) + '};\n'
    binds = _B['reflect'](key, res['stage'], msl, res)
    for e in binds:
        e['binding'] = e.pop('arg_id'); e['set'] = 0; e['vk'] = kinds.get(e['binding'], '?')
        e['name'] = e.pop('msl_name'); e.pop('msl_type', None)
    out = {'key': key, 'stage': res['stage'], 'bindings': binds, 'frag_out_map': res.get('frag_out_map'), 'uses_frag_coord': res.get('uses_frag_coord'),
           'inputs': sorted(a[2] for op, a in ins if op == 71 and a[1] == 30 and any(op2 == 59 and a2[1] == a[0] and a2[2] == 1 for op2, a2 in ins)),
           'outputs': sorted(a[2] for op, a in ins if op == 71 and a[1] == 30 and any(op2 == 59 and a2[1] == a[0] and a2[2] == 3 for op2, a2 in ins))}
    json.dump(out, open(os.path.join(OUT, key + '.vk.json'), 'w'))
    return binds

# ---- canonical bindings: <key>.vk.spv and <key>.vkb ----
# Shader objects of different stages are bound together, so their descriptor set layouts must agree. Every stage gets a set of its own (vertex 0, pixel 1, geometry 2,
# hull 3, domain 4; compute 0 in its own layout), and each resource a fixed binding by what it is (the table above vk_reflect's answers): the host then has one set layout
# per stage for every shader. <key>.vk.spv is the SPIR-V with its DescriptorSet / Binding decorations rewritten so; <key>.vkb lists the bindings for the host, one per line:
#   binding <TAB> descriptor (u uniform, s storage, t combined texture, i storage image, b texel buffer, p sampler) <TAB> kind <TAB> reg <TAB> off <TAB> sampler kind <TAB> sampler reg/id <TAB> sampler off
#   <TAB> dim (1 1D 2 2D 3 3D 4 cube 5 buffer) <TAB> arrayed <TAB> multisampled <TAB> depth compare <TAB> array count
CANON = {'support': 0, 'c1': 15}
def canon_binding(e, seq, desc):
    k, r = e['kind'], e.get('reg')
    if k in CANON: return CANON[k]
    if k == 'cbuf' and r is not None and 0 <= r < 14: return 1 + r
    if k == 'cbuf' and r == 21: return 19   # NVN2 push constants, c[24] (dmsh_abi.h DMSH_PUSH_BANK): vk_draw.c push_fill binds the replay's push block there
    if k == 'drvcb': return {0: 16, 2: 17, 24: 18}.get(e.get('bank'), 18)
    if k == 'sbuf' and r is not None and 0 <= r < 16: return 20 + r
    if k == 'ubo_ptr' and r is not None and 0 <= r < 14: return 36 + r
    if k in ('sbuf', 'sbuf_ubo'): return seq('s', 50, 14)
    if k == 'tex' and r is not None and 0 <= r < 256: return 64 + r
    if k == 'tex_ubo': return seq('u', 320, 32)
    if k == 'tex_c2a': return seq('a', 352, 16)
    if k == 'tbuf' and r is not None and 0 <= r < 32: return 400 + r
    if k == 'img' and r is not None and 0 <= r < 16: return 460 + r
    if k == 'geometry' and r is not None: return 480 + r
    # A fallback is not one descriptor class: an unplaced storage buffer,
    # texture, or sampler may all arrive here.  The host forms one union set
    # layout across every shader, so different descriptor types must never
    # reuse a binding merely because they are in different modules.
    return descriptor_binding(seq, 'fallback', 'x', desc)
def vk_canon(key):
    vk = json.load(open(os.path.join(OUT, key + '.vk.json')))
    ins = _spv(os.path.join(OUT, key + '.spv'))
    bind = {a[0]: a[2] for op, a in ins if op == 71 and a[1] == 33}
    by_binding = {}
    for e in vk['bindings']: by_binding.setdefault(e['binding'], []).append(e)
    used, nxt = {}, {}
    def seq(tag, base, n):
        i = nxt.get(tag, 0); nxt[tag] = i + 1
        if i >= n: raise ValueError(f'more than {n} bindings of kind {tag}')
        return base + i
    # image types of the resources: dim, arrayed, ms, depth (OpTypeImage), and array counts
    types = {a[0]: (op, a) for op, a in ins if op in (25, 26, 27, 28, 29, 32)}
    const = {a[1]: a[2] for op, a in ins if op == 43 and len(a) > 2}
    info = {}
    for op, a in ins:
        if op != 59 or a[1] not in bind: continue
        t = types.get(a[0], (0, ()))[1]; t = t[2] if len(t) > 2 else None; cnt = 1
        while t in types and types[t][0] in (28, 29):
            ta = types[t][1]; cnt = const.get(ta[2], 1) if types[t][0] == 28 else 0; t = ta[1]
        img = None
        if t in types and types[t][0] == 27: t = types[t][1][1]
        if t in types and types[t][0] == 25: img = types[t][1]
        dim = {0: 1, 1: 2, 2: 3, 3: 4, 5: 5}.get(img[2], 2) if img else 0
        info[bind[a[1]]] = (dim, img[4] if img else 0, img[5] if img else 0, img[3] if img else 0, cnt)
    remap, rows, htab_tables = {}, [], []
    for b, es in sorted(by_binding.items()):
        main = next((e for e in es if not e['name'].endswith('Smplr')), es[0])
        smp = next((e for e in es if e['name'].endswith('Smplr')), None)
        desc = {'uniform': 'u', 'storage': 's', 'texture': 't', 'image': 'i', 'texel_buffer': 'b', 'sampler': 'p'}.get(main['vk'], '?')
        cb = canon_binding(main, seq, desc)
        # A second variable on a slot that already has one (a texture register sampled through two samplers or image types, e.g. plain and depth compare) gets a binding
        # of its own from 600 on, 32 per descriptor type so the host's union layout keeps one type per binding; its row names the same slot, so the host binds the same
        # resource there. These shaders used to be dropped whole: 3,174 of Breath of the Wild's, 4,501 of Mario Kart 8 Deluxe's
        if cb in used: cb = descriptor_binding(seq, 'duplicate', 'dup', desc)
        used[cb] = b; remap[b] = cb
        dim, arr, ms, dep, cnt = info.get(b, (0, 0, 0, 0, 1))
        main_kind = main['kind']
        samp_kind = smp['kind'] if smp else ''
        if main_kind == _abi.KIND_TEX_UBO and cnt >= 33:
            moff = main.get('off', 0) if main.get('off') is not None else 0
            soff = smp.get('off', 0) if smp and smp.get('off') is not None else 0
            if (not smp or samp_kind != _abi.KIND_SAMP_UBO or
                    main.get('reg') != smp.get('reg') or moff != soff):
                raise HtabError('binding %d large tex_ubo is not one combined handle table' % cb)
            cbuf_reg = main.get('reg')
            if cbuf_reg is None or not 0 <= cbuf_reg < 14:
                raise HtabError('binding %d large tex_ubo has no canonical cbuf' % cb)
            htab_tables.append({'binding': cb, 'cbuf_binding': 1 + cbuf_reg,
                                'off': moff, 'count': cnt})
            main_kind, samp_kind = _abi.KIND_TEX_HTAB, _abi.KIND_SAMP_HTAB
        sreg = smp.get('reg') if smp else ''
        rows.append('\t'.join(str(x) for x in (cb, desc, main_kind, '' if main.get('reg') is None else main['reg'], main.get('off', '') if main.get('off') is not None else '',
                                                 samp_kind, '' if sreg is None else (sreg if not isinstance(sreg, (list, tuple)) else sreg[1]),
                                                 smp.get('off', '') if smp and smp.get('off') is not None else '', dim, arr, ms, dep, cnt)))
    st = STAGE_SET.get(vk['stage'], 0)
    b = open(os.path.join(OUT, key + '.spv'), 'rb').read(); w = list(struct.unpack('<%dI' % (len(b) // 4), b)); i = 5
    while i < len(w):
        op, n = w[i] & 0xFFFF, w[i] >> 16
        if not n: break
        if op == 71 and w[i + 1] in bind:
            if w[i + 2] == 34: w[i + 3] = st
            elif w[i + 2] == 33: w[i + 3] = remap.get(w[i + 3], w[i + 3])
        i += n
    data = struct.pack('<%dI' % len(w), *w)
    htab = {'tables': 0, 'loads': 0}
    if htab_tables:
        data, htab = transform_tex_ubo_tables(
            data, st, htab_tables, _abi.C2_TEX_BASE, _abi.C2_SAMP_BASE)
        rows.append('\t'.join(str(x) for x in
                    (17, 'u', _abi.KIND_DRVCB, 2, '', '', '', '', 0, 0, 0, 0, 1)))
        job_path = os.path.join(OUT, key + '.job.json')
        job = json.load(open(job_path))
        job['vk_handle_tables'] = htab['tables']
        job['vk_handle_table_loads'] = htab['loads']
        with open(job_path, 'w') as jf:
            json.dump(job, jf, indent=2, sort_keys=True); jf.write('\n')
    if vk['stage'] in ('ts', 'ms'):
        job = json.load(open(os.path.join(OUT, key + '.job.json')))
        if vk['stage'] == 'ms' and 'mesh_has_task' not in job:
            raise ValueError('native mesh job has no mesh_has_task ownership fact')
        data, first = transform_mesh_first(data, vk['stage'], bool(job.get('mesh_has_task', False)), st)
        job['mesh_first'] = first['binding'] is not None
        job['mesh_first_owner'] = bool(first['owner'])
        job['mesh_first_owner_no_use'] = bool(first.get('owner_no_use', False))
        with open(os.path.join(OUT, key + '.job.json'), 'w') as jf:
            json.dump(job, jf, indent=2, sort_keys=True); jf.write('\n')
        if first['binding'] is not None:
            rows.append('\t'.join(str(x) for x in (MESH_FIRST_BINDING, 'u', 'mesh_first', '', '', '', '', '', 0, 0, 0, 0, 1)))
    try:
        data, vector = optimize_vector_loads(data, vk)
        validator = os.environ.get('SPIRV_VAL') or shutil.which('spirv-val')
        if not validator:
            raise ValueError('canonical SPIR-V validation needs SPIRV_VAL or spirv-val on PATH')
        temp = os.path.join(OUT, key + '.vk.spv.validate.tmp')
        try:
            with open(temp, 'wb') as f: f.write(data)
            # Dynamic Offset on OpImageFetch is valid with VK_KHR_maintenance8;
            # the runtime queries and enables that feature before loading one.
            result = _run([validator, '--target-env', 'vulkan1.3',
                           '--allow-offset-texture-operand', temp],
                          capture_output=True, text=True)
            if result.returncode:
                raise ValueError((result.stderr or result.stdout or 'spirv-val failed').strip()[:300])
        finally:
            try: os.unlink(temp)
            except FileNotFoundError: pass
    except Exception:
        # An earlier run may have left a loadable canonical shader. ABI drift
        # or failed validation must remove it and its binding table together.
        remove_outputs(OUT, key)
        raise
    spv_path = os.path.join(OUT, key + '.vk.spv'); spv_tmp = spv_path + '.tmp'
    vkb_path = os.path.join(OUT, key + '.vkb'); vkb_tmp = vkb_path + '.tmp'
    try:
        with open(spv_tmp, 'wb') as f: f.write(data)
        with open(vkb_tmp, 'w') as f: f.write(f'# {key} {vk["stage"]} set {st}\n' + ''.join(r + '\n' for r in rows))
        # Remove both old halves first, then publish the shader last. A crash
        # can leave the pair absent, never a new shader with stale bindings.
        for old in (spv_path, vkb_path):
            try: os.unlink(old)
            except FileNotFoundError: pass
        os.replace(vkb_tmp, vkb_path)
        os.replace(spv_tmp, spv_path)
    except Exception:
        # The runtime consumes the shader and binding table as one ABI unit.
        # Never leave one new half paired with an old or missing other half.
        remove_outputs(OUT, key)
        for temp in (spv_tmp, vkb_tmp):
            try: os.unlink(temp)
            except FileNotFoundError: pass
        raise
    vector['htab_tables'] = htab['tables']; vector['htab_loads'] = htab['loads']
    return vector

sys.argv = [os.path.join(HERE, 'dmsh_build.py')] + sys.argv[1:]
try: runpy.run_path(sys.argv[0], run_name='__main__')
except Translated: pass
import collections as _c
_kinds, _noreg = _c.Counter(), 0
for f in sorted(os.listdir(OUT)):
    if not f.endswith('.spv') or '.reader.' in f or f.endswith('.vk.spv'): continue   # <key>.vk.spv is vk_canon's output from an earlier run, not a shader
    try:
        bs = vk_reflect(f[:-4])
        vector = vk_canon(f[:-4])
        _state['htab_tables'] += vector['htab_tables']; _state['htab_loads'] += vector['htab_loads']; _state['htab_shaders'] += bool(vector['htab_tables'])
        _state['vec2'] += vector['groups2']; _state['vec4'] += vector['groups4']; _state['vec_loads'] += vector['scalar_loads']; _state['vec_shaders'] += bool(vector['scalar_loads'])
        for b in bs: _kinds[b['kind']] += 1; _noreg += b['kind'] in ('cbuf', 'tex', 'img', 'sbuf', 'samp', 'samp_tex', 'tbuf') and b.get('reg') is None
    except Exception as e:
        remove_outputs(OUT, f[:-4])
        print(f'  [RED] {f}: reflection failed: {e}')
        _state['failed'][f[:-4]] = 'reflection: ' + str(e)
print('Vulkan reflection (.vk.json): bindings by kind ' + ', '.join(f'{k} {v}' for k, v in _kinds.most_common()) + f'; without a slot {_noreg}')
print(f'SPIR-V: {_state["spv"]} shaders written from {_state["jobs"]} jobs to {OUT} (native task/mesh imports {_state["native_mesh"]}); translation/reflection failed {len(_state["failed"])}; storage image capabilities added to {_state["caps"]}')
print(f'SPIR-V HMMA: {_state["hmma"]} current-ABI helper calls lowered in {_state["hmma_shaders"]} compute shaders')
print(f'SPIR-V handle tables: {_state["htab_tables"]} large per-draw arrays lowered to {_state["htab_loads"]} persistent-pool accesses in {_state["htab_shaders"]} shaders')
print(f'SPIR-V vector loads: {_state["vec_loads"]} scalar loads coalesced as {_state["vec2"]} x 64-bit and {_state["vec4"]} x 128-bit groups in {_state["vec_shaders"]} shaders')
print('SPIR-V stages omitted (no validated DMSH_VK_MESH_EXT_DIR input): ' + (', '.join(f'{k} {v}' for k, v in sorted(_state['skipped_stage'].items())) or 'none') + f'; Metal conversions bypassed {_state["cross"]}')
for key, msg in list(_state['failed'].items())[:10]: print(f'  [RED] {key}: {msg}')
sys.exit(0 if _state['jobs'] else 1)
