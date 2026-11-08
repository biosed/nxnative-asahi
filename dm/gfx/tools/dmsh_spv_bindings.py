"""Side-effect-free canonical fallback ranges shared by dmsh_spv and tests."""

DESCRIPTOR_CODES = 'ustibp?'
DESCRIPTOR_BINDINGS = 32
FALLBACK_BINDING_BASE = 500
DUPLICATE_BINDING_BASE = 736
VK_BINDING_LIMIT = 1024
MESH_FIRST_BINDING = 512


def descriptor_range(group, desc):
    """The half-open canonical binding range for a descriptor class."""
    if desc not in DESCRIPTOR_CODES:
        raise ValueError(f'unknown descriptor class {desc!r}')
    try:
        base = {'fallback': FALLBACK_BINDING_BASE,
                'duplicate': DUPLICATE_BINDING_BASE}[group]
    except KeyError as e:
        raise ValueError(f'unknown canonical binding group {group!r}') from e
    start = base + DESCRIPTOR_BINDINGS * DESCRIPTOR_CODES.index(desc)
    stop = start + DESCRIPTOR_BINDINGS
    if stop > VK_BINDING_LIMIT:
        raise ValueError(f'{group} {desc} range {start}..{stop - 1} exceeds Vulkan binding limit')
    return range(start, stop)


def descriptor_binding(seq, group, tag, desc):
    """Allocate one binding from a descriptor-specific canonical range."""
    r = descriptor_range(group, desc)
    # 512 is the dedicated Linux native-mesh first-base UBO.  It lies inside
    # the historical fallback-uniform block, so that one class explicitly
    # skips it instead of relying on today's title not reaching index 12.
    values = tuple(v for v in r if v != MESH_FIRST_BINDING)
    return values[seq(tag + desc, 0, len(values))]
