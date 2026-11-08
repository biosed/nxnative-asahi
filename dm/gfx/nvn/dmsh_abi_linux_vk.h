// Linux canonical-SPIR-V ABI.  This is deliberately separate from
// DMSH_MESH_FIRST_SLOT (Metal buffer slot 29) and dmsh_abi_vk.h (the packed
// Android library): Linux forms one union descriptor layout from .vkb files.
#pragma once

#define DMSH_LINUX_VK_MESH_FIRST_BINDING 512  // owner-stage u32 NVN mesh first base, kind mesh_first
#define DMSH_LINUX_VK_BINDING_LIMIT 1024

