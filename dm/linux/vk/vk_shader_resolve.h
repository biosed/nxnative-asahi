// Device-free selection of a translated shader triplet.  Ordinary stages live
// in DM_VK_SPV; native TaskEXT/MeshEXT artifacts remain in the separately
// generated DMSH_VK_MESH_EXT_DIR and need their exact job sidecar.
#pragma once
#include <stddef.h>
#include "vk_mesh.h"

typedef struct {
    char dir[1024], spv[1200], vkb[1200], job[1200];
    int stage, descriptor_set, mesh_ext;
    vkx_mesh_stage_meta mesh_meta;
} vkx_shader_resolved;

// wanted_stage < 0 accepts the stage named by the sidecar.  Native task/mesh
// stages are accepted only from mesh_dir and ordinary stages only from
// standard_dir, so a duplicate key cannot silently substitute one ABI for the
// other.  Returns zero on missing/mismatched sidecars.
int vkx_shader_resolve(const char *standard_dir, const char *mesh_dir,
                       const char key[17], int wanted_stage,
                       vkx_shader_resolved *out);

