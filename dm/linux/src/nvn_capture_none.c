// Capture is optional; Linux uses explicit no-capture hooks for now.
#include "nvn_capture.h"

void nvn_cap_submit(int n, const uint64_t *handles) {
    (void)n;
    (void)handles;
}

void nvn_cap_present(const tex_t *t, const int crop[4]) {
    (void)t;
    (void)crop;
}

// The Metal renderer exposes a residency diagnostic through DM_CTL. Vulkan's
// memory tracker has no equivalent report yet.
void dm_gfx_restop(void) {}
