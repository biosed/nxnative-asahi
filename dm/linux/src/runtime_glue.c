// Optional subsystems not present in the first Linux runtime executable.
// Hibernation is an Apple/Android snapshot feature; ordinary Linux execution
// keeps it off while the native runtime and renderer are brought up.
#include "dm.h"
void dm_hib_exec_fixed(int argc, char **argv) { (void)argc; (void)argv; }
int dm_hib_restore_prepare(void) { return 0; }
void dm_hib_restore_main(void) { die("Linux hibernation restore is unavailable"); }

// The served-JIT tool mode is optional and has no configured image directory
// in the normal game run.  Keep its absence explicit until loader parity is
// merged from the current host.
const char *dm_jit_serve_dir(void) { return NULL; }
uint8_t *dm_jit_layout_base(void) { return NULL; }
void nro_load_alone(module_t *m) { (void)m; die("Linux served JIT modules are unavailable"); }
