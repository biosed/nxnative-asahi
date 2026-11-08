// Linux SDL focus policy without SDL or a window. DM_FOCUS_KEEP suppresses
// only ordinary focus loss; input release remains the caller's unconditional
// action, and minimize/restore keep their background lifecycle states.
#include <stdio.h>
#include "front_focus.h"

static int bad;
static void check(int c, const char *what) {
    if (!c) { printf("FAIL: %s\n", what); bad++; }
}

int main(void) {
    check(front_focus_state(FRONT_FOCUS_LOST, 0, 0) == 2, "focus loss defaults to out-of-focus");
    check(front_focus_state(FRONT_FOCUS_LOST, 1, 0) == 0, "focus keep suppresses only ordinary focus loss");
    check(front_focus_state(FRONT_FOCUS_LOST, 0, 1) == 0, "focus loss cannot replace minimized background state");
    check(front_focus_state(FRONT_FOCUS_LOST, 1, 1) == 0, "focus keep leaves minimized background state alone");
    check(front_focus_state(FRONT_FOCUS_GAINED, 1, 0) == 1, "focus gain returns to in-focus");
    check(front_focus_state(FRONT_FOCUS_MINIMIZED, 1, 1) == 3, "focus keep does not mask minimize/background");
    check(front_focus_state(FRONT_FOCUS_RESTORED, 1, 0) == 2, "restore remains out-of-focus until focus gain");
    printf("t_focus: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
