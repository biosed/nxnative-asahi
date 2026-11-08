// Guest focus state selected for each SDL window lifecycle event. A zero state
// means that the event must not change oe focus (DM_FOCUS_KEEP parity with the
// Mac front end); callers still perform the event's input/audio/GPU actions.
#pragma once

enum front_focus_event {
    FRONT_FOCUS_LOST,
    FRONT_FOCUS_GAINED,
    FRONT_FOCUS_MINIMIZED,
    FRONT_FOCUS_RESTORED,
};

static inline int front_focus_state(enum front_focus_event ev, int keep, int minimized) {
    switch (ev) {
        case FRONT_FOCUS_LOST: return keep || minimized ? 0 : 2;
        case FRONT_FOCUS_GAINED: return 1;
        case FRONT_FOCUS_MINIMIZED: return 3;
        case FRONT_FOCUS_RESTORED: return 2;
    }
    return 0;
}
