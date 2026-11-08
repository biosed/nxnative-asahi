#pragma once

#include <stdint.h>

enum {
    DM_SLOT_K18,
    DM_SLOT_KTLS,
    DM_SLOT_KTLR,
    DM_SLOT_KGATE,
    DM_THREAD_SLOT_COUNT
};

typedef unsigned dm_thread_slot_t;

int dm_thread_slots_init(void);
void dm_thread_slots_clear(void);
uintptr_t dm_thread_pointer(void);
int64_t dm_thread_slot_offset(dm_thread_slot_t slot);
void *dm_thread_slot_get(dm_thread_slot_t slot);
int dm_thread_slot_set(dm_thread_slot_t slot, void *value);
