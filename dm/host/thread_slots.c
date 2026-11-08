#include "thread_slots.h"

#include <pthread.h>
#include <stdlib.h>

static void check_slot(dm_thread_slot_t slot) {
    if (slot >= DM_THREAD_SLOT_COUNT) abort();
}

#if defined(__APPLE__) && defined(__aarch64__)

static pthread_key_t keys[DM_THREAD_SLOT_COUNT];

int dm_thread_slots_init(void) {
    for (unsigned slot = 0; slot < DM_THREAD_SLOT_COUNT; slot++) {
        if (pthread_key_create(&keys[slot], NULL) || keys[slot] == 0) {
            if (keys[slot] == 0) pthread_key_delete(keys[slot]);
            while (slot) pthread_key_delete(keys[--slot]);
            return -1;
        }
    }
    return 0;
}

uintptr_t dm_thread_pointer(void) {
    uintptr_t pointer;
    __asm__ volatile("mrs %0, tpidrro_el0" : "=r"(pointer));
    return pointer;
}

int64_t dm_thread_slot_offset(dm_thread_slot_t slot) {
    check_slot(slot);
    return (int64_t)keys[slot] * 8;
}

void *dm_thread_slot_get(dm_thread_slot_t slot) {
    check_slot(slot);
    return pthread_getspecific(keys[slot]);
}

int dm_thread_slot_set(dm_thread_slot_t slot, void *value) {
    check_slot(slot);
    return pthread_setspecific(keys[slot], value);
}

#elif defined(__linux__) && defined(__aarch64__)

static _Thread_local void *slots[DM_THREAD_SLOT_COUNT]
    __attribute__((tls_model("initial-exec"), aligned(16)));

int dm_thread_slots_init(void) { return 0; }

uintptr_t dm_thread_pointer(void) {
    uintptr_t pointer;
    __asm__ volatile("mrs %0, tpidr_el0" : "=r"(pointer));
    return pointer;
}

int64_t dm_thread_slot_offset(dm_thread_slot_t slot) {
    check_slot(slot);
    return (int64_t)((uintptr_t)&slots[slot] - dm_thread_pointer());
}

void *dm_thread_slot_get(dm_thread_slot_t slot) {
    check_slot(slot);
    return slots[slot];
}

int dm_thread_slot_set(dm_thread_slot_t slot, void *value) {
    check_slot(slot);
    slots[slot] = value;
    return 0;
}

#else
#error "nxnative thread slots require Darwin/AArch64 or Linux/AArch64"
#endif

void dm_thread_slots_clear(void) {
    for (unsigned slot = 0; slot < DM_THREAD_SLOT_COUNT; slot++)
        if (dm_thread_slot_set(slot, NULL)) abort();
}
