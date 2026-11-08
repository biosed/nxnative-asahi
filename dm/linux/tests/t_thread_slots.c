#define _GNU_SOURCE
#include "thread_slots.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { THREADS = 8 };

static int64_t expected[DM_THREAD_SLOT_COUNT];
static pthread_barrier_t barrier;
static _Atomic unsigned checked;

static void fail(const char *message) {
    fprintf(stderr, "nxnative thread slots probe: %s\n", message);
    exit(1);
}

static void *worker(void *argument) {
    uintptr_t id = (uintptr_t)argument + 1;
    uintptr_t tp = dm_thread_pointer();
    if (!tp || (tp & 15)) fail("TPIDR_EL0 is null or not 16-byte aligned");

    for (unsigned slot = 0; slot < DM_THREAD_SLOT_COUNT; slot++) {
        int64_t offset = dm_thread_slot_offset(slot);
        if (offset != expected[slot]) fail("a TLS slot offset changed between threads");
        if (offset & 7) fail("a TLS slot is not pointer aligned");
        void **raw = (void **)((intptr_t)tp + offset);
        void *value = (void *)((id << 8) | slot | 1);
        if (dm_thread_slot_set(slot, value)) fail("slot write failed");
        if (*raw != dm_thread_slot_get(slot)) fail("raw and typed slot reads disagree");
        atomic_fetch_add_explicit(&checked, 1, memory_order_relaxed);
    }

    int result = pthread_barrier_wait(&barrier);
    if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD) fail("barrier wait failed");

    for (unsigned slot = 0; slot < DM_THREAD_SLOT_COUNT; slot++) {
        void *want = (void *)((id << 8) | slot | 1);
        if (dm_thread_slot_get(slot) != want) fail("another pthread changed this thread's slot");
        atomic_fetch_add_explicit(&checked, 1, memory_order_relaxed);
    }
    return NULL;
}

int main(void) {
    if (dm_thread_slots_init()) fail("slot initialization failed");
    uintptr_t tp = dm_thread_pointer();
    for (unsigned slot = 0; slot < DM_THREAD_SLOT_COUNT; slot++) {
        expected[slot] = dm_thread_slot_offset(slot);
        if (expected[slot] & 7) fail("main-thread slot is not pointer aligned");
        if ((intptr_t)tp + expected[slot] == 0) fail("main-thread slot address is null");
    }

    if (pthread_barrier_init(&barrier, NULL, THREADS)) fail("barrier init failed");
    pthread_t threads[THREADS];
    for (uintptr_t i = 0; i < THREADS; i++)
        if (pthread_create(&threads[i], NULL, worker, (void *)i)) fail("pthread_create failed");
    for (unsigned i = 0; i < THREADS; i++)
        if (pthread_join(threads[i], NULL)) fail("pthread_join failed");
    pthread_barrier_destroy(&barrier);
    dm_thread_slots_clear();

    unsigned total = atomic_load_explicit(&checked, memory_order_relaxed);
    if (total != THREADS * DM_THREAD_SLOT_COUNT * 2) fail("check count mismatch");
    printf("{\"status\":\"pass\",\"probe\":\"nxnative-thread-slots-linux\","
           "\"threads\":%u,\"slots\":%u,\"checks\":%u,\"offset0\":%lld}\n",
           THREADS, DM_THREAD_SLOT_COUNT, total, (long long)expected[0]);
    return 0;
}
