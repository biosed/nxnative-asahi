// Device-free unit test for first-match range lookup caching and epoch
// invalidation in nvn_dirty.c (upstream PR #1227 semantics).
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include "dm.h"
#include "nvn_dirty.h"

static int (*g_fault_fn)(void *, int, void *);
static void (*g_prep_fn)(void *, size_t);

void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) {
    g_fault_fn = fn;
}

void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len)) {
    g_prep_fn = fn;
}

uint64_t mach_absolute_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

kern_return_t mach_timebase_info(mach_timebase_info_t i) {
    i->numer = i->denom = 1;
    return KERN_SUCCESS;
}

static void segv_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig;
    if (g_fault_fn && g_fault_fn(si->si_addr, 1, uc))
        return;
    fprintf(stderr, "unhandled SIGSEGV at %p\n", si->si_addr);
    abort();
}

static void install_fault_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = segv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, NULL) != 0) {
        perror("sigaction");
        exit(2);
    }
}

static int bad;
static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        bad++;
    }
}

enum { GUEST_PAGE_SIZE = 0x1000, CACHE_PROBE_SIZE = 16 };
enum overlap_layout {
    EARLIER_RANGE_ENDS_SOONER,
    EARLIER_RANGE_STARTS_LATER,
};

static size_t subpage_step(size_t pg) {
    return pg > 0x1000 ? 0x1000 : pg / 2;
}

static void *test_shared_page_cache(void *arg) {
    enum overlap_layout layout = (enum overlap_layout)(intptr_t)arg;
    size_t pg = (size_t)getpagesize();
    size_t step = subpage_step(pg);
    size_t map_sz = 4 * pg;
    uint8_t *m = mmap(NULL, map_sz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(2); }

    uint8_t *first_pool, *second_pool;
    const void *second_only;
    size_t first_sz, second_sz;
    if (layout == EARLIER_RANGE_STARTS_LATER) {
        first_pool = m + 2 * pg - step;
        first_sz = step;
        second_pool = m;
        second_sz = 2 * pg - step;
        second_only = m;
    } else {
        first_pool = m;
        first_sz = pg - step;
        second_pool = m + pg - step;
        second_sz = 2 * pg;
        second_only = m + pg;
    }

    nvn_dirty_add_range(first_pool, first_sz);
    const void *first_range = nvn_dirty_span(first_pool, CACHE_PROBE_SIZE);
    nvn_dirty_add_range(second_pool, second_sz);
    uint64_t stamp = nvn_dirty_now();
    nvn_dirty_protect(first_pool, CACHE_PROBE_SIZE);
    const void *other_range = nvn_dirty_span(second_only, pg);
    ((volatile uint8_t *)first_pool)[0] = 7;

    check(first_range != NULL && other_range != NULL && first_range != other_range,
          "warmed distinct containing descriptor for second pool");
    check(nvn_dirty_since(first_pool, CACHE_PROBE_SIZE, stamp) == 1,
          "cached lookup detects write after warming overlapping descriptor");
    check(nvn_dirty_span(first_pool, CACHE_PROBE_SIZE) == first_range,
          "cached lookup preserves first-match range");

    nvn_dirty_remove_range(first_pool);
    nvn_dirty_remove_range(second_pool);
    munmap(m, map_sz);
    return NULL;
}

typedef struct {
    const void *probe_address;
    const void *expected_range;
    _Atomic int phase;
    int matched;
} cache_reuse_test_t;

static void *cache_reuse_reader(void *arg) {
    cache_reuse_test_t *t = arg;
    nvn_dirty_span(t->probe_address, CACHE_PROBE_SIZE);
    atomic_store(&t->phase, 1);
    while (atomic_load(&t->phase) != 2)
        usleep(100);
    t->matched = (nvn_dirty_span(t->probe_address, CACHE_PROBE_SIZE) == t->expected_range);
    return NULL;
}

static void test_cache_slot_reuse(void) {
    size_t pg = (size_t)getpagesize();
    size_t step = subpage_step(pg);
    size_t map_sz = 4 * pg;
    uint8_t *m = mmap(NULL, map_sz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(2); }

    uint8_t *later_pool = m + 2 * pg;
    nvn_dirty_add_range(m, pg);
    nvn_dirty_add_range(later_pool, 2 * pg);
    const void *first_slot = nvn_dirty_span(m, CACHE_PROBE_SIZE);
    const void *later_slot = nvn_dirty_span(later_pool, 2 * pg);
    nvn_dirty_remove_range(m);

    uint8_t *replacement_pool = later_pool + step;
    cache_reuse_test_t test = {
        .probe_address = replacement_pool,
        .expected_range = first_slot,
        .phase = 0,
    };
    pthread_t reader;
    if (pthread_create(&reader, NULL, cache_reuse_reader, &test)) exit(2);
    while (atomic_load(&test.phase) != 1)
        usleep(100);

    nvn_dirty_add_range(replacement_pool, pg - step);
    atomic_store(&test.phase, 2);
    pthread_join(reader, NULL);
    check(test.matched == 1, "slot reuse invalidates another thread's cached selection");
    check(nvn_dirty_span(replacement_pool, CACHE_PROBE_SIZE) == first_slot,
          "reused earlier slot is canonical first match");

    nvn_dirty_remove_range(replacement_pool);
    check(nvn_dirty_span(replacement_pool, CACHE_PROBE_SIZE) == later_slot,
          "retiring cached match falls back to remaining range");
    nvn_dirty_remove_range(later_pool);
    munmap(m, map_sz);
}

int main(void) {
    install_fault_handler();
    check(nvn_dirty_tracked((const void *)0x1000, 0) == 0, "zero-length lookup is untracked");
    check(nvn_dirty_tracked((const void *)UINTPTR_MAX, 16) == 0, "overflowing span lookup fails closed");

    for (int layout = EARLIER_RANGE_ENDS_SOONER; layout <= EARLIER_RANGE_STARTS_LATER; layout++) {
        pthread_t th;
        if (pthread_create(&th, NULL, test_shared_page_cache, (void *)(intptr_t)layout)) exit(2);
        pthread_join(th, NULL);
    }
    test_cache_slot_reuse();
    printf("t_nvn_dirty: %s (%d failures)\n", bad ? "FAILED" : "passed", bad);
    return bad != 0;
}
