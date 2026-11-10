#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dm.h"
#include "nvn_obj.h"
#include "nvn_capture.h"
#include "nvn_mtl.h"

pthread_mutex_t g_sub_mu = PTHREAD_MUTEX_INITIALIZER;
__thread _Atomic int dm_host_work;
int pthread_set_qos_class_self_np(qos_class_t qos, int relpri) { (void)qos; (void)relpri; return 0; }
static _Atomic uint64_t g_sub_wall_ns, g_sub_cpu_ns, g_sub_instr;
static _Atomic uint64_t g_cum_sub_instr, g_cum_sub_cpu_ns;
static _Atomic uint64_t g_cum_sub_cyc, g_cum_sub_sys_ns, g_cum_sub_wall_ns;
static _Atomic uint64_t g_nvn_ns;

uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static uint64_t thr_sys_ns(void) { return 0; }
static void thr_counts(uint64_t *instr, uint64_t *cycles) { *instr = *cycles = 0; }
static void gfx_enter(void) {}
static void gfx_leave(void) {}
uint64_t dm_nvn_frame_count(void) { return 0; }

void nvn_cap_queue(uint64_t serial, int waits) { (void)serial; (void)waits; }
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }

// This fixture uses only pointer-free marker records, event signals, waits and
// render targets. Exercise the same object callbacks as the production walk for
// the one descriptor-bearing record in that set.
int cap_rec_fields(uint32_t op, uint8_t *payload, uint32_t size, const cap_ops *ops, void *ctx) {
    if (op == OP_SET_RT) {
        if (size < sizeof(op_hdr) + sizeof(p_set_rt)) return 0;
        p_set_rt *r = (p_set_rt *)payload;
        if (r->n < 0 || r->n > 8) return 0;
        for (int i = 0; i < r->n; i++) if (!ops->obj(ctx, CAP_K_TEX, (uint64_t *)&r->color[i], payload, size - 8)) return 0;
        return ops->obj(ctx, CAP_K_TEX, (uint64_t *)&r->depth, payload, size - 8);
    }
    return op == OP_BARRIER || op == OP_SIGNAL_EVENT;
}

static void nvn_prog_snapshot(prog_t *copy, const prog_t *source) { *copy = *source; }
static void nvn_prog_snapshot_drop(prog_t *copy) { (void)copy; }
void mtl_tex_pin(const tex_t *t) { (void)t; }
void mtl_tex_unpin(uint64_t serial) { (void)serial; }

typedef struct { volatile uint32_t *word; unsigned mode; uint32_t value; uint64_t fence; int applied; } fake_signal;
static pthread_mutex_t fake_mu = PTHREAD_MUTEX_INITIALIZER;
static fake_signal fake_pending[64], fake_current[16];
static unsigned fake_pending_n, fake_current_n;
static _Atomic uint64_t fake_fence, fake_done, fake_commits;
static _Atomic unsigned marker_count;
static uint64_t marker_rt[32], current_rt;

void mtl_prepare(void) {}
void mtl_sub_begin(void) { fake_current_n = 0; }
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_set_queue_origin_lower_left(int origin) { (void)origin; }
void mtl_set_frame(uint64_t frame) { (void)frame; }
void mtl_sub_index(const uint64_t *handles, int n) { (void)handles; (void)n; }
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }
void nvn_ft_poll(void) {}

void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + sizeof h;
        if (h.op == OP_END) return;
        if (h.op == OP_SET_RT) {
            const p_set_rt *r = (const p_set_rt *)a;
            current_rt = r->n > 0 && r->color[0] ? r->color[0]->serial : 0;
        } else if (h.op == OP_SIGNAL_EVENT) {
            fake_signal *s = &fake_current[fake_current_n++];
            memcpy(&s->word, a, 8); memcpy(&s->mode, a + 8, 4); memcpy(&s->value, a + 12, 4);
        } else if (h.op == OP_BARRIER) {
            unsigned i = atomic_load_explicit(&marker_count, memory_order_relaxed);
            marker_rt[i] = current_rt;
            atomic_store_explicit(&marker_count, i + 1, memory_order_release);
        }
        p += h.size;
    }
}

uint64_t mtl_sub_commit(void) {
    uint64_t fence = atomic_fetch_add_explicit(&fake_fence, 1, memory_order_relaxed) + 1;
    pthread_mutex_lock(&fake_mu);
    for (unsigned i = 0; i < fake_current_n; i++) {
        fake_current[i].fence = fence;
        fake_pending[fake_pending_n++] = fake_current[i];
    }
    pthread_mutex_unlock(&fake_mu);
    atomic_fetch_add_explicit(&fake_commits, 1, memory_order_release);
    return fence;
}
uint64_t mtl_completed_fence(void) { return atomic_load_explicit(&fake_done, memory_order_acquire); }
uint64_t mtl_last_fence(void) { return atomic_load_explicit(&fake_fence, memory_order_acquire); }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    uint64_t start = now_ns();
    while (mtl_completed_fence() < fence) {
        if (timeout_ns != UINT64_MAX && now_ns() - start >= timeout_ns) return 0;
        struct timespec tick = {0, 1000000}; nanosleep(&tick, NULL);
    }
    return 1;
}
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) { (void)t; (void)period; (void)crop; return 0; }
void mtl_present_wait(uint64_t ticket) { (void)ticket; }

#include "nvn_queue.inc"

static int complete_one(uint64_t fence) {
    fake_signal signal = {0}; int found = 0;
    pthread_mutex_lock(&fake_mu);
    for (unsigned i = 0; i < fake_pending_n; i++) if (!fake_pending[i].applied && fake_pending[i].fence <= fence) {
        fake_pending[i].applied = 1; signal = fake_pending[i]; found = 1; break;
    }
    pthread_mutex_unlock(&fake_mu);
    if (found) nvn_event_publish_gpu(signal.word, signal.mode, signal.value, signal.fence);
    return found;
}
static void complete_to(uint64_t fence) {
    while (complete_one(fence)) {}
    atomic_store_explicit(&fake_done, fence, memory_order_release);
    nvn_queue_gpu_progress();
}

typedef struct { _Alignas(8) uint8_t data[1024]; size_t n; } recording;
static void record_op(recording *r, uint32_t op, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    op_hdr h = {op, (uint32_t)size};
    memcpy(r->data + r->n, &h, sizeof h);
    if (bytes) memcpy(r->data + r->n + sizeof h, payload, bytes);
    if (size > sizeof h + bytes) memset(r->data + r->n + sizeof h + bytes, 0, size - sizeof h - bytes);
    r->n += size;
}
static void record_end(recording *r) { record_op(r, OP_END, NULL, 0); }
static void record_signal(recording *r, volatile uint32_t *word, unsigned mode, uint32_t value) {
    struct { volatile uint32_t *word; uint32_t mode, value; } p = {word, mode, value};
    record_op(r, OP_SIGNAL_EVENT, &p, sizeof p);
}
static void record_wait(recording *r, volatile uint32_t *word, unsigned mode, uint32_t value) {
    struct { volatile uint32_t *word; uint32_t mode, value; } p = {word, mode, value};
    record_op(r, OP_WAIT_EVENT, &p, sizeof p);
}
static void record_marker(recording *r) { record_op(r, OP_BARRIER, NULL, 0); }
static void record_rt(recording *r, const tex_t *texture) {
    p_set_rt rt = {.n = 1, .color = {texture}};
    record_op(r, OP_SET_RT, &rt, sizeof rt);
}
static void submit(queue_t *q, recording *r) {
    uint64_t handle = (uintptr_t)r->data;
    nvn_queue_submit(q, 1, &handle);
}
static int wait_at_least(_Atomic uint64_t *value, uint64_t target, uint64_t timeout_ns) {
    uint64_t start = now_ns();
    while (atomic_load_explicit(value, memory_order_acquire) < target) {
        if (now_ns() - start >= timeout_ns) return 0;
        struct timespec tick = {0, 1000000}; nanosleep(&tick, NULL);
    }
    return 1;
}
static int wait_markers(unsigned target, uint64_t timeout_ns) {
    uint64_t start = now_ns();
    while (atomic_load_explicit(&marker_count, memory_order_acquire) < target) {
        if (now_ns() - start >= timeout_ns) return 0;
        struct timespec tick = {0, 1000000}; nanosleep(&tick, NULL);
    }
    return 1;
}

static int failures;
#define CHECK(c, message) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", message); failures++; } } while (0)

int main(void) {
    queue_t producer = {.serial = 1}, consumer = {.serial = 2};
    volatile uint32_t event = 0;

    uint64_t commits = atomic_load(&fake_commits); unsigned markers = atomic_load(&marker_count);
    recording write = {0}, wait = {0};
    record_signal(&write, &event, 0, 1); record_end(&write); submit(&producer, &write);
    CHECK(wait_at_least(&fake_commits, commits + 1, 1000000000ull), "WRITE producer encodes");
    uint64_t write_fence = atomic_load(&fake_fence);
    record_wait(&wait, &event, 0, 1); record_marker(&wait); record_end(&wait); submit(&consumer, &wait);
    CHECK(wait_markers(markers + 1, 1000000000ull), "wait encodes behind a queued exact WRITE");
    CHECK(event == 0, "queued WRITE need not land on the CPU before its consumer encodes");
    complete_to(atomic_load(&fake_fence));
    CHECK(event == 1 && write_fence != 0, "queued WRITE publishes on completion");

    event = 0; commits = atomic_load(&fake_commits); markers = atomic_load(&marker_count);
    recording add = {0}, add_wait = {0};
    record_signal(&add, &event, 1, 1); record_end(&add); submit(&producer, &add);
    CHECK(wait_at_least(&fake_commits, commits + 1, 1000000000ull), "ADD producer encodes");
    uint64_t add_fence = atomic_load(&fake_fence);
    record_wait(&add_wait, &event, 0, 1); record_marker(&add_wait); record_end(&add_wait); submit(&consumer, &add_wait);
    struct timespec pause = {0, 50000000}; nanosleep(&pause, NULL);
    CHECK(atomic_load(&marker_count) == markers, "wait stays parked while an ADD result is unknown");
    complete_to(add_fence);
    CHECK(wait_markers(markers + 1, 1000000000ull), "ADD completion releases its waiter");
    complete_to(atomic_load(&fake_fence));

    sync_t never = {0}; markers = atomic_load(&marker_count);
    nvn_queue_wait_sync(&consumer, &never);
    recording after_never = {0}; record_marker(&after_never); record_end(&after_never); submit(&consumer, &after_never);
    CHECK(wait_markers(markers + 1, 1000000000ull), "wait on a never-fenced sync passes");
    complete_to(atomic_load(&fake_fence));

    volatile uint32_t first = 0, second = 0; commits = atomic_load(&fake_commits); markers = atomic_load(&marker_count);
    recording signals = {0}; record_signal(&signals, &first, 0, 3); record_marker(&signals); record_signal(&signals, &second, 0, 7); record_end(&signals); submit(&producer, &signals);
    CHECK(wait_markers(markers + 1, 1000000000ull) && wait_at_least(&fake_commits, commits + 1, 1000000000ull), "multi-signal recording encodes");
    CHECK(atomic_load(&fake_commits) == commits + 1, "event signals do not split a physical submission");
    complete_to(atomic_load(&fake_fence));
    CHECK(first == 3 && second == 7, "every signal in the recording publishes");

    // Completion publication is a batch: landing an earlier event must not
    // retire the queued-write guard of a later event at the same fence.
    volatile uint32_t earlier = 0, reset = 1; markers = atomic_load(&marker_count); commits = atomic_load(&fake_commits);
    recording resets = {0}, reset_wait = {0};
    record_signal(&resets, &earlier, 0, 9); record_signal(&resets, &reset, 0, 0); record_end(&resets); submit(&producer, &resets);
    CHECK(wait_at_least(&fake_commits, commits + 1, 1000000000ull), "same-fence event batch encodes");
    uint64_t reset_fence = atomic_load(&fake_fence);
    record_wait(&reset_wait, &reset, 0, 1); record_marker(&reset_wait); record_end(&reset_wait); submit(&consumer, &reset_wait);
    CHECK(complete_one(reset_fence), "first event in the completion batch publishes");
    struct timespec batch_pause = {0, 50000000}; nanosleep(&batch_pause, NULL);
    CHECK(atomic_load(&marker_count) == markers, "earlier publication keeps the later event's queued reset guarded");
    complete_to(reset_fence);
    CHECK(reset == 0 && atomic_load(&marker_count) == markers, "the queued reset does not satisfy the stale-value wait");
    nvn_event_publish(&reset, 0, 1);
    CHECK(wait_markers(markers + 1, 1000000000ull), "a later CPU signal releases the reset event wait");
    complete_to(atomic_load(&fake_fence));

    tex_t red = {.serial = 101}, blue = {.serial = 202}; markers = atomic_load(&marker_count);
    recording red_set = {0}, blue_set = {0}, red_again = {0};
    record_rt(&red_set, &red); record_marker(&red_set); record_end(&red_set); submit(&producer, &red_set);
    CHECK(wait_markers(markers + 1, 1000000000ull), "first queue binds its render target");
    record_rt(&blue_set, &blue); record_marker(&blue_set); record_end(&blue_set); submit(&consumer, &blue_set);
    CHECK(wait_markers(markers + 2, 1000000000ull), "second queue binds its render target");
    record_marker(&red_again); record_end(&red_again); submit(&producer, &red_again);
    CHECK(wait_markers(markers + 3, 1000000000ull), "first queue resumes without rebinding explicitly");
    CHECK(marker_rt[markers] == 101 && marker_rt[markers + 1] == 202 && marker_rt[markers + 2] == 101,
          "logical queue switches restore persistent render targets");
    complete_to(atomic_load(&fake_fence));

    printf("t_nvn_queue: %s (%d failures)\n", failures ? "FAILED" : "passed", failures);
    return failures != 0;
}
