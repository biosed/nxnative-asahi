// Logical NVN queue scheduler. Ready continuations share the renderer's one
// ordered Vulkan queue; an unresolved dependency parks only its logical queue.
// Based on nxnative ea0c98df (MIT), with its queue follow-ups through 25c2546e.
#include "nvn_capture.h"
#include "nvn_hmap.h"

typedef struct nq_receipt { unsigned refs, published; uint64_t fence, present; } nq_receipt;
typedef struct nq_signal { volatile uint32_t *event; unsigned mode; uint32_t value; } nq_signal;
typedef struct nq_owned nq_owned;
typedef struct nq_node {
    struct nq_node *next;
    uint8_t *bytes; size_t length, capacity;
    unsigned kind, mode, closed, waiting; volatile uint32_t *event; uint32_t value;
    uint64_t wait_fence, event_predecessor, parked_at; unsigned wait_kind; const sync_t *sync_source; nq_receipt *receipt;
    nq_signal *signals; unsigned signal_count, signal_capacity;
    nq_owned *texture; double period; int crop[4];
} nq_node;
typedef struct nq_ref { struct nq_ref *next; nq_owned *object; } nq_ref;
typedef struct nq_job {
    struct nq_job *next; nq_node *first, *last; nq_receipt *receipt;
    nq_ref *owned; nvn_hmap objects; uint64_t frame;
} nq_job;
typedef struct { uint64_t identity, kind; } nq_object_key;
struct nq_owned { unsigned refs, kind; nq_object_key key; uint64_t data[]; };

// Protected by g_sub_mu. A serial identifies immutable initialization metadata,
// so jobs share one owned copy and persistent queue bindings retain that copy.
static nvn_hmap nq_object_cache = {.klen = sizeof(nq_object_key)};
static void nq_owned_put(nq_owned *o) {
    if (!o || --o->refs) return;
    nvn_hm_del(&nq_object_cache, &o->key);
    if (o->kind == CAP_K_PROG) nvn_prog_snapshot_drop((prog_t *)o->data);
    else if (o->kind == CAP_K_TEX) mtl_tex_unpin(((const tex_t *)o->data)->serial);
    free(o);
}
static nq_owned *nq_owned_get(nq_job *j, unsigned kind, const void *source) {
    const uint64_t serial = kind == CAP_K_PROG ? ((const prog_t *)source)->serial : ((const tex_t *)source)->serial;
    nq_object_key key = {serial ? serial : (uint64_t)(uintptr_t)source, kind};
    nq_owned *o = nvn_hm_get(&j->objects, &key);
    if (o) return o;
    o = nvn_hm_get(&nq_object_cache, &key);
    if (!o) {
        size_t bytes = kind == CAP_K_PROG ? sizeof(prog_t) : sizeof(tex_t);
        o = calloc(1, sizeof *o + bytes); if (!o) die("queue descriptor allocation failed");
        o->kind = kind; o->key = key;
        if (kind == CAP_K_PROG) nvn_prog_snapshot((prog_t *)o->data, source);
        else { memcpy(o->data, source, bytes); mtl_tex_pin((const tex_t *)o->data); }
        nvn_hm_put(&nq_object_cache, &key, o);
    }
    o->refs++;
    nq_ref *r = malloc(sizeof *r); if (!r) die("queue descriptor reference allocation failed");
    r->object = o; r->next = j->owned; j->owned = r;
    nvn_hm_put(&j->objects, &key, o);
    return o;
}
static void nq_job_free(nq_job *j) {
    for (nq_ref *r = j->owned, *next; r; r = next) { next = r->next; nq_owned_put(r->object); free(r); }
    free(j->objects.keys); free(j->objects.vals); free(j);
}
static int nq_snapshot_address(void *ctx, uint64_t *field, uint64_t len, int writable) {
    (void)ctx; (void)field; (void)len; (void)writable; return 1;
}
static int nq_snapshot_object(void *ctx, int kind, uint64_t *field, uint8_t *payload, uint32_t size) {
    (void)payload; (void)size;
    if (!*field || (kind != CAP_K_PROG && kind != CAP_K_TEX)) return 1;
    nq_owned *o = nq_owned_get(ctx, (unsigned)kind, (const void *)(uintptr_t)*field);
    *field = (uintptr_t)o->data; return 1;
}
static const cap_ops nq_snapshot_ops = {nq_snapshot_address, nq_snapshot_object};

typedef struct nq_queue {
    struct nq_queue *next; uint64_t serial, physical, present; int origin;
    nq_job *first, *last; nq_receipt *tail;
    nq_owned *program[32], *target[9];
    p_set_rt rt;
} nq_queue;
static pthread_mutex_t nq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t nq_cv = PTHREAD_COND_INITIALIZER;
static pthread_once_t nq_once = PTHREAD_ONCE_INIT;
static nq_queue *nq_queues;

static _Atomic uint64_t nq_stats[NVN_QUEUE_STAT_COUNT];
static pthread_once_t nq_mode_once = PTHREAD_ONCE_INIT;
static int nq_scheduler = 1;
static void nq_mode_init(void) {
    const char *e = getenv("DM_NVN_QUEUE_SCHED");
    nq_scheduler = !(e && !strcmp(e, "0"));
    if (!nq_scheduler) LOG("[nvn] DM_NVN_QUEUE_SCHED=0: logical queue waits are ignored (diagnostic old in-order replay)");
}
int nvn_queue_scheduler_enabled(void) { pthread_once(&nq_mode_once, nq_mode_init); return nq_scheduler; }
void nvn_queue_stats(uint64_t out[NVN_QUEUE_STAT_COUNT]) {
    for (int i = 0; i < NVN_QUEUE_STAT_COUNT; i++) out[i] = atomic_load_explicit(&nq_stats[i], memory_order_relaxed);
}
static void nq_stat(unsigned stat) { atomic_fetch_add_explicit(&nq_stats[stat], 1, memory_order_relaxed); }

typedef struct nq_event_write {
    struct nq_event_write *next; volatile uint32_t *word; uint64_t fence;
    uint32_t value; int exact;
} nq_event_write;
static nq_event_write *nq_event_writes;
static uint64_t nq_completed;
static nq_event_write *nq_event_write_of(volatile uint32_t *word) {
    nq_event_write *w = nq_event_writes; while (w && w->word != word) w = w->next; return w;
}
static uint64_t nq_event_pending(volatile uint32_t *word) {
    nq_event_write *w = nq_event_write_of(word); return w ? w->fence : 0;
}
static void nq_event_issued(volatile uint32_t *word, uint64_t fence, unsigned mode, uint32_t value) {
    nq_event_write *w = nq_event_write_of(word);
    if (!w) { w = calloc(1, sizeof *w); if (!w) die("event ordering allocation failed"); w->word = word; w->next = nq_event_writes; nq_event_writes = w; }
    w->fence = fence; w->value = value; w->exact = mode == 0;
}
static void nq_event_reap(uint64_t done) {
    if (done > nq_completed) nq_completed = done;
    nq_event_write **p = &nq_event_writes;
    while (*p) { nq_event_write *w = *p; if (w->fence <= done) { *p = w->next; free(w); } else p = &w->next; }
}

enum { NQ_COMMANDS, NQ_EVENT, NQ_SYNC, NQ_FENCE, NQ_PRESENT };
static nq_receipt *nq_new(void) { nq_receipt *r = calloc(1, sizeof *r); if (!r) die("queue receipt allocation failed"); r->refs = 1; return r; }
static void nq_put(nq_receipt *r) { if (r && !--r->refs) free(r); }
static void nq_set_sync(sync_t *s, nq_receipt *r) {
    if (s->fenced == 3) nq_put((nq_receipt *)(uintptr_t)s->fence);
    r->refs++; s->fence = (uintptr_t)r; s->fenced = 3;
}
void nvn_sync_release(sync_t *s) {
    pthread_mutex_lock(&nq_mu); if (s->fenced == 3) nq_put((nq_receipt *)(uintptr_t)s->fence);
    s->fenced = 0; s->fence = 0; pthread_mutex_unlock(&nq_mu);
}
static nq_queue *nq_get(queue_t *q) {
    nq_queue *p = nq_queues; while (p && p->serial != q->serial) p = p->next;
    if (!p) { p = calloc(1, sizeof *p); if (!p) die("queue state allocation failed"); p->serial = q->serial; p->origin = q->origin_ll; p->next = nq_queues; nq_queues = p; }
    return p;
}
static void nq_publish(nq_receipt *r, uint64_t fence) { r->fence = fence; r->published = 1; pthread_cond_broadcast(&nq_cv); }
static int nq_predicate(uint32_t current, unsigned mode, uint32_t target) { return mode == 0 ? current == target : (uint32_t)(current - target) < 0x80000000u; }
static int nq_sync_encoded(const sync_t *s, uint64_t *fence) {
    if (s->fenced == 3) { nq_receipt *r = (nq_receipt *)(uintptr_t)s->fence; if (!r->published) return 0; *fence = r->fence; return 1; }
    if (!s->fenced) return 0; *fence = s->fence; return 1;
}
void nvn_queue_gpu_progress(void) {
    uint64_t done = mtl_completed_fence();
    pthread_mutex_lock(&nq_mu); nq_event_reap(done); pthread_cond_broadcast(&nq_cv); pthread_mutex_unlock(&nq_mu);
}
static void nq_event_publish(volatile uint32_t *word, unsigned mode, uint32_t value, uint64_t gpu_fence) {
    pthread_mutex_lock(&nq_mu);
    if (mode == 1) atomic_fetch_add_explicit((_Atomic uint32_t *)word, value, memory_order_release);
    else atomic_store_explicit((_Atomic uint32_t *)word, value, memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);
    // Latch waits already reached: a later signal must not erase a release.
    for (nq_queue *q = nq_queues; q; q = q->next) if (q->first && q->first->first) {
        nq_node *n = q->first->first;
        if (n->kind == NQ_EVENT && n->waiting && n->event == word &&
            (gpu_fence ? gpu_fence >= n->event_predecessor : nq_event_pending(word) <= nq_completed) &&
            nq_predicate(atomic_load_explicit((_Atomic uint32_t *)word, memory_order_acquire), n->mode, n->value)) {
            n->event = NULL; n->wait_fence = gpu_fence;
            nq_stat(gpu_fence ? NVN_QUEUE_STAT_COMPLETION_RELEASES : NVN_QUEUE_STAT_CPU_RELEASES);
        }
    }
    pthread_cond_broadcast(&nq_cv); pthread_mutex_unlock(&nq_mu);
}
void nvn_event_publish(volatile uint32_t *word, unsigned mode, uint32_t value) { nq_event_publish(word, mode, value, 0); }
void nvn_event_publish_gpu(volatile uint32_t *word, unsigned mode, uint32_t value, uint64_t fence) { nq_event_publish(word, mode, value, fence); }
static void nq_snapshot_sync(nq_node *n, const sync_t *s) {
    n->wait_kind = s->fenced; n->wait_fence = s->fence;
    if (s->fenced == 3) { n->receipt = (nq_receipt *)(uintptr_t)s->fence; n->receipt->refs++; }
}

typedef struct { uint64_t wall, cpu, instr, cycles, sys; } nq_cost;
static nq_cost nq_cost_begin(void) {
    nq_cost c = {.wall = now_ns(), .cpu = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID), .sys = thr_sys_ns()};
    thr_counts(&c.instr, &c.cycles); return c;
}
static void nq_cost_end(nq_cost c) {
    uint64_t dw = now_ns() - c.wall, dc = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - c.cpu, ins, cyc; thr_counts(&ins, &cyc);
    atomic_fetch_add(&g_nvn_ns, dw); atomic_fetch_add(&g_sub_wall_ns, dw); atomic_fetch_add(&g_cum_sub_wall_ns, dw);
    atomic_fetch_add(&g_sub_cpu_ns, dc); atomic_fetch_add(&g_cum_sub_cpu_ns, dc);
    atomic_fetch_add(&g_sub_instr, ins - c.instr); atomic_fetch_add(&g_cum_sub_instr, ins - c.instr);
    atomic_fetch_add(&g_cum_sub_cyc, cyc - c.cycles); atomic_fetch_add(&g_cum_sub_sys_ns, thr_sys_ns() - c.sys);
}
static void nq_owned_replace(nq_owned **slot, nq_owned *next) {
    if (*slot == next) return; if (next) next->refs++; nq_owned_put(*slot); *slot = next;
}
static nq_owned *nq_owned_from_data(const void *data) {
    return data ? (nq_owned *)((uint8_t *)data - offsetof(nq_owned, data)) : NULL;
}
// The Linux replay keeps program and render-target pointers across submissions.
// Retain the exact owned snapshots selected by this node until later bindings replace them.
static void nq_persist(nq_queue *q, const nq_node *n) {
    if (n->kind != NQ_COMMANDS) return;
    for (const uint8_t *p = n->bytes;;) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        if (h.op == OP_END) break;
        if (h.op == OP_BIND_PROGRAM) {
            const p_bind_program *b = (const p_bind_program *)a; nq_owned *o = nq_owned_from_data(b->prog);
            for (unsigned st = 0; st < 32; st++) if ((unsigned)b->stages >> st & 1u) nq_owned_replace(&q->program[st], o);
        } else if (h.op == OP_SET_RT) {
            const p_set_rt *r = (const p_set_rt *)a;
            for (int i = 0; i < 8; i++) nq_owned_replace(&q->target[i], nq_owned_from_data(i < r->n ? r->color[i] : NULL));
            nq_owned_replace(&q->target[8], nq_owned_from_data(r->depth));
            q->rt = *r;
            for (int i = 0; i < 8; i++) q->rt.color[i] = q->target[i] ? (const tex_t *)q->target[i]->data : NULL;
            q->rt.depth = q->target[8] ? (const tex_t *)q->target[8]->data : NULL;
        }
        p += h.size;
    }
}
static void nq_persist_drop(nq_queue *q) {
    for (int i = 0; i < 32; i++) nq_owned_replace(&q->program[i], NULL);
    for (int i = 0; i < 9; i++) nq_owned_replace(&q->target[i], NULL);
    memset(&q->rt, 0, sizeof q->rt);
}
// The Vulkan draw state is already selected per logical queue. Render targets
// live in the replay walker instead, so restore their value snapshot whenever
// command replay switches queues. This emits no extra physical submission.
static void nq_restore_rt(const nq_queue *q) {
    struct { op_hdr set; p_set_rt rt; op_hdr end; } records = {
        .set = {OP_SET_RT, sizeof(op_hdr) + sizeof(p_set_rt)},
        .rt = q->rt,
        .end = {OP_END, sizeof(op_hdr)},
    };
    _Static_assert(sizeof records == 256, "queue render-target restore records changed size");
    mtl_exec((const uint8_t *)&records);
}

static void *nq_worker(void *unused) {
    (void)unused; pthread_setname_np(pthread_self(), "NVN queue replay");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t frame = UINT64_MAX, active_commands = UINT64_MAX;
    for (;;) {
        pthread_mutex_lock(&nq_mu); nq_queue *q = NULL; nq_node *n = NULL; nq_job *j = NULL;
        for (;;) {
        scan_again:;
            int mapped_event_wait = 0;
            for (q = nq_queues; q; q = q->next) {
                j = q->first; if (!j) continue; n = j->first;
                if (!n) {
                    nq_publish(j->receipt, q->physical); q->first = j->next; if (!q->first) q->last = NULL; nq_put(j->receipt);
                    pthread_mutex_unlock(&nq_mu); pthread_mutex_lock(&g_sub_mu); nq_job_free(j); pthread_mutex_unlock(&g_sub_mu); pthread_mutex_lock(&nq_mu); goto scan_again;
                }
                if (n->kind == NQ_EVENT && n->event) {
                    uint64_t predecessor = nq_event_pending(n->event);
                    if (!n->waiting) n->event_predecessor = predecessor;
                    // A queued WRITE has an exact value. This continuation can be encoded
                    // behind it on the one physical queue without idling for completion.
                    nq_event_write *queued = predecessor > nq_completed ? nq_event_write_of(n->event) : NULL;
                    if (queued && queued->exact && nq_predicate(queued->value, n->mode, n->value)) {
                        n->event = NULL; n->wait_fence = predecessor; nq_stat(NVN_QUEUE_STAT_EXACT_QUEUED_RELEASES); break;
                    }
                    if (predecessor > nq_completed || !nq_predicate(atomic_load_explicit((_Atomic uint32_t *)n->event, memory_order_acquire), n->mode, n->value)) {
                        n->waiting = 1; if (!n->parked_at) n->parked_at = now_ns(); mapped_event_wait = 1; continue;
                    }
                    n->wait_fence = predecessor;
                }
                if (n->kind == NQ_SYNC && n->receipt && !n->receipt->published) {
                    if (!n->parked_at) n->parked_at = now_ns(); continue;
                }
                // A sync not yet fenced when submitted is a snapshot of nothing to wait for.
                break;
            }
            if (q) break;
            if (mapped_event_wait) { struct timespec tick = {0, 1000000}; pthread_cond_timedwait_relative_np(&nq_cv, &nq_mu, &tick); }
            else pthread_cond_wait(&nq_cv, &nq_mu);
        }
        j->first = n->next; if (!j->first) j->last = NULL;
        if (n->parked_at)
            atomic_fetch_add_explicit(&nq_stats[NVN_QUEUE_STAT_PARKED_NS], now_ns() - n->parked_at, memory_order_relaxed);
        // Round-robin ready queues prevent a continually fed producer starving another queue.
        if (q->next) { nq_queue **link = &nq_queues; while (*link != q) link = &(*link)->next; *link = q->next; nq_queue *tail = nq_queues; while (tail->next) tail = tail->next; tail->next = q; q->next = NULL; }
        if (n->kind == NQ_SYNC || n->kind == NQ_EVENT) {
            uint64_t f = n->receipt ? n->receipt->fence : n->wait_fence;
            nq_stat(f > q->physical ? NVN_QUEUE_STAT_CROSS_QUEUE_WAITS : NVN_QUEUE_STAT_LOCAL_WAITS);
            if (f > q->physical) q->physical = f;
        }
        if (n->kind == NQ_FENCE) nq_publish(n->receipt, q->physical);
        pthread_mutex_unlock(&nq_mu);

        uint64_t fence = 0, present = 0;
        if (n->kind == NQ_COMMANDS || n->kind == NQ_PRESENT) {
            dm_host_mutex_lock(&g_sub_mu);
            dm_host_work++; mtl_prepare(); dm_host_work--;
            gfx_enter(); nq_cost cost = nq_cost_begin();
            if (frame != j->frame) { frame = j->frame; mtl_set_frame(frame); }
            mtl_queue_activate(q->serial); mtl_set_queue_origin_lower_left(q->origin);
            if (n->kind == NQ_COMMANDS) {
                uint64_t handle = (uintptr_t)n->bytes;
                nvn_cap_submit(1, &handle);
                mtl_sub_begin(); nvn_ft_poll(); mtl_sub_index(&handle, 1);
                mtl_exec_rest(NULL, 0);
                if (active_commands != q->serial) {
                    if (active_commands != UINT64_MAX) nq_stat(NVN_QUEUE_STAT_QUEUE_SWITCHES);
                    nq_restore_rt(q); active_commands = q->serial;
                }
                mtl_exec(n->bytes); fence = mtl_sub_commit(); nq_stat(NVN_QUEUE_STAT_PHYSICAL_SUBMITS);
                nq_persist(q, n);
            } else {
                const tex_t *texture = (const tex_t *)n->texture->data;
                nvn_cap_present(texture, n->crop); present = mtl_present(texture, n->period, n->crop);
            }
            nq_cost_end(cost); gfx_leave(); pthread_mutex_unlock(&g_sub_mu);
        }
        pthread_mutex_lock(&nq_mu);
        if (n->kind == NQ_PRESENT) { q->present = present; j->receipt->present = present; }
        if (fence) { q->physical = fence; for (unsigned i = 0; i < n->signal_count; i++) nq_event_issued(n->signals[i].event, fence, n->signals[i].mode, n->signals[i].value); }
        if (n->receipt) nq_put(n->receipt);
        free(n->signals); free(n->bytes); free(n); pthread_cond_broadcast(&nq_cv); pthread_mutex_unlock(&nq_mu);
    }
    return NULL;
}
static void nq_start(void) { pthread_t t; if (pthread_create(&t, NULL, nq_worker, NULL)) die("NVN queue worker failed"); pthread_detach(t); }
static nq_node *nq_append(nq_job *j, unsigned kind) {
    nq_node *n = calloc(1, sizeof *n); if (!n) die("queue continuation allocation failed"); n->kind = kind;
    static const unsigned stat[] = {
        NVN_QUEUE_STAT_COMMAND_NODES, NVN_QUEUE_STAT_EVENT_NODES, NVN_QUEUE_STAT_SYNC_NODES,
        NVN_QUEUE_STAT_FENCE_NODES, NVN_QUEUE_STAT_PRESENT_NODES,
    };
    nq_stat(stat[kind]);
    if (j->last) j->last->next = n; else j->first = n; j->last = n; return n;
}
static void nq_bytes(nq_node *n, const void *bytes, size_t length) {
    if (n->length + length > n->capacity) {
        size_t capacity = (n->length + length + 4095) & ~(size_t)4095;
        uint8_t *copy = realloc(n->bytes, capacity); if (!copy) die("queue commands allocation failed"); n->bytes = copy; n->capacity = capacity;
    }
    memcpy(n->bytes + n->length, bytes, length); n->length += length;
}
static void nq_signal_add(nq_node *n, nq_signal signal) {
    if (n->signal_count == n->signal_capacity) {
        unsigned capacity = n->signal_capacity ? 2 * n->signal_capacity : 4;
        nq_signal *signals = realloc(n->signals, (size_t)capacity * sizeof *signals); if (!signals) die("queue signals allocation failed");
        n->signals = signals; n->signal_capacity = capacity;
    }
    n->signals[n->signal_count++] = signal;
}
// A signal followed by its own satisfied wait in one logical queue is an
// execution/memory dependency, not a reason to end the physical submission.
// Look at the last write to the word: an earlier exact write must not hide a
// later ADD whose result is unknown until the GPU completes.
static int nq_self_event(const nq_node *n, volatile uint32_t *event,
                         unsigned mode, uint32_t value) {
    if (!n || n->kind != NQ_COMMANDS || n->closed) return 0;
    for (unsigned i = n->signal_count; i-- > 0;) if (n->signals[i].event == event)
        return n->signals[i].mode == 0 && nq_predicate(n->signals[i].value, mode, value);
    return 0;
}
static void nq_self_barrier(nq_node *n) {
    // Separate physical submissions ordered every earlier GPU write before
    // the continuation. Keep that dependency while retaining both ranges in
    // one submit; nonzero bits are required by the Metal replay as well.
    struct { op_hdr h; int32_t bits; uint32_t pad; } b = {
        {OP_BARRIER, 16}, NVN_BARRIER_BITS_LARGE, 0
    };
    nq_bytes(n, &b, sizeof b);
}
static void nq_end(nq_job *j) {
    if (j->last && j->last->kind == NQ_COMMANDS && !j->last->closed) { op_hdr end = {OP_END, 8}; nq_bytes(j->last, &end, 8); j->last->closed = 1; }
}
static void nq_flatten(nq_job *j, const uint8_t *p, unsigned depth) {
    if (depth > 64) die("NVN command call nesting exceeds 64");
    for (;;) {
        op_hdr h; memcpy(&h, p, 8); if (h.size < 8 || h.size & 7) die("invalid NVN command size"); const uint8_t *a = p + 8;
        if (h.op == OP_END) return;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_CALL) { const uint8_t *called; memcpy(&called, a, 8); nq_flatten(j, called, depth + 1); p += h.size; continue; }
        if (h.op == OP_WAIT_EVENT) {
            volatile uint32_t *event; unsigned mode; uint32_t value;
            memcpy(&event, a, 8); memcpy(&mode, a + 8, 4); memcpy(&value, a + 12, 4);
            if (nq_self_event(j->last, event, mode, value)) {
                nq_self_barrier(j->last); nq_stat(NVN_QUEUE_STAT_SELF_EVENT_COALESCED);
                p += h.size; continue;
            }
            nq_end(j); nq_node *n = nq_append(j, NQ_EVENT);
            n->event = event; n->mode = mode; n->value = value;
        } else if (h.op == OP_WAIT_SYNC || h.op == OP_FENCE_SYNC) {
            nq_end(j); nq_node *n = nq_append(j, h.op == OP_WAIT_SYNC ? NQ_SYNC : NQ_FENCE);
            memcpy(&n->sync_source, a, 8); if (h.op == OP_FENCE_SYNC) n->receipt = nq_new();
        } else {
            nq_node *n = j->last; if (!n || n->kind != NQ_COMMANDS || n->closed) n = nq_append(j, NQ_COMMANDS);
            size_t offset = n->length; nq_bytes(n, p, h.size);
            if (!cap_rec_fields(h.op, n->bytes + offset + 8, h.size, &nq_snapshot_ops, j)) die("queue snapshot: unknown command %u", h.op);
            if (h.op == OP_SIGNAL_EVENT) { nq_signal signal; memcpy(&signal.event, a, 8); memcpy(&signal.mode, a + 8, 4); memcpy(&signal.value, a + 12, 4); nq_signal_add(n, signal); }
        }
        p += h.size;
    }
}
static void nq_enqueue(nq_queue *q, nq_job *j) {
    if (q->last) q->last->next = j; else q->first = j; q->last = j;
    nq_put(q->tail); q->tail = j->receipt; q->tail->refs++; pthread_cond_broadcast(&nq_cv);
}
void nvn_queue_submit(queue_t *q, int count, const uint64_t *handles) {
    if (!nvn_queue_scheduler_enabled()) {
        nvn_cap_queue(q->serial, 0); nq_stat(NVN_QUEUE_STAT_COMMAND_NODES);
        dm_host_mutex_lock(&g_sub_mu); dm_host_work++; mtl_prepare(); dm_host_work--;
        gfx_enter(); nq_cost cost = nq_cost_begin();
        mtl_queue_activate(q->serial); mtl_set_queue_origin_lower_left(q->origin_ll);
        nvn_cap_submit(count, handles); mtl_sub_begin(); nvn_ft_poll(); mtl_sub_index(handles, count);
        for (int i = 0; i < count; i++) {
            mtl_exec_rest(handles + i + 1, count - i - 1);
            mtl_exec((const uint8_t *)(uintptr_t)handles[i]);
        }
        mtl_exec_rest(NULL, 0); mtl_sub_commit(); nq_stat(NVN_QUEUE_STAT_PHYSICAL_SUBMITS);
        nq_cost_end(cost); gfx_leave(); pthread_mutex_unlock(&g_sub_mu); return;
    }
    pthread_once(&nq_once, nq_start); nq_job *j = calloc(1, sizeof *j); if (!j) die("queue job allocation failed");
    j->receipt = nq_new(); j->objects.klen = sizeof(nq_object_key); j->frame = dm_nvn_frame_count();
    // Snapshot guest command/object memory before Submit returns and the game may reuse it.
    dm_host_mutex_lock(&g_sub_mu); dm_host_work++; mtl_prepare(); dm_host_work--; nq_cost cost = nq_cost_begin();
    for (int i = 0; i < count; i++) nq_flatten(j, (const uint8_t *)(uintptr_t)handles[i], 0); nq_end(j);
    nq_cost_end(cost); pthread_mutex_unlock(&g_sub_mu);
    int waits = 0; for (nq_node *node = j->first; node; node = node->next) waits |= node->kind == NQ_EVENT || node->kind == NQ_SYNC;
    nvn_cap_queue(q->serial, waits);
    pthread_mutex_lock(&nq_mu);
    for (nq_node *node = j->first; node; node = node->next) if (node->sync_source) {
        if (node->kind == NQ_FENCE) nq_set_sync((sync_t *)node->sync_source, node->receipt); else nq_snapshot_sync(node, node->sync_source);
        node->sync_source = NULL;
    }
    nq_enqueue(nq_get(q), j); pthread_mutex_unlock(&nq_mu);
}
uint64_t nvn_queue_present(queue_t *q, const tex_t *t, double period, const int crop[4]) {
    nvn_cap_queue(q->serial, 0);
    if (!nvn_queue_scheduler_enabled()) {
        nq_receipt *r = nq_new(); nq_stat(NVN_QUEUE_STAT_PRESENT_NODES);
        nvn_cap_present(t, crop); r->present = mtl_present(t, period, crop); r->published = 1;
        return (uintptr_t)r;
    }
    pthread_once(&nq_once, nq_start);
    nq_job *j = calloc(1, sizeof *j); if (!j) die("queue present allocation failed");
    j->receipt = nq_new(); j->objects.klen = sizeof(nq_object_key); j->frame = dm_nvn_frame_count(); nq_node *n = nq_append(j, NQ_PRESENT);
    dm_host_mutex_lock(&g_sub_mu); n->texture = nq_owned_get(j, CAP_K_TEX, t); pthread_mutex_unlock(&g_sub_mu);
    n->period = period; if (crop) memcpy(n->crop, crop, sizeof n->crop);
    pthread_mutex_lock(&nq_mu); j->receipt->refs++; uint64_t ticket = (uintptr_t)j->receipt; nq_enqueue(nq_get(q), j); pthread_mutex_unlock(&nq_mu); return ticket;
}
void nvn_queue_present_release(uint64_t ticket) { if (!ticket) return; pthread_mutex_lock(&nq_mu); nq_put((nq_receipt *)(uintptr_t)ticket); pthread_mutex_unlock(&nq_mu); }
void nvn_queue_present_wait(uint64_t ticket) {
    if (!ticket) return; nq_receipt *r = (nq_receipt *)(uintptr_t)ticket;
    pthread_mutex_lock(&nq_mu); while (!r->published) pthread_cond_wait(&nq_cv, &nq_mu); uint64_t physical = r->present; pthread_mutex_unlock(&nq_mu);
    if (physical) mtl_present_wait(physical);
}
void nvn_queue_fence(queue_t *q, sync_t *s) {
    if (!nvn_queue_scheduler_enabled()) {
        (void)q;
        if (s->fenced == 3) { pthread_mutex_lock(&nq_mu); nq_put((nq_receipt *)(uintptr_t)s->fence); pthread_mutex_unlock(&nq_mu); }
        s->fenced = 1; s->fence = mtl_last_fence(); return;
    }
    pthread_mutex_lock(&nq_mu); nq_queue *p = nq_get(q);
    if (p->tail) nq_set_sync(s, p->tail); else { if (s->fenced == 3) nq_put((nq_receipt *)(uintptr_t)s->fence); s->fenced = 1; s->fence = p->physical; }
    pthread_mutex_unlock(&nq_mu);
}
int nvn_sync_wait(const sync_t *s, uint64_t timeout) {
    uint64_t start = now_ns(), fence = 0; pthread_mutex_lock(&nq_mu);
    while (!nq_sync_encoded(s, &fence)) {
        uint64_t elapsed = now_ns() - start; if (timeout != UINT64_MAX && elapsed >= timeout) { pthread_mutex_unlock(&nq_mu); return 0; }
        if (timeout == UINT64_MAX) pthread_cond_wait(&nq_cv, &nq_mu);
        else { uint64_t left = timeout - elapsed; struct timespec ts = {(time_t)(left / 1000000000ull), (long)(left % 1000000000ull)}; pthread_cond_timedwait_relative_np(&nq_cv, &nq_mu, &ts); }
    }
    pthread_mutex_unlock(&nq_mu); uint64_t elapsed = now_ns() - start;
    return mtl_wait(fence, timeout == UINT64_MAX ? timeout : elapsed >= timeout ? 0 : timeout - elapsed);
}
void nvn_queue_wait_sync(queue_t *q, const sync_t *s) {
    nvn_cap_queue(q->serial, 1);
    if (!nvn_queue_scheduler_enabled()) { (void)s; return; }
    pthread_once(&nq_once, nq_start);
    nq_job *j = calloc(1, sizeof *j); if (!j) die("queue wait allocation failed"); j->receipt = nq_new();
    pthread_mutex_lock(&nq_mu); nq_snapshot_sync(nq_append(j, NQ_SYNC), s); nq_enqueue(nq_get(q), j); pthread_mutex_unlock(&nq_mu);
}
void nvn_queue_finish(queue_t *q) {
    if (!nvn_queue_scheduler_enabled()) { (void)q; mtl_finish(); return; }
    sync_t s = {0}; nvn_queue_fence(q, &s); nvn_sync_wait(&s, UINT64_MAX); nvn_sync_release(&s);
    pthread_mutex_lock(&nq_mu); uint64_t present = nq_get(q)->present; pthread_mutex_unlock(&nq_mu); if (present) mtl_present_wait(present);
}
void nvn_queue_forget(queue_t *q) {
    if (!nvn_queue_scheduler_enabled()) { (void)q; mtl_finish(); return; }
    nvn_queue_finish(q); pthread_mutex_lock(&nq_mu); nq_queue **p = &nq_queues; while (*p && (*p)->serial != q->serial) p = &(*p)->next;
    nq_queue *old = *p; if (old) *p = old->next; pthread_mutex_unlock(&nq_mu);
    if (old) { pthread_mutex_lock(&g_sub_mu); nq_persist_drop(old); pthread_mutex_unlock(&g_sub_mu); pthread_mutex_lock(&nq_mu); nq_put(old->tail); pthread_mutex_unlock(&nq_mu); free(old); }
}
