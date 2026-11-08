// AArch64 signal-context access on Darwin and Linux.
#pragma once
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/ucontext.h>

#ifdef __APPLE__
#define UC_X(uc, n) ((uc)->uc_mcontext->__ss.__x[n])
#define UC_FP(uc) ((uc)->uc_mcontext->__ss.__fp)
#define UC_LR(uc) ((uc)->uc_mcontext->__ss.__lr)
#define UC_SP(uc) ((uc)->uc_mcontext->__ss.__sp)
#define UC_PC(uc) ((uc)->uc_mcontext->__ss.__pc)
#define UC_PSTATE(uc) ((uc)->uc_mcontext->__ss.__cpsr)
static inline uint64_t uc_esr(ucontext_t *uc) { return uc->uc_mcontext->__es.__esr; }
static inline int uc_fpsimd(ucontext_t *uc, void *v, uint32_t *fpcr, uint32_t *fpsr) {
    memcpy(v, uc->uc_mcontext->__ns.__v, 32 * 16);
    *fpcr = uc->uc_mcontext->__ns.__fpcr; *fpsr = uc->uc_mcontext->__ns.__fpsr;
    return 1;
}
#else
#define UC_X(uc, n) ((uc)->uc_mcontext.regs[n])
#define UC_FP(uc) ((uc)->uc_mcontext.regs[29])
#define UC_LR(uc) ((uc)->uc_mcontext.regs[30])
#define UC_SP(uc) ((uc)->uc_mcontext.sp)
#define UC_PC(uc) ((uc)->uc_mcontext.pc)
#define UC_PSTATE(uc) ((uc)->uc_mcontext.pstate)
static inline const uint8_t *uc_record(ucontext_t *uc, uint32_t magic) {
    const uint8_t *p = (const uint8_t *)uc->uc_mcontext.__reserved;
    const uint8_t *end = p + sizeof uc->uc_mcontext.__reserved;
    while (p + 8 <= end) {
        uint32_t found, size;
        memcpy(&found, p, 4); memcpy(&size, p + 4, 4);
        if (!found || size < 8 || p + size > end) return NULL;
        if (found == magic) return p;
        p += size;
    }
    return NULL;
}
static inline uint64_t uc_esr(ucontext_t *uc) {
    const uint8_t *record = uc_record(uc, 0x45535201);
    uint64_t value = 0;
    if (record) memcpy(&value, record + 8, 8);
    return value;
}
static inline int uc_fpsimd(ucontext_t *uc, void *v, uint32_t *fpcr, uint32_t *fpsr) {
    const uint8_t *record = uc_record(uc, 0x46508001);
    if (!record) return 0;
    memcpy(fpsr, record + 8, 4); memcpy(fpcr, record + 12, 4);
    memcpy(v, record + 16, 32 * 16);
    return 1;
}
#endif
