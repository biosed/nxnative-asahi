// Renderer-only Linux platform helpers. Runtime owns the full host platform.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

// Copy through the kernel so an unreadable guest page returns an error instead
// of faulting this process. BananaNative's draw binder uses 0 for full success.
static inline int plat_mem_read(uint64_t addr, void *out, size_t len) {
    struct iovec local = {out, len};
    struct iovec remote = {(void *)(uintptr_t)addr, len};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)len ? 0 : -1;
}

// Renderer-only stand-ins until Runtime supplies its guest mapping bitmap.
// Probe one byte per 4-KiB guest page through the kernel: PROT_NONE/unmapped
// pages fail without faulting this process, even on a 16-KiB host-page kernel.
static inline uint64_t vka_linux_guest_unmapped(uintptr_t addr, size_t len) {
    uint64_t count = 0;
    if (!len) return 0;
    uintptr_t last = addr + len - 1;
    if (last < addr) return UINT64_MAX;
    for (uintptr_t page = addr & ~(uintptr_t)0xfff; page <= last; page += 0x1000) {
        unsigned char byte;
        count += plat_mem_read(page, &byte, 1) != 0;
        if (page > UINTPTR_MAX - 0x1000) break;
    }
    return count;
}

// Read each guest page independently and zero any page that became inaccessible.
// Returns the number of unreadable guest pages, matching dm_guest_read_mapped.
static inline uint64_t vka_linux_guest_read_mapped(void *dst, uintptr_t addr, size_t len) {
    uint64_t count = 0;
    unsigned char *out = dst;
    uintptr_t end = addr + len;
    if (end < addr) { memset(dst, 0, len); return UINT64_MAX; }
    for (uintptr_t pos = addr; pos < end;) {
        uintptr_t next = (pos & ~(uintptr_t)0xfff) + 0x1000;
        if (next > end) next = end;
        size_t chunk = next - pos;
        if (plat_mem_read(pos, out + (pos - addr), chunk)) {
            memset(out + (pos - addr), 0, chunk);
            count++;
        }
        pos = next;
    }
    return count;
}
