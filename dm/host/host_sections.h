#pragma once

#ifdef __APPLE__
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#define DM_SECTION(name) "__DATA," #name
#define DM_SECTION_BOUNDS(name, type, first, count) do {                                  \
    unsigned long dm_section_size_;                                                       \
    (first) = (const type *)getsectiondata(&_mh_execute_header, "__DATA", #name,         \
                                           &dm_section_size_);                            \
    (count) = (first) ? (unsigned)(dm_section_size_ / sizeof(type)) : 0;                  \
} while (0)
#else
#define DM_SECTION(name) #name
#define DM_SECTION_BOUNDS(name, type, first, count) do {                                  \
    extern const type __start_##name[], __stop_##name[];                                  \
    (first) = __start_##name;                                                             \
    (count) = (unsigned)(__stop_##name - __start_##name);                                 \
} while (0)
#endif
