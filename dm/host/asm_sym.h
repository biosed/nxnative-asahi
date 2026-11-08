#pragma once
#ifdef __APPLE__
#define SYM(name) _##name
#define SYM_HIDDEN .private_extern
#else
#define SYM(name) name
#define SYM_HIDDEN .hidden
#endif

#ifdef __APPLE__
#define PAGE(name) _##name@PAGE
#define PAGEOFF(name) _##name@PAGEOFF
#else
#define PAGE(name) name
#define PAGEOFF(name) :lo12:name
#endif

#ifdef __APPLE__
#define CSTRING_SECTION .cstring
#else
#define CSTRING_SECTION .section .rodata.str1.1,"aMS",%progbits,1
#endif
