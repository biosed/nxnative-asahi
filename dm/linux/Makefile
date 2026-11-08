HOST_CC ?= clang
GAME ?= 0400486024DEC000
BUILD := build
CFLAGS := -std=c11 -O2 -g -Wall -Wextra -Werror

.DEFAULT_GOAL := probes
.PHONY: probes probe-thread-slots probe-entry-object probe-runtime-objects test-host-tp-encoding profile clean

probes: probe-thread-slots probe-entry-object test-host-tp-encoding

RUNTIME_OBJECTS := \
	gen/$(GAME)/obj/android/compat/compat.c.o \
	gen/$(GAME)/obj/host/thread_slots.c.o \
	gen/$(GAME)/obj/host/entry.S.o \
	gen/$(GAME)/obj/host/loader.c.o \
	gen/$(GAME)/obj/host/hle.c.o \
	gen/$(GAME)/obj/host/thread.c.o \
	gen/$(GAME)/obj/host/hle_thread.c.o \
	gen/$(GAME)/obj/gfx/nvn/hle_nvn.c.o \
	gen/$(GAME)/obj/android/null/nvn_null.c.o

probe-runtime-objects:
	$(MAKE) -C ../android GAME=$(GAME) BACKEND=null CC=$(HOST_CC) AR=llvm-ar $(RUNTIME_OBJECTS)

$(BUILD)/thread-slots-probe: tests/t_thread_slots.c ../host/thread_slots.c ../host/thread_slots.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(CFLAGS) -pthread -I../host tests/t_thread_slots.c ../host/thread_slots.c -o $@

probe-thread-slots: $(BUILD)/thread-slots-probe
	$<

$(BUILD)/entry.o: ../host/entry.S ../host/asm_sym.h
	@mkdir -p $(dir $@)
	$(HOST_CC) -g -I../host -c $< -o $@

probe-entry-object: $(BUILD)/entry.o
	llvm-nm $< | grep -Eq ' [BT] dm_(gate|thread_enter)$$'
	llvm-readelf -r $< | awk '/R_AARCH64_ADR_PREL_PG_HI21.*dm_gate_top/ { found = 1 } END { exit !found }'

test-host-tp-encoding:
	DM_PATCH_TP=apple python3 ../tools/patch_nso.py --selftest-host-tp
	DM_PATCH_TP=linux python3 ../tools/patch_nso.py --selftest-host-tp

profile:
	$(MAKE) -C ../host GAME=$(GAME) game_profile

clean:
	rm -rf $(BUILD)
