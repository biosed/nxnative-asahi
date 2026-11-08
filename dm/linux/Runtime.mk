# Native Linux AArch64 host for the consolidated asahi branch. The ordinary
# dm/linux/Makefile owns renderer-only builds and delegates its final host
# targets here, keeping the two generated-object namespaces independent.
CC := clang
AR := ar
BACKEND ?= null

ifeq ($(strip $(GAME)),)
ifneq ($(filter-out clean,$(MAKECMDGOALS)),)
$(error GAME is not set: make GAME=<base game title ID>)
endif
endif

HOST := ../host
OUT := gen/$(GAME)
RUNTIME_OBJ := $(OUT)/obj/runtime
GAME_DIR := ../games/$(GAME)
GAME_H := $(OUT)/game_profile.h

CFLAGS := -MMD -MP -O1 -g -fPIE -fvisibility=hidden -D_GNU_SOURCE \
	-Wall -Wno-unused-function -Wno-format -Wno-incompatible-pointer-types-discards-qualifiers \
	-include compat/dm_compat.h -Icompat -Isrc -I$(HOST) -I../gfx/nvn \
	-I../gfx/nvn/include -I../third_party/vulkan-headers/include -I$(OUT) \
	-I../third_party/libvpx -I../third_party/stb_vorbis \
	$(shell pkg-config --cflags sdl3 dav1d)

# The portable host has explicit Linux replacements for these Darwin sources.
HOST_ALL := $(HOST)/main.c $(HOST)/loader.c $(HOST)/hle.c $(HOST)/crash.c \
	$(HOST)/probe.c $(HOST)/thread.c $(HOST)/save_sync.c \
	$(sort $(filter-out $(HOST)/hle_cheats.c,$(wildcard $(HOST)/hle_*.c))) \
	$(HOST)/entry.S
PORT_HOST := main.c loader.c hle.c crash.c probe.c \
	hle_svc.c hle_fdstat.c hle_lifecycle.c save_sync.c thread.c entry.S
HOST_SRC := $(filter-out $(addprefix $(HOST)/,$(PORT_HOST)) \
	$(HOST)/hle_movie.c $(HOST)/hle_movie_avc.c $(HOST)/hle_movie_av1.c,$(HOST_ALL))
COMMON_LOCAL_SRC := compat/compat.c src/cheats_none.c src/audiounit_sdl.c \
	src/movie_none.c src/hle_movie_av1.c src/runtime_glue.c \
	$(addprefix src/,$(PORT_HOST))

ifeq ($(BACKEND),vulkan)
GFX_SRC :=
LOCAL_SRC := $(COMMON_LOCAL_SRC) src/nvn_capture_none.c sdl/front_main.c
RENDERER_OBJ := $(OUT)/renderer-nxasahi.o
RENDERER_FRONT_OBJ := \
	$(OUT)/obj/linux/vk/front_sdl.o \
	$(OUT)/obj/linux/vk/front_input.o
EXTERNAL_OBJ := $(RENDERER_OBJ) $(RENDERER_FRONT_OBJ)
EXTRA_LIBS := $(shell pkg-config --libs vulkan)
BIN := $(OUT)/dmhost-vulkan
else ifeq ($(BACKEND),null)
GFX_SRC := ../gfx/hle_nv.c ../gfx/hle_vi.c \
	$(sort $(wildcard ../gfx/nvn/*.c ../gfx/nvn/*.S ../gfx/tex/*.c))
LOCAL_SRC := $(COMMON_LOCAL_SRC) null/nvn_null.c cli/front_cli.c
EXTERNAL_OBJ :=
EXTRA_LIBS :=
BIN := $(OUT)/dmhost
else
$(error unsupported BACKEND '$(BACKEND)': use null or vulkan)
endif

SRC := $(HOST_SRC) $(GFX_SRC) $(LOCAL_SRC)
OBJ := $(patsubst ../%,$(RUNTIME_OBJ)/%.o,$(filter ../%,$(SRC))) \
	$(patsubst %,$(RUNTIME_OBJ)/linux/%.o,$(filter-out ../%,$(SRC)))

.DEFAULT_GOAL := dmhost
.PHONY: dmhost objects profile renderer-artifacts t_fs_linux clean
dmhost: $(BIN)
objects: $(OBJ)

ifeq ($(BACKEND),vulkan)
renderer-artifacts:
	$(MAKE) GAME=$(GAME) nxasahi-renderer nxasahi-front
else
renderer-artifacts:
	@:
endif

profile: $(GAME_H)
$(GAME_H): $(GAME_DIR)/game.tsv ../games/game.py
	@mkdir -p $(dir $@)
	python3 ../games/game.py header $(GAME) $@

$(OUT)/t_fs_linux: tests/t_fs_linux.c compat/sys/xattr.h $(HOST)/fs_error_rate.h | profile
	@mkdir -p $(dir $@)
	$(CC) -D_GNU_SOURCE -include compat/dm_compat.h -Icompat -I$(HOST) \
		-std=gnu11 -O1 -g -Wall -Wextra -Werror -pthread -o $@ $<
t_fs_linux: $(OUT)/t_fs_linux
	$<

$(RUNTIME_OBJ)/linux/src/main.c.o: CFLAGS += -Dmain=dm_main
$(RUNTIME_OBJ)/%.o: ../% $(GAME_H)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(RUNTIME_OBJ)/linux/%.o: % $(GAME_H)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

VPX := ../third_party/libvpx
VPX_SRC := $(sort $(shell find $(VPX) -name '*.c'))
VPX_OBJ := $(patsubst $(VPX)/%.c,$(RUNTIME_OBJ)/vpx/%.o,$(VPX_SRC))
$(RUNTIME_OBJ)/vpx/%.o: $(VPX)/%.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -g -fPIC -I$(VPX) -I$(VPX)/config -c $< -o $@
$(OUT)/libvpx-runtime.a: $(VPX_OBJ)
	rm -f $@
	$(AR) rcs $@ $^

$(BIN): $(OBJ) $(OUT)/libvpx-runtime.a renderer-artifacts
	$(CC) -fuse-ld=lld -pie -rdynamic -o $@ $(OBJ) $(EXTERNAL_OBJ) $(OUT)/libvpx-runtime.a \
		-pthread -lm -ldl $(shell pkg-config --libs sdl3 dav1d) $(EXTRA_LIBS)

-include $(shell find $(RUNTIME_OBJ) -name '*.d' 2>/dev/null)
clean:
	rm -rf gen
