# Native Linux renderer bring-up. This intentionally stops at one relocatable
# renderer object: the runtime worktree owns the Linux loader, services, audio,
# movies, and thread platform. GAME selects the tracked profile exactly as the
# Apple and Android builds do.
ifeq ($(strip $(GAME)),)
$(error GAME is not set: make GAME=<base game title ID>)
endif

CC = clang
AR = llvm-ar
PKG_CONFIG ?= pkg-config
HOST := ../host
OUT := gen/$(GAME)
GAME_GEN := $(HOST)/gen/$(GAME)
GAME_DIR := $(if $(wildcard ../games/$(GAME)/game.tsv),../games/$(GAME),$(GAME_GEN)/profile)

CPPFLAGS := -D_GNU_SOURCE -include compat/dm_compat.h -Icompat -I$(HOST) \
            -I../gfx/nvn -I../gfx/nvn/include -I../third_party/vulkan-headers/include \
            -I$(GAME_GEN) -I$(GAME_DIR)
CFLAGS := -std=gnu11 -MMD -MP -O1 -g -fPIC -fvisibility=hidden -Wall -Wextra \
          -Wno-unused-function -Wno-format -Wno-deprecated-declarations

NVN_SRC := ../gfx/hle_nv.c ../gfx/hle_vi.c \
           $(sort $(wildcard ../gfx/nvn/*.c ../gfx/nvn/*.S ../gfx/tex/*.c))
NULL_SRC := null/nvn_null.c
VK_REPLAY_SRC := $(sort $(wildcard ../gfx/vk_android/*.c))
VK_REPLAY_OBJ := $(patsubst ../%,$(OUT)/obj/%.o,$(VK_REPLAY_SRC))
NXASAHI_VK_SRC := vk/headless.c vk/spv_reflect.c vk/vk_dev.c vk/vk_draw.c vk/vk_shader_resolve.c \
                  vk/vk_fmt.c vk/vk_mem.c vk/vk_replay.c vk/vk_window.c vk/mtl_adapters.c
NXASAHI_VK_OBJ := $(patsubst vk/%.c,$(OUT)/obj/linux/vk/%.o,$(NXASAHI_VK_SRC))
# The SDL3 window front end for the Runtime host (replaces cli/front_cli.c when linking a windowed host: DM_WINDOW=1 opens the window, else headless as before).
# Not part of the renderer relocatable: the host's link picks front_cli.c or these two.
NXASAHI_FRONT_SRC := vk/front_sdl.c vk/front_input.c
NXASAHI_FRONT_OBJ := $(patsubst vk/%.c,$(OUT)/obj/linux/vk/%.o,$(NXASAHI_FRONT_SRC))
SRC := $(NVN_SRC) $(NULL_SRC)
NVN_OBJ := $(patsubst ../%,$(OUT)/obj/%.o,$(NVN_SRC))
OBJ := $(NVN_OBJ) \
       $(patsubst %,$(OUT)/obj/linux/%.o,$(NULL_SRC))

.DEFAULT_GOAL := null-renderer
.PHONY: profile null-renderer vulkan-replay-objects nxasahi-renderer-objects nxasahi-renderer nxasahi-front vulkan-selftest vk-mem-selftest t_push t_skipcs t_gpstamp t_focus t_gmem_trace t_missing_log t_counter_report t_vk_stage t_vk_compile_watch t_vk_format t_vk_cs_gpu_lifecycle t_vk_image_refs t_vk_gmem_map t_vk_bindless_lru t_vk_mesh t_vk_mesh_compile t_vk_descriptor_pool t_vk_mesh_coverage t_vk_shader_resolve t_vk_movie dmhost dmhost-vulkan t_fs_linux clean

# The consolidated asahi branch also carries the Linux host. Keep its object
# namespace and rules separate so renderer-only checks remain bounded and a
# change of backend flags cannot accidentally reuse a renderer object.
dmhost:
	$(MAKE) -f Runtime.mk GAME=$(GAME) BACKEND=null dmhost

dmhost-vulkan:
	$(MAKE) -f Runtime.mk GAME=$(GAME) BACKEND=vulkan dmhost

t_fs_linux:
	$(MAKE) -f Runtime.mk GAME=$(GAME) t_fs_linux

profile:
	$(MAKE) -C $(HOST) GAME=$(GAME) game_profile

$(OUT)/obj/%.o: ../% | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OUT)/obj/linux/%.o: % | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OUT)/obj/gfx/vk_android/%.c.o: ../gfx/vk_android/%.c | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -DDM_LINUX $(CFLAGS) -c $< -o $@

$(OUT)/obj/linux/vk/%.o: vk/%.c | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -DDM_LINUX $(CFLAGS) $$($(PKG_CONFIG) --cflags sdl3 vulkan) -c $< -o $@

# -r resolves the whole current NVN layer against the selected backend while
# deliberately retaining the host/runtime symbols for the integration worktree.
$(OUT)/renderer-null.o: $(OBJ)
	$(CC) -nostdlib -Wl,-r -o $@ $^

$(OUT)/libdm-renderer-null.a: $(OUT)/renderer-null.o
	$(AR) rcs $@ $<

null-renderer: $(OUT)/libdm-renderer-null.a

vulkan-replay-objects: $(VK_REPLAY_OBJ)

nxasahi-renderer-objects: $(NXASAHI_VK_OBJ)

$(OUT)/renderer-nxasahi.o: $(NVN_OBJ) $(NXASAHI_VK_OBJ)
	$(CC) -nostdlib -Wl,-r -o $@ $^
	@bad="$$(llvm-nm -u $@ | awk '$$2 ~ /^mtl_/ { print $$2 }')"; \
	  if [ -n "$$bad" ]; then printf '%s\n' "unresolved Vulkan callbacks:" "$$bad"; exit 1; fi

nxasahi-renderer: $(OUT)/renderer-nxasahi.o

nxasahi-front: $(NXASAHI_FRONT_OBJ)

$(OUT)/vulkan-selftest: vk/vulkan_selftest.c
	@mkdir -p $(dir $@)
	$(CC) -std=gnu11 -O1 -g -Wall -Wextra -Werror $$($(PKG_CONFIG) --cflags sdl3 vulkan) $< -o $@ $$($(PKG_CONFIG) --libs sdl3 vulkan)

vulkan-selftest: $(OUT)/vulkan-selftest

# Device-free renderer self-tests (vk/tests): the push-constant block logic shared with vk_replay.c (vk/vk_push.h)
$(OUT)/t_push: vk/tests/t_push.c vk/vk_push.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_push: $(OUT)/t_push
	$<
$(OUT)/t_skipcs: vk/tests/t_skipcs.c vk/vk_skipcs.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_skipcs: $(OUT)/t_skipcs
	$<
$(OUT)/t_vk_movie: vk/tests/t_vk_movie.c vk/vk_movie.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_movie: $(OUT)/t_vk_movie
	$<
$(OUT)/t_gpstamp: vk/tests/t_gpstamp.c vk/vk_gpstamp.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_gpstamp: $(OUT)/t_gpstamp
	$<
$(OUT)/t_focus: vk/tests/t_focus.c vk/front_focus.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_focus: $(OUT)/t_focus
	$<
$(OUT)/t_gmem_trace: vk/tests/t_gmem_trace.c vk/vk_gmem_trace.h vk/spv_reflect.c vk/spv_reflect.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_gmem_trace.c vk/spv_reflect.c
t_gmem_trace: $(OUT)/t_gmem_trace
	$<
$(OUT)/t_bindless_spv: vk/tests/t_bindless_spv.c vk/spv_reflect.c vk/spv_reflect.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_bindless_spv.c vk/spv_reflect.c
t_bindless_spv: $(OUT)/t_bindless_spv
	$<
$(OUT)/t_missing_log: vk/tests/t_missing_log.c vk/vk_missing_log.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_missing_log: $(OUT)/t_missing_log
	$<
$(OUT)/t_counter_report: vk/tests/t_counter_report.c vk/vk_counter_report.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_counter_report: $(OUT)/t_counter_report
	$<
$(OUT)/t_vk_stage: vk/tests/t_vk_stage.c vk/vk_stage.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ $<
t_vk_stage: $(OUT)/t_vk_stage
	$<
$(OUT)/t_vk_compile_watch: vk/tests/t_vk_compile_watch.c vk/vk_compile_watch.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_compile_watch: $(OUT)/t_vk_compile_watch
	$<

$(OUT)/t_vk_image_refs: vk/tests/t_vk_image_refs.c vk/vk_image_refs.h vk/vk_tex_lifetime.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_image_refs: $(OUT)/t_vk_image_refs
	$<

$(OUT)/t_vk_gmem_map: vk/tests/t_vk_gmem_map.c vk/vk_gmem_map.h ../gfx/nvn/dmsh_abi.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_gmem_map: $(OUT)/t_vk_gmem_map
	$<

$(OUT)/t_vk_bindless_lru: vk/tests/t_vk_bindless_lru.c vk/vk_bindless_lru.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_bindless_lru: $(OUT)/t_vk_bindless_lru
	$<

$(OUT)/t_vk_format: vk/tests/t_vk_format.c vk/vk_replay.h vk/vk_fmt.c vk/vk_fmt.h ../gfx/tex/nvn_fmt_mtl.c ../gfx/tex/nvn_fmt_mtl.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_vk_format.c vk/vk_fmt.c ../gfx/tex/nvn_fmt_mtl.c
t_vk_format: $(OUT)/t_vk_format
	$<
$(OUT)/t_vk_cs_gpu_lifecycle: vk/tests/t_vk_cs_gpu_lifecycle.c vk/vk_cs_gpu_lifecycle.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_cs_gpu_lifecycle: $(OUT)/t_vk_cs_gpu_lifecycle
	$<
$(OUT)/t_vk_mesh: vk/tests/t_vk_mesh.c vk/vk_mesh.h vk/vk_mesh_trace.h vk/vk_stage.h vk/spv_reflect.c vk/spv_reflect.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_vk_mesh.c vk/spv_reflect.c
t_vk_mesh: $(OUT)/t_vk_mesh
	$<

$(OUT)/t_vk_mesh_compile: vk/tests/t_vk_mesh_compile.c vk/spv_reflect.c vk/spv_reflect.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_vk_mesh_compile.c vk/spv_reflect.c $$($(PKG_CONFIG) --libs vulkan)
t_vk_mesh_compile: $(OUT)/t_vk_mesh_compile
	@test -n "$(SPV)" || { echo 'SPV=<mesh.spv> is required'; exit 2; }
	$< $(SPV)

$(OUT)/t_vk_descriptor_pool: vk/tests/t_vk_descriptor_pool.c vk/vk_draw.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ $<
t_vk_descriptor_pool: $(OUT)/t_vk_descriptor_pool
	$<
$(OUT)/t_vk_omap: vk/tests/t_vk_omap.c vk/vk_omap.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk -o $@ $<
t_vk_omap: $(OUT)/t_vk_omap
	$<
$(OUT)/t_vk_mesh_coverage: vk/tests/t_vk_mesh_coverage.c vk/spv_reflect.c vk/spv_reflect.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_vk_mesh_coverage.c vk/spv_reflect.c
t_vk_mesh_coverage: $(OUT)/t_vk_mesh_coverage
	$<
$(OUT)/t_vk_shader_resolve: vk/tests/t_vk_shader_resolve.c vk/vk_shader_resolve.c vk/vk_shader_resolve.h vk/vk_mesh.h vk/vk_stage.h vk/spv_reflect.c vk/spv_reflect.h ../gfx/nvn/nvn_shader_stage.h | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=gnu11 -O1 -g -Wall -Wextra -Werror -Ivk $$($(PKG_CONFIG) --cflags vulkan) -o $@ vk/tests/t_vk_shader_resolve.c vk/vk_shader_resolve.c vk/spv_reflect.c
t_vk_shader_resolve: $(OUT)/t_vk_shader_resolve
	$<

$(OUT)/vk-mem-selftest: vk/vk_mem_selftest.c vk/vk_dev.c vk/vk_mem.c | profile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -DDM_LINUX $(CFLAGS) -Werror $$($(PKG_CONFIG) --cflags sdl3 vulkan) $^ -o $@ $$($(PKG_CONFIG) --libs sdl3 vulkan)

vk-mem-selftest: $(OUT)/vk-mem-selftest

clean:
	rm -rf gen

# Keep the incremental Vulkan build ABI-consistent too.  OBJ only covers the
# null renderer, so omitting these dependency files can leave (for example)
# vk_dev.o and vk_draw.o built against different layouts of vk_dev.h.
-include $(sort $(OBJ:.o=.d) $(VK_REPLAY_OBJ:.o=.d) \
	$(NXASAHI_VK_OBJ:.o=.d) $(NXASAHI_FRONT_OBJ:.o=.d))
