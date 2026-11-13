// Linux build: the one Vulkan instance and device the host uses (vk_dev.c): the window's presentation (vk_window.c), guest memory backing (vk_mem.c) and the backend
#pragma once
#include <pthread.h>
#include <vulkan/vulkan.h>
#include "vk_mesh.h"
struct SDL_Window;
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceMemoryProperties mem;
    int window;                   // made with the window's instance extensions and VK_KHR_swapchain
    int placed;                   // VK_EXT_map_memory_placed (guest memory backed in place, vk_mem.c)
    int zero_init;                // VK_EXT_zero_initialize_device_memory (a guest chunk's closed pages stay uncommitted, vk_mem.c)
    int maintenance8;             // VK_KHR_maintenance8 (dynamic texel offsets in translated shaders)
    int shader_object;            // VK_EXT_shader_object (the backend's shaders, vk_draw.c)
    int index_u8;                 // VK_KHR_index_type_uint8 (NVN's byte indices drawn as they are)
    vkx_mesh_caps mesh;           // driver capability; runtime_ready stays 0 until renderer physical proof
    vkx_mesh_caps mesh_title;     // explicit non-conformant title profile; consumed only after the proof gate selects it
    PFN_vkCreateShadersEXT create_shaders; PFN_vkDestroyShaderEXT destroy_shader; PFN_vkCmdBindShadersEXT bind_shaders; PFN_vkGetShaderBinaryDataEXT shader_binary;
    PFN_vkCmdDrawMeshTasksEXT draw_mesh;
    PFN_vkCmdDrawMeshTasksIndirectEXT draw_mesh_indirect;
    PFN_vkMapMemory2KHR map2;
} vkd_t;
extern vkd_t g_vkd;
// Before the game starts (front_sdl.c): with a window, after SDL's video init (its instance extensions), else headless. 0 = no usable device (logged); the host then runs without one
int vkd_init(int with_window);
static inline int vkd_ok(void) { return g_vkd.dev != VK_NULL_HANDLE; }
const vkx_mesh_caps *vkd_mesh_active(void); // NULL unless the exact root proof token and all driver gates pass
// The queue is shared by the present thread, sparse binds and the backend: every vkQueue* call holds this
extern pthread_mutex_t g_vkd_qlock;
extern volatile unsigned long g_vkd_wsubs;   // presentation submissions and presents made (DM_VK_PART_GAPS tells gaps behind them apart)
