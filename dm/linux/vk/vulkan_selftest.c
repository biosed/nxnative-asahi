// Game-free Linux Vulkan bring-up for the nxnative renderer. This is the
// shader-free prefix of BananaNative's Vulkan self-test: select the physical
// device by capabilities, create an SDL surface/swapchain, copy a host-visible
// buffer, clear swapchain images, and present them. No game data is used.
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN(a) ((uint32_t)(sizeof(a) / sizeof((a)[0])))
#define VK_OK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "[RED] vk selftest: %s returned %d\n", #call, r_); goto fail; \
} } while (0)

typedef struct {
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    uint32_t qf;
    int mesh_ext;
    int map2_ext;
    int placed_ext;
    int push_ext;
    int gpl_ext;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT placed;
} candidate;

static int has_ext(const VkExtensionProperties *ext, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(ext[i].extensionName, name)) return 1;
    return 0;
}

static uint32_t memory_type(const VkPhysicalDeviceMemoryProperties *m, uint32_t bits,
                            VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < m->memoryTypeCount; i++)
        if ((bits & (1u << i)) && (m->memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static int inspect_device(VkPhysicalDevice pd, VkSurfaceKHR surface, candidate *c) {
    memset(c, 0, sizeof *c);
    c->pd = pd;
    vkGetPhysicalDeviceProperties(pd, &c->props);
    vkGetPhysicalDeviceMemoryProperties(pd, &c->mem);

    uint32_t ne = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL) != VK_SUCCESS) return 0;
    VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
    if (!ext) return 0;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext) != VK_SUCCESS) { free(ext); return 0; }
    int swap = has_ext(ext, ne, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    c->mesh_ext = has_ext(ext, ne, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    c->map2_ext = has_ext(ext, ne, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME);
    c->placed_ext = has_ext(ext, ne, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME);
    c->push_ext = has_ext(ext, ne, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    c->gpl_ext = has_ext(ext, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) &&
                 has_ext(ext, ne, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    free(ext);
    if (!swap || c->props.apiVersion < VK_API_VERSION_1_3) return 0;

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    c->qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        if (present && (qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                       (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { c->qf = i; break; }
    }
    free(qp);
    if (c->qf == UINT32_MAX) return 0;

    c->f12 = (VkPhysicalDeviceVulkan12Features){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    c->f13 = (VkPhysicalDeviceVulkan13Features){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    c->mesh = (VkPhysicalDeviceMeshShaderFeaturesEXT){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    c->placed = (VkPhysicalDeviceMapMemoryPlacedFeaturesEXT){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    void *chain = &c->f12;
    c->f12.pNext = &c->f13;
    if (c->mesh_ext) { c->mesh.pNext = chain; chain = &c->mesh; }
    if (c->placed_ext) { c->placed.pNext = chain; chain = &c->placed; }
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = chain};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    return c->f12.bufferDeviceAddress && c->f12.timelineSemaphore && c->f13.dynamicRendering;
}

static VkSurfaceFormatKHR choose_format(const VkSurfaceFormatKHR *f, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        if (f[i].format == VK_FORMAT_B8G8R8A8_UNORM || f[i].format == VK_FORMAT_R8G8B8A8_UNORM) return f[i];
    return f[0];
}

int main(int argc, char **argv) {
    int frames = 4;
    if (argc == 2) { frames = atoi(argv[1]); if (frames < 1 || frames > 600) frames = 4; }
    int rc = 1;
    SDL_Window *window = NULL;
    VkInstance inst = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkSwapchainKHR swap = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkBuffer src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    VkDeviceMemory src_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;
    void *src_map = NULL, *dst_map = NULL;
    VkImage *images = NULL;

    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "[RED] vk selftest: SDL_Init: %s\n", SDL_GetError()); return 1; }
    window = SDL_CreateWindow("nxnative Vulkan self-test", 640, 360, SDL_WINDOW_VULKAN);
    if (!window) { fprintf(stderr, "[RED] vk selftest: SDL_CreateWindow: %s\n", SDL_GetError()); goto fail; }
    uint32_t nie = 0;
    const char *const *ie = SDL_Vulkan_GetInstanceExtensions(&nie);
    if (!ie || !nie) { fprintf(stderr, "[RED] vk selftest: SDL Vulkan extensions: %s\n", SDL_GetError()); goto fail; }

    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "nxnative Linux selftest",
                            .pEngineName = "nxnative", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai,
                                .enabledExtensionCount = nie, .ppEnabledExtensionNames = ie};
    VK_OK(vkCreateInstance(&ici, NULL, &inst));
    if (!SDL_Vulkan_CreateSurface(window, inst, NULL, &surface)) {
        fprintf(stderr, "[RED] vk selftest: SDL_Vulkan_CreateSurface: %s\n", SDL_GetError()); goto fail;
    }

    uint32_t npd = 0;
    VK_OK(vkEnumeratePhysicalDevices(inst, &npd, NULL));
    VkPhysicalDevice *pds = calloc(npd ? npd : 1, sizeof *pds);
    candidate *cand = calloc(npd ? npd : 1, sizeof *cand);
    if (!pds || !cand) { free(pds); free(cand); goto fail; }
    VK_OK(vkEnumeratePhysicalDevices(inst, &npd, pds));
    int chosen = -1;
    for (uint32_t i = 0; i < npd; i++) {
        if (!inspect_device(pds[i], surface, &cand[i])) continue;
        fprintf(stderr, "[vk] candidate %u: %s, Vulkan %u.%u.%u%s\n", i, cand[i].props.deviceName,
                VK_API_VERSION_MAJOR(cand[i].props.apiVersion), VK_API_VERSION_MINOR(cand[i].props.apiVersion),
                VK_API_VERSION_PATCH(cand[i].props.apiVersion), cand[i].mesh.meshShader ? ", mesh+task capable" : "");
        if (chosen < 0 || strstr(cand[i].props.deviceName, "Apple M1 Pro")) chosen = (int)i;
    }
    free(pds);
    if (chosen < 0) { fprintf(stderr, "[RED] vk selftest: no Vulkan 1.3 graphics+compute+present device with BDA/timeline/dynamic rendering\n"); free(cand); goto fail; }
    candidate C = cand[chosen]; free(cand);
    fprintf(stderr, "[vk] selected %s; mesh=%d task=%d queries=%d; placed=%d range=%d reserve=%d; push=%d gpl=%d\n",
            C.props.deviceName, C.mesh.meshShader, C.mesh.taskShader, C.mesh.meshShaderQueries,
            C.placed.memoryMapPlaced, C.placed.memoryMapRangePlaced, C.placed.memoryUnmapReserve, C.push_ext, C.gpl_ext);

    const char *dext[5]; uint32_t nde = 0;
    dext[nde++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    if (C.mesh_ext && C.mesh.meshShader) dext[nde++] = VK_EXT_MESH_SHADER_EXTENSION_NAME;
    if (C.placed_ext && C.placed.memoryMapPlaced) dext[nde++] = VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME;
    if (C.map2_ext) dext[nde++] = VK_KHR_MAP_MEMORY_2_EXTENSION_NAME;
    float priority = 1.f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = C.qf,
                                   .queueCount = 1, .pQueuePriorities = &priority};
    VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .bufferDeviceAddress = VK_TRUE, .timelineSemaphore = VK_TRUE};
    VkPhysicalDeviceVulkan13Features e13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = VK_TRUE, .synchronization2 = C.f13.synchronization2};
    e12.pNext = &e13;
    void *echain = &e12;
    VkPhysicalDeviceMeshShaderFeaturesEXT emesh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (C.mesh_ext && C.mesh.meshShader) {
        emesh.meshShader = VK_TRUE; emesh.taskShader = C.mesh.taskShader; emesh.meshShaderQueries = C.mesh.meshShaderQueries;
        emesh.pNext = echain; echain = &emesh;
    }
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT eplaced = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    if (C.placed_ext && C.placed.memoryMapPlaced) {
        eplaced.memoryMapPlaced = VK_TRUE; eplaced.memoryMapRangePlaced = C.placed.memoryMapRangePlaced;
        eplaced.memoryUnmapReserve = C.placed.memoryUnmapReserve; eplaced.pNext = echain; echain = &eplaced;
    }
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = echain,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = nde, .ppEnabledExtensionNames = dext};
    VK_OK(vkCreateDevice(C.pd, &dci, NULL, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, C.qf, 0, &queue);

    VkSurfaceCapabilitiesKHR caps;
    VK_OK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(C.pd, surface, &caps));
    uint32_t nf = 0; VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(C.pd, surface, &nf, NULL));
    VkSurfaceFormatKHR *formats = calloc(nf ? nf : 1, sizeof *formats);
    if (!formats) goto fail;
    VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(C.pd, surface, &nf, formats));
    VkSurfaceFormatKHR sf = choose_format(formats, nf); free(formats);
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) extent = (VkExtent2D){640, 360};
    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount && image_count > caps.maxImageCount) image_count = caps.maxImageCount;
    VkSwapchainCreateInfoKHR sci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface,
        .minImageCount = image_count, .imageFormat = sf.format, .imageColorSpace = sf.colorSpace, .imageExtent = extent,
        .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
    VK_OK(vkCreateSwapchainKHR(dev, &sci, NULL, &swap));
    VK_OK(vkGetSwapchainImagesKHR(dev, swap, &image_count, NULL));
    images = calloc(image_count, sizeof *images); if (!images) goto fail;
    VK_OK(vkGetSwapchainImagesKHR(dev, swap, &image_count, images));

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VK_OK(vkCreateBuffer(dev, &bci, NULL, &src));
    VK_OK(vkCreateBuffer(dev, &bci, NULL, &dst));
    VkMemoryRequirements smr, dmr; vkGetBufferMemoryRequirements(dev, src, &smr); vkGetBufferMemoryRequirements(dev, dst, &dmr);
    uint32_t smt = memory_type(&C.mem, smr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t dmt = memory_type(&C.mem, dmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (smt == UINT32_MAX || dmt == UINT32_MAX) { fprintf(stderr, "[RED] vk selftest: no coherent host-visible buffer memory\n"); goto fail; }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = smr.size, .memoryTypeIndex = smt};
    VK_OK(vkAllocateMemory(dev, &mai, NULL, &src_mem)); mai.allocationSize = dmr.size; mai.memoryTypeIndex = dmt;
    VK_OK(vkAllocateMemory(dev, &mai, NULL, &dst_mem));
    VK_OK(vkBindBufferMemory(dev, src, src_mem, 0)); VK_OK(vkBindBufferMemory(dev, dst, dst_mem, 0));
    VK_OK(vkMapMemory(dev, src_mem, 0, 4096, 0, &src_map)); VK_OK(vkMapMemory(dev, dst_mem, 0, 4096, 0, &dst_map));
    for (uint32_t i = 0; i < 1024; i++) ((uint32_t *)src_map)[i] = 0x5a000000u ^ i;
    memset(dst_map, 0, 4096);

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = C.qf};
    VK_OK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VK_OK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkSemaphoreCreateInfo sem = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_OK(vkCreateSemaphore(dev, &sem, NULL, &acquired)); VK_OK(vkCreateSemaphore(dev, &sem, NULL, &rendered));
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_OK(vkCreateFence(dev, &fci, NULL, &fence));

    for (int frame = 0; frame < frames; frame++) {
        uint32_t ii; VK_OK(vkAcquireNextImageKHR(dev, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &ii));
        VK_OK(vkResetFences(dev, 1, &fence)); VK_OK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_OK(vkBeginCommandBuffer(cb, &bi));
        if (!frame) {
            vkCmdFillBuffer(cb, dst, 0, 4096, 0);
            VkBufferCopy copy = {.size = 4096}; vkCmdCopyBuffer(cb, src, dst, 1, &copy);
        }
        VkImageMemoryBarrier to_clear = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = images[ii], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_clear);
        VkClearColorValue color = {.float32 = {frame & 1 ? 0.08f : 0.55f, frame & 1 ? 0.35f : 0.08f, 0.7f, 1.f}};
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, images[ii], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        VkImageMemoryBarrier to_present = to_clear; to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; to_present.dstAccessMask = 0;
        to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &to_present);
        VK_OK(vkEndCommandBuffer(cb));
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired,
            .pWaitDstStageMask = &wait_stage, .commandBufferCount = 1, .pCommandBuffers = &cb,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered};
        VK_OK(vkQueueSubmit(queue, 1, &si, fence));
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &rendered,
            .swapchainCount = 1, .pSwapchains = &swap, .pImageIndices = &ii};
        VkResult pr = vkQueuePresentKHR(queue, &pi); if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) { fprintf(stderr, "[RED] vk selftest: present %d\n", pr); goto fail; }
        VK_OK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    }
    if (memcmp(src_map, dst_map, 4096)) { fprintf(stderr, "[RED] vk selftest: buffer copy mismatch\n"); goto fail; }
    fprintf(stderr, "[vk] selftest passed: capability-selected device created; 4096-byte copy verified; %d clear/present frames\n", frames);
    rc = 0;

fail:
    if (dev) vkDeviceWaitIdle(dev);
    if (src_map) vkUnmapMemory(dev, src_mem); if (dst_map) vkUnmapMemory(dev, dst_mem);
    if (src) vkDestroyBuffer(dev, src, NULL); if (dst) vkDestroyBuffer(dev, dst, NULL);
    if (src_mem) vkFreeMemory(dev, src_mem, NULL); if (dst_mem) vkFreeMemory(dev, dst_mem, NULL);
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (acquired) vkDestroySemaphore(dev, acquired, NULL); if (rendered) vkDestroySemaphore(dev, rendered, NULL);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    free(images);
    if (swap) vkDestroySwapchainKHR(dev, swap, NULL);
    if (dev) vkDestroyDevice(dev, NULL);
    if (surface) SDL_Vulkan_DestroySurface(inst, surface, NULL);
    if (inst) vkDestroyInstance(inst, NULL);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    return rc;
}
