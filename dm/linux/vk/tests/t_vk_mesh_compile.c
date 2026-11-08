// Physical compile-only gate for one MeshEXT SPIR-V module through the active
// Vulkan ICD.  This runs the driver's real compiler without recording a draw.
#include "spv_reflect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t *read_spv(const char *path, size_t *words) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || (n & 3)) { fclose(f); return NULL; }
    uint32_t *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); p = NULL; }
    fclose(f); *words = (size_t)n / 4; return p;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: t_vk_mesh_compile <mesh.spv>\n"); return 2; }
    size_t words = 0; uint32_t *code = read_spv(argv[1], &words); char why[160]; spvr_info r;
    if (!code || spvr_reflect(code, words, &r, why, sizeof why) || r.stage != VK_SHADER_STAGE_MESH_BIT_EXT) {
        fprintf(stderr, "FAIL: %s\n", code ? why : "unreadable SPIR-V"); free(code); return 1;
    }
    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "t_vk_mesh_compile", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai}; VkInstance inst;
    if (vkCreateInstance(&ii, NULL, &inst)) { fprintf(stderr, "FAIL: no Vulkan instance\n"); free(code); return 1; }
    uint32_t np = 8; VkPhysicalDevice pd[8], gpu = VK_NULL_HANDLE; vkEnumeratePhysicalDevices(inst, &np, pd);
    for (uint32_t i = 0; i < np; i++) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[i], &p); if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { gpu = pd[i]; break; } }
    if (!gpu) { fprintf(stderr, "FAIL: no GPU\n"); vkDestroyInstance(inst, NULL); free(code); return 1; }
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceShaderObjectFeaturesEXT shader = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, .pNext = &mesh};
    VkPhysicalDeviceFeatures2 feat = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &shader}; vkGetPhysicalDeviceFeatures2(gpu, &feat);
    if (!shader.shaderObject || !mesh.meshShader) { fprintf(stderr, "FAIL: shaderObject=%u meshShader=%u\n", shader.shaderObject, mesh.meshShader); vkDestroyInstance(inst, NULL); free(code); return 1; }
    uint32_t nq = 16; VkQueueFamilyProperties qp[16]; vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, qp); uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }
    float priority = 1.0f; VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &priority};
    const char *ext[] = {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feat, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = 2, .ppEnabledExtensionNames = ext}; VkDevice dev;
    VkResult vr = vkCreateDevice(gpu, &di, NULL, &dev); if (vr) { fprintf(stderr, "FAIL: vkCreateDevice %d\n", vr); vkDestroyInstance(inst, NULL); free(code); return 1; }
    uint32_t nset = 0; for (int i = 0; i < r.nbind; i++) if (r.bind[i].set + 1 > nset) nset = r.bind[i].set + 1;
    VkDescriptorSetLayout set[8] = {0}; int bad = nset > 8;
    for (uint32_t s = 0; s < nset && !bad; s++) {
        VkDescriptorSetLayoutBinding b[SPVR_MAX_BIND]; uint32_t nb = 0;
        for (int i = 0; i < r.nbind; i++) if (r.bind[i].set == s) b[nb++] = (VkDescriptorSetLayoutBinding){r.bind[i].binding, r.bind[i].type, r.bind[i].count ? r.bind[i].count : 1, VK_SHADER_STAGE_MESH_BIT_EXT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = b}; bad = vkCreateDescriptorSetLayout(dev, &li, NULL, &set[s]) != VK_SUCCESS;
    }
    PFN_vkCreateShadersEXT create = (PFN_vkCreateShadersEXT)vkGetDeviceProcAddr(dev, "vkCreateShadersEXT"); PFN_vkDestroyShaderEXT destroy = (PFN_vkDestroyShaderEXT)vkGetDeviceProcAddr(dev, "vkDestroyShaderEXT");
    VkShaderCreateInfoEXT ci = {.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = VK_SHADER_STAGE_MESH_BIT_EXT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = words * 4, .pCode = code, .pName = "main", .setLayoutCount = nset, .pSetLayouts = set}; VkShaderEXT object = VK_NULL_HANDLE;
    if (!bad) vr = create(dev, 1, &ci, NULL, &object); else vr = VK_ERROR_INITIALIZATION_FAILED;
    if (object) destroy(dev, object, NULL); for (uint32_t s = 0; s < nset; s++) if (set[s]) vkDestroyDescriptorSetLayout(dev, set[s], NULL);
    vkDestroyDevice(dev, NULL); vkDestroyInstance(inst, NULL); free(code);
    if (vr) { fprintf(stderr, "FAIL: vkCreateShadersEXT %d\n", vr); return 1; }
    printf("t_vk_mesh_compile: passed\n"); return 0;
}
