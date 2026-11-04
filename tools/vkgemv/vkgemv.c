// vkgemv: how fast can a hand-written 4-bit mat-vec run on the Mali? (derived from vkprobe; prints the device info too)
// vkprobe: can the adb shell user drive the TV's Mali GPU through Vulkan, what does it support, and how fast can
// a compute shader read RAM? Built 32-bit (armeabi-v7a) with the NDK because the box only has 32-bit GPU drivers.
// usage: vkprobe [MB]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include "gemv.spv.h"
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const uint32_t MB = argc > 1 ? atoi(argv[1]) : 128;
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vkprobe", 1, 0, 0, VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai };
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t n = 0; CK(vkEnumeratePhysicalDevices(inst, &n, 0));
    printf("physical devices: %u\n", n); if (!n) return 1;
    VkPhysicalDevice pd[4]; n = n > 4 ? 4 : n; CK(vkEnumeratePhysicalDevices(inst, &n, pd));
    VkPhysicalDevice d = pd[0];

    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(d, &pr);
    printf("device: %s  api %u.%u.%u  driver 0x%x  vendor 0x%x  type %d\n", pr.deviceName, VK_VERSION_MAJOR(pr.apiVersion),
           VK_VERSION_MINOR(pr.apiVersion), VK_VERSION_PATCH(pr.apiVersion), pr.driverVersion, pr.vendorID, pr.deviceType);
    printf("limits: maxComputeWorkGroupInvocations %u, maxComputeSharedMemorySize %u, maxStorageBufferRange %u MB, maxPushConstants %u\n",
           pr.limits.maxComputeWorkGroupInvocations, pr.limits.maxComputeSharedMemorySize, pr.limits.maxStorageBufferRange >> 20, pr.limits.maxPushConstantsSize);

    // extensions that llama.cpp's Vulkan backend cares about
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(d, 0, &ne, 0);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(d, 0, &ne, ex);
    const char *want[] = { "VK_KHR_16bit_storage", "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage", "VK_KHR_shader_integer_dot_product",
                           "VK_KHR_cooperative_matrix", "VK_EXT_subgroup_size_control", "VK_KHR_maintenance4", "VK_KHR_shader_subgroup_extended_types",
                           "VK_EXT_shader_subgroup_ballot", "VK_KHR_buffer_device_address", "VK_EXT_pipeline_robustness", "VK_KHR_shader_bfloat16" };
    printf("extensions (%u total):", ne);
    for (unsigned w = 0; w < sizeof want / sizeof *want; w++) { int has = 0; for (uint32_t i = 0; i < ne; i++) if (!strcmp(ex[i].extensionName, want[w])) has = 1;
        printf(" %s=%d", want[w] + 3, has); }
    printf("\n");

    VkPhysicalDeviceShaderFloat16Int8Features f16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
    VkPhysicalDevice16BitStorageFeatures s16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &f16 };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &s16 };
    vkGetPhysicalDeviceFeatures2(d, &f2);
    printf("features: shaderFloat16 %d shaderInt8 %d storageBuffer16BitAccess %d shaderInt16 %d shaderInt64 %d\n",
           f16.shaderFloat16, f16.shaderInt8, s16.storageBuffer16BitAccess, f2.features.shaderInt16, f2.features.shaderInt64);
    VkPhysicalDeviceSubgroupProperties sg = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg };
    vkGetPhysicalDeviceProperties2(d, &p2);
    printf("subgroup: size %u ops 0x%x stages 0x%x\n", sg.subgroupSize, sg.supportedOperations, sg.supportedStages);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(d, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) printf("heap %u: %llu MB flags 0x%x\n", i, (unsigned long long) mp.memoryHeaps[i].size >> 20, mp.memoryHeaps[i].flags);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) printf("memtype %u: heap %u flags 0x%x\n", i, mp.memoryTypes[i].heapIndex, mp.memoryTypes[i].propertyFlags);

    // compute queue + device
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, 0);
    VkQueueFamilyProperties qf[8]; nq = nq > 8 ? 8 : nq; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, qi, 1, &prio };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &qci };
    VkDevice dev; CK(vkCreateDevice(d, &dci, 0, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);


    // weights: nrows x nb blocks (uvec4 = 32 nibbles) + f32 scale per block; activations: nb blocks x 2 ivec4 + scale
    const uint32_t K = argc > 2 ? atoi(argv[2]) : 2048, NR = argc > 3 ? atoi(argv[3]) : 8192, ROWS = argc > 4 ? atoi(argv[4]) : 4;
    const uint32_t nb = K / 32;
    VkDeviceSize szs[5] = { (VkDeviceSize) NR * nb * 16, (VkDeviceSize) NR * nb * 4, (VkDeviceSize) nb * 32, nb * 4, NR * 4 };
    VkBuffer b[5]; VkDeviceMemory m[5];
    for (int i = 0; i < 5; i++) {
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, szs[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        CK(vkCreateBuffer(dev, &bci, 0, &b[i]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b[i], &mr);
        uint32_t t = 0; for (; t < mp.memoryTypeCount; t++) if ((mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & 6) == 6) break;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, t };
        CK(vkAllocateMemory(dev, &mai, 0, &m[i])); CK(vkBindBufferMemory(dev, b[i], m[i], 0));
        void *ptr; CK(vkMapMemory(dev, m[i], 0, szs[i], 0, &ptr));
        if (i == 1 || i == 3) { float *f = ptr; for (VkDeviceSize j = 0; j < szs[i] / 4; j++) f[j] = 0.01f; }
        else { uint8_t *u = ptr; for (VkDeviceSize j = 0; j < szs[i]; j++) u[j] = (uint8_t) (j * 2654435761u >> 13); }
        vkUnmapMemory(dev, m[i]);
    }
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, sizeof gemv_spv, (const uint32_t *) gemv_spv };
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, 0, &sm));
    VkDescriptorSetLayoutBinding lb[5]; for (int i = 0; i < 5; i++) lb[i] = (VkDescriptorSetLayoutBinding) { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 5, lb };
    VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, 0, &dl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dl, 1, &pcr };
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, 0, &pl));
    VkSpecializationMapEntry me = { 0, 0, 4 };
    VkSpecializationInfo spi = { 1, &me, 4, &ROWS };
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &spi }, pl };
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, 0, 1, &cpi, 0, &pipe));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &ps };
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, dp, 1, &dl };
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo bi[5]; VkWriteDescriptorSet wd[5];
    for (int i = 0; i < 5; i++) { bi[i] = (VkDescriptorBufferInfo) { b[i], 0, szs[i] };
        wd[i] = (VkWriteDescriptorSet) { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bi[i] }; }
    vkUpdateDescriptorSets(dev, 5, wd, 0, 0);
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, qi };
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, 0, &cp));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fe; CK(vkCreateFence(dev, &fci, 0, &fe));
    const int REPS = 10; uint32_t pc[2] = { nb, NR }; double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        VkCommandBufferBeginInfo bgi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CK(vkBeginCommandBuffer(cb, &bgi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, 0);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
        for (int r = 0; r < REPS; r++) vkCmdDispatch(cb, (NR + ROWS - 1) / ROWS, 1, 1);
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb };
        double t0 = now();
        CK(vkQueueSubmit(q, 1, &si, fe)); CK(vkWaitForFences(dev, 1, &fe, 1, ~0ull)); CK(vkResetFences(dev, 1, &fe));
        double t = (now() - t0) / REPS; if (t < best) best = t;
        CK(vkResetCommandBuffer(cb, 0));
    }
    const double wbytes = (double) NR * nb * 18;  // 16 B nibbles + 2 B scale-equivalent, as in Q4_0 (4.5 bits/weight)
    printf("gemv K=%u rows=%u rows/wg=%u: %.3f ms  %.2f GB/s (Q4_0-equivalent bytes)  %.1f Gweights/s\n", K, NR, ROWS, best * 1e3,
           wbytes / best / 1e9, (double) NR * K / best / 1e9);
    return 0;
}
