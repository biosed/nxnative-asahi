#pragma once

#include <stdint.h>

/*
 * Until the Linux replay owns real occlusion queries, match the existing
 * Android Vulkan backend: a completed samples-passed report is conservatively
 * visible.  One is enough; zero is the only value that would cull the draw.
 */
#define VKR_SAMPLES_VISIBLE 1ull

static inline uint64_t
vkr_counter_value(uint64_t type, uint64_t timestamp, int render_cond,
                  uint64_t samples_passed_type)
{
   if (type == samples_passed_type)
      return VKR_SAMPLES_VISIBLE;

   if ((type == 0 || type == 0x10) && !render_cond)
      return timestamp;

   return 0;
}
