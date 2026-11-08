#include <stdint.h>
#include <stdio.h>

#include "vk_counter_report.h"

static int failures;

#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #x); \
      failures++; \
   } \
} while (0)

int
main(void)
{
   const uint64_t samples_passed = 7;
   const uint64_t timestamp = 0x123456789abcdef0ull;

   CHECK(VKR_SAMPLES_VISIBLE == 1);
   CHECK(vkr_counter_value(samples_passed, timestamp, 0,
                           samples_passed) == 1);
   CHECK(vkr_counter_value(samples_passed, timestamp, 1,
                           samples_passed) == 1);
   CHECK(vkr_counter_value(0, timestamp, 0, samples_passed) == timestamp);
   CHECK(vkr_counter_value(0x10, timestamp, 0, samples_passed) == timestamp);
   CHECK(vkr_counter_value(0, timestamp, 1, samples_passed) == 0);
   CHECK(vkr_counter_value(0x10, timestamp, 1, samples_passed) == 0);
   CHECK(vkr_counter_value(3, timestamp, 0, samples_passed) == 0);

   if (failures) {
      fprintf(stderr, "t_counter_report: failed (%d failures)\n", failures);
      return 1;
   }

   puts("t_counter_report: passed (0 failures)");
   return 0;
}
