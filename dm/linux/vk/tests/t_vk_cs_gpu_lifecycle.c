#include <stdio.h>
#include "vk_cs_gpu_lifecycle.h"

static int failures;
#define CHECK(msg, x) do { if (!(x)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    vkr_cs_gpu_batch b = {0};

    CHECK("first recording token", vkr_cs_gpu_record(&b, 2) == 0);
    CHECK("second recording token", vkr_cs_gpu_record(&b, 2) == 1);
    CHECK("recording bound", vkr_cs_gpu_record(&b, 2) < 0);
    CHECK("in-progress recording is never harvestable", !vkr_cs_gpu_harvestable(&b, UINT64_MAX));

    CHECK("publish moves the complete batch", vkr_cs_gpu_publish(&b, 19) == 2);
    CHECK("published count", b.recording == 0 && b.submitted == 2);
    CHECK("published batch refuses metadata overwrite", vkr_cs_gpu_record(&b, 2) < 0);
    CHECK("published batch refuses replacement", vkr_cs_gpu_publish(&b, 23) < 0 && b.timeline == 19);
    CHECK("earlier completion does not harvest", !vkr_cs_gpu_harvestable(&b, 18));
    CHECK("exact submission completion harvests", vkr_cs_gpu_harvestable(&b, 19));
    vkr_cs_gpu_retire(&b);

    CHECK("reused slot records a new batch", vkr_cs_gpu_record(&b, 2) == 0);
    CHECK("old completed value cannot harvest the reuse", !vkr_cs_gpu_harvestable(&b, 19));
    CHECK("new batch publishes its own timeline", vkr_cs_gpu_publish(&b, 37) == 1);
    CHECK("stale completion still cannot harvest", !vkr_cs_gpu_harvestable(&b, 19));
    CHECK("new completion harvests", vkr_cs_gpu_harvestable(&b, 37));
    vkr_cs_gpu_retire(&b);

    CHECK("empty submission publishes nothing", vkr_cs_gpu_publish(&b, 51) == 0 && !b.timeline);
    CHECK("zero timeline is rejected", vkr_cs_gpu_record(&b, 2) == 0 && vkr_cs_gpu_publish(&b, 0) < 0);
    CHECK("failed publish preserves recording", b.recording == 1 && !b.submitted && !b.timeline);

    printf("vk compute GPU lifecycle: %s (%d failures)\n", failures ? "FAILED" : "passed", failures);
    return failures != 0;
}
