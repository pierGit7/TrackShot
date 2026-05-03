#pragma once

#include <cstdint>
#include <vector>

struct BoundingBox {
    float cx;
    float cy;
    float w;
    float h;
    float confidence;
};

bool inference_init();
bool inference_run(const int8_t *image_tensor, std::vector<BoundingBox>& out_boxes);