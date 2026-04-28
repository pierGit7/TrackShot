#pragma once

#include <cstdint>

bool camera_init(void);
bool camera_capture_frame(int8_t *frame_tensor);

// Keep ESP32 preprocessing aligned with project training config:
// configs/data/default.yaml -> img_size: 96
#define TENSOR_W 96
#define TENSOR_H 96
#define TENSOR_C 3