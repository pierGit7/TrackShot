#pragma once

#include <cstdint>

#include "esp_camera.h"

bool camera_init(void);
camera_fb_t* camera_capture_frame(int8_t *frame_tensor);
void camera_return_frame(camera_fb_t *fb);

bool camera_get_jpeg(const camera_fb_t *fb, uint8_t **out_jpg, size_t *out_len);
void camera_free_jpeg(uint8_t *jpg);

// Keep ESP32 preprocessing aligned with project training config:
// configs/data/default.yaml -> img_size: 96
#define TENSOR_W 96
#define TENSOR_H 96
#define TENSOR_C 3