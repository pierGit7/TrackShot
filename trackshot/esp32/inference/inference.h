#pragma once

#include <cstdint>

bool inference_init();
bool inference_run(const int8_t *image_tensor);