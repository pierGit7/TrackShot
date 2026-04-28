#include <cstdio>
#include <cstdint>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

// Project includes
#include "camera.h"
#include "../inference/inference.h"

static const char *TAG = "MAIN";
static int8_t image_tensor[TENSOR_W * TENSOR_H * TENSOR_C]; // all pixels

void setup()
{
    // Initialize NVS (required by some drivers) non volatile storage
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // Initialize camera
    // Do this before loading the model - the reverse doesn't work: Due to the pre-existing TFLM allocation, the camera
    // buffers probably land in a memory spot that has slightly higher latency or access conflicts.
    if (!camera_init()) {
        abort();
    }

    if (!inference_init()) {
        ESP_LOGE(TAG, "Inference init failed.");
        abort();
    }

    ESP_LOGI(TAG, "Setup complete. Capturing %dx%dx%d tensors and printing model output.",
             TENSOR_W, TENSOR_H, TENSOR_C);
}

void loop(void)
{
    if (camera_capture_frame(image_tensor)) {
        if (!inference_run(image_tensor)) {
            ESP_LOGE(TAG, "Inference run failed.");
        }
    }

    vTaskDelay(pdMS_TO_TICKS(1000));
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    while (true) {
        loop();
    }
}
