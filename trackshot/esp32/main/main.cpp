#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "nvs_flash.h"

// Project includes
#include "camera.h"
#include "../inference/inference.h"

static const char *TAG = "MAIN";
static int8_t image_tensor[TENSOR_W * TENSOR_H * TENSOR_C]; // all pixels

static std::string base64_encode(const uint8_t *data, size_t length)
{
    static const char *b64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((length + 2) / 3) * 4);
    
    uint32_t val = 0;
    int valb = -6;
    for (size_t i = 0; i < length; i++) {
        val = (val << 8) + data[i];
        valb += 8;
        while (valb >= 0) {
            out.push_back(b64_table[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(b64_table[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}


static QueueHandle_t result_queue;

struct InferenceResult {
    uint8_t *jpg_buf;
    size_t jpg_len;
    std::vector<BoundingBox> boxes;
    camera_fb_t *fb;
    int src_w;
    int src_h;
};

// Worker task pinned to Core 1 (APP_CPU) to handle Base64 and string formatting
void output_worker_task(void *pvParameters)
{
    InferenceResult res;
    while (xQueueReceive(result_queue, &res, portMAX_DELAY) == pdTRUE) {
        int64_t t_start = esp_timer_get_time();
        
        // Encode the JPEG explicitly to a Base64 string
        std::string b64 = base64_encode(res.jpg_buf, res.jpg_len);
        
        // Build JSON output. We need: [x, y, w, h, "object", confidence]
        std::string json_boxes = "[";
        
        int min_dim = (res.src_w < res.src_h) ? res.src_w : res.src_h;
        int offset_x = (res.src_w - min_dim) / 2;
        int offset_y = (res.src_h - min_dim) / 2;

        for (size_t i = 0; i < res.boxes.size(); ++i) {
            const auto& b = res.boxes[i];
            
            // Scale from tensor coords to original image pixels
            float cx_img = offset_x + (b.cx / (float)TENSOR_W) * min_dim;
            float cy_img = offset_y + (b.cy / (float)TENSOR_H) * min_dim;
            float w_img = (b.w / (float)TENSOR_W) * min_dim;
            float h_img = (b.h / (float)TENSOR_H) * min_dim;
            
            // Convert center to top-left
            float x_tl = cx_img - w_img / 2.0f;
            float y_tl = cy_img - h_img / 2.0f;

            char buf[128];
            snprintf(buf, sizeof(buf), "[%.2f, %.2f, %.2f, %.2f, \"object\", %.3f]",
                     x_tl, y_tl, w_img, h_img, b.confidence);
            json_boxes += buf;
            if (i < res.boxes.size() - 1) json_boxes += ", ";
        }
        json_boxes += "]";
        
        // Print out single JSON line string
        printf("{\"image\":\"%s\",\"boxes\":%s}\n", b64.c_str(), json_boxes.c_str());
        
        int64_t t_end = esp_timer_get_time();
        ESP_LOGI("WORKER", "I/O Task took %lld ms", (t_end - t_start) / 1000);

        // Cleanup
        camera_free_jpeg(res.jpg_buf);
        camera_return_frame(res.fb);
    }
}

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
    
    // Create queue and pin worker task to APP_CPU (Core 1)
    result_queue = xQueueCreate(2, sizeof(InferenceResult));
    xTaskCreatePinnedToCore(output_worker_task, "output_worker", 8192, NULL, 5, NULL, 1);


    if (!inference_init()) {
        ESP_LOGE(TAG, "Inference init failed.");
        abort();
    }

    ESP_LOGI(TAG, "Setup complete. Capturing %dx%dx%d tensors and printing model output.",
             TENSOR_W, TENSOR_H, TENSOR_C);
}

void loop(void)
{
    int64_t t_start = esp_timer_get_time();

    camera_fb_t *fb = camera_capture_frame(image_tensor);
    int64_t t_cam = esp_timer_get_time();

    if (fb) {
        std::vector<BoundingBox> boxes;
        if (!inference_run(image_tensor, boxes)) {
            ESP_LOGE(TAG, "Inference run failed.");
        }
        int64_t t_infer = esp_timer_get_time();
        
        uint8_t *jpg_buf = nullptr;
        size_t jpg_len = 0;
        
        if (camera_get_jpeg(fb, &jpg_buf, &jpg_len)) {
            int64_t t_jpeg = esp_timer_get_time();

            // Dispatch to worker thread
            InferenceResult res;
            res.jpg_buf = jpg_buf;
            res.jpg_len = jpg_len;
            res.boxes = boxes;
            res.fb = fb;
            res.src_w = fb->width;
            res.src_h = fb->height;
            
            if (xQueueSend(result_queue, &res, 0) != pdTRUE) {
                ESP_LOGW(TAG, "Queue full, dropping frame output");
                camera_free_jpeg(jpg_buf);
                camera_return_frame(fb);
            }
        } else {
            ESP_LOGE(TAG, "Failed to encode JPEG.");
            camera_return_frame(fb);
        }
    }

    // Give it a tiny bit of breathing room instead of 1 second so that you can see a stream
    vTaskDelay(pdMS_TO_TICKS(100));
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    while (true) {
        loop();
    }
}
