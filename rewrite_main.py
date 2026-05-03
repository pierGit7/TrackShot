import re

with open('trackshot/esp32/main/main.cpp', 'r') as f:
    content = f.read()

# 1. Add queue variable and struct before setup
queue_code = """
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
            snprintf(buf, sizeof(buf), "[%.2f, %.2f, %.2f, %.2f, \\"object\\", %.3f]",
                     x_tl, y_tl, w_img, h_img, b.confidence);
            json_boxes += buf;
            if (i < res.boxes.size() - 1) json_boxes += ", ";
        }
        json_boxes += "]";
        
        // Print out single JSON line string
        printf("{\\"image\\":\\"%s\\",\\"boxes\\":%s}\\n", b64.c_str(), json_boxes.c_str());
        
        int64_t t_end = esp_timer_get_time();
        ESP_LOGI("WORKER", "I/O Task took %lld ms", (t_end - t_start) / 1000);

        // Cleanup
        camera_free_jpeg(res.jpg_buf);
        camera_return_frame(res.fb);
    }
}

void setup()"""

content = content.replace("void setup()", queue_code)

# 2. Setup additions
setup_additions = """    if (!camera_init()) {
        abort();
    }
    
    // Create queue and pin worker task to APP_CPU (Core 1)
    result_queue = xQueueCreate(2, sizeof(InferenceResult));
    xTaskCreatePinnedToCore(output_worker_task, "output_worker", 8192, NULL, 5, NULL, 1);
"""
content = re.sub(r'    if \(\!camera_init\(\)\) \{\n        abort\(\);\n    \}', setup_additions, content)

# 3. Replace processing in loop
loop_repl_start = content.index("            // Encode the JPEG explicitly to a Base64 string")
loop_repl_end = content.index("        camera_return_frame(fb);\n    }")

process_code = """            // Dispatch to worker thread
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
        }\n"""

content = content[:loop_repl_start] + process_code + content[loop_repl_end + len("        camera_return_frame(fb);\n    }"):]

with open('trackshot/esp32/main/main.cpp', 'w') as f:
    f.write(content)

