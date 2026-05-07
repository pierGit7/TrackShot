#include "inference.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "../main/camera.h"
#include "../model/model.h"

// Include ESP-NN Optimizations
#include "esp_nn.h"

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

static float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
    float x1_min = box1.cx - box1.w / 2.0f;
    float y1_min = box1.cy - box1.h / 2.0f;
    float x1_max = box1.cx + box1.w / 2.0f;
    float y1_max = box1.cy + box1.h / 2.0f;

    float x2_min = box2.cx - box2.w / 2.0f;
    float y2_min = box2.cy - box2.h / 2.0f;
    float x2_max = box2.cx + box2.w / 2.0f;
    float y2_max = box2.cy + box2.h / 2.0f;

    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    if (inter_x_max <= inter_x_min || inter_y_max <= inter_y_min) {
        return 0.0f;
    }

    float inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    float box1_area = box1.w * box1.h;
    float box2_area = box2.w * box2.h;

    return inter_area / (box1_area + box2_area - inter_area);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// YOLO-style models typically need far more than 30KB.
#define TENSOR_ARENA_SIZE (185 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t *tensor_arena = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG_INF = "Inference";

bool inference_init()
{
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        return false;
    }

    tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!tensor_arena)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensor arena in PSRAM");
        return false;
    }

    static tflite::MicroMutableOpResolver<24> resolver;
    static bool resolver_initialized = false;
    
    if (!resolver_initialized)
    {
        // Register a broad set of ops commonly used by exported detection models.
        if (resolver.AddConv2D() != kTfLiteOk ||
            resolver.AddDepthwiseConv2D() != kTfLiteOk ||
            resolver.AddFullyConnected() != kTfLiteOk ||
            resolver.AddMaxPool2D() != kTfLiteOk ||
            resolver.AddAveragePool2D() != kTfLiteOk ||
            resolver.AddReshape() != kTfLiteOk ||
            resolver.AddTranspose() != kTfLiteOk ||
            resolver.AddConcatenation() != kTfLiteOk ||
            resolver.AddSlice() != kTfLiteOk ||
            resolver.AddSplit() != kTfLiteOk ||
            resolver.AddSplitV() != kTfLiteOk ||
            resolver.AddResizeNearestNeighbor() != kTfLiteOk ||
            resolver.AddPad() != kTfLiteOk ||
            resolver.AddMul() != kTfLiteOk ||
            resolver.AddAdd() != kTfLiteOk ||
            resolver.AddSub() != kTfLiteOk ||
            resolver.AddLogistic() != kTfLiteOk ||
            resolver.AddSoftmax() != kTfLiteOk ||
            resolver.AddStridedSlice() != kTfLiteOk ||
            resolver.AddDequantize() != kTfLiteOk ||
            resolver.AddQuantize() != kTfLiteOk)
        {
            ESP_LOGE(TAG_INF, "Failed to register one or more TFLM ops");
            return false;
        }
        resolver_initialized = true;
    }
    ESP_LOGI(TAG_INF, "Free PSRAM: %d", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG_INF, "Largest PSRAM block: %d", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
        return false;
    }

    ESP_LOGI(TAG_INF, "Tensor Arena used bytes: %d", interpreter->arena_used_bytes());

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input->dims->size != 4)
    {
        ESP_LOGE(TAG_INF, "Unexpected input dims size: %d", input->dims->size);
        return false;
    }

    if (input->dims->data[1] != TENSOR_H || input->dims->data[2] != TENSOR_W || input->dims->data[3] != TENSOR_C)
    {
        ESP_LOGE(
            TAG_INF,
            "Input shape mismatch. Model expects [%d,%d,%d], camera provides [%d,%d,%d]",
            input->dims->data[1], input->dims->data[2], input->dims->data[3],
            TENSOR_H, TENSOR_W, TENSOR_C);
        return false;
    }

    ESP_LOGI(
        TAG_INF,
        "Model input: %s [%d,%d,%d,%d]",
        TfLiteTypeGetName(input->type),
        input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG_INF, "Model output: %s (dims=%d)", TfLiteTypeGetName(output->type), output->dims->size);

    return true;
}

bool inference_run(const int8_t *image_tensor, std::vector<BoundingBox>& out_boxes)
{
    if (!interpreter || !input || !output || !image_tensor)
    {
        ESP_LOGE(TAG_INF, "Inference not initialized or bad input tensor");
        return false;
    }

    const int in_elements = TENSOR_W * TENSOR_H * TENSOR_C;
    if (input->type == kTfLiteInt8)
    {
        memcpy(input->data.int8, image_tensor, in_elements);
    }
    else if (input->type == kTfLiteFloat32)
    {
        for (int i = 0; i < in_elements; ++i)
        {
            input->data.f[i] = static_cast<float>(static_cast<int>(image_tensor[i]) + 128) / 255.0f;
        }
    }
    else
    {
        ESP_LOGE(TAG_INF, "Unsupported input tensor type: %s", TfLiteTypeGetName(input->type));
        return false;
    }

    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Inference invoke failed");
        return false;
    }

    // Fix: Don't require output dims to have specific shape - just check we have data
    if (output->dims->size < 2) {
        ESP_LOGE(TAG_INF, "Unexpected output shape! dims: %d", output->dims->size);
        return false;
    }

    // Calculate num_anchors - handle different possible shapes
    int num_anchors = 0;
    if (output->dims->size == 3 && output->dims->data[1] == 5) {
        // Shape [1, 5, N]
        num_anchors = output->dims->data[2];
    } else if (output->dims->size == 2) {
        // Shape [1, N*5] or [N, 5]
        int total_values = output->dims->data[1];
        if (total_values % 5 == 0) {
            num_anchors = total_values / 5;
        }
    } else {
        // Flat output - calculate from bytes
        int total_values = output->bytes / sizeof(float);
        if (total_values % 5 == 0) {
            num_anchors = total_values / 5;
        }
    }
    
    if (num_anchors == 0) {
        ESP_LOGE(TAG_INF, "Could not determine number of anchors");
        return false;
    }
    
    ESP_LOGI(TAG_INF, "num_anchors: %d", num_anchors);
    
    const float conf_threshold = 0.90f; 
    std::vector<BoundingBox> valid_boxes;

    // Dequantization lambda function
    auto get_val = [&](int index) -> float {
        if (output->type == kTfLiteFloat32) {
            return output->data.f[index];
        } else if (output->type == kTfLiteInt8) {
            return (static_cast<float>(output->data.int8[index]) - static_cast<float>(output->params.zero_point)) * output->params.scale;
        } else if (output->type == kTfLiteUInt8) {
            return (static_cast<float>(output->data.uint8[index]) - static_cast<float>(output->params.zero_point)) * output->params.scale;
        }
        return 0.0f;
    };
    
    // Debug: Print first few anchors
    ESP_LOGI(TAG_INF, "First 10 anchors (after sigmoid):");
    for (int i = 0; i < 10 && i < num_anchors; i++) {
        int base = i * 5;
        ESP_LOGI(TAG_INF, "Anchor %d: cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f",
                 i, 
                 sigmoid(get_val(base+0)),
                 sigmoid(get_val(base+1)),
                 sigmoid(get_val(base+2)),
                 sigmoid(get_val(base+3)),
                 sigmoid(get_val(base+4)));
    }
    
    // Process all anchors - FIXED: Use sigmoid for ALL values (no expf!)
    for (int i = 0; i < num_anchors; ++i) {
        int base_idx = i * 5;

        // Raw values from model
        float cx_raw = get_val(base_idx + 0);
        float cy_raw = get_val(base_idx + 1);
        float w_raw  = get_val(base_idx + 2);
        float h_raw  = get_val(base_idx + 3);
        float conf_raw = get_val(base_idx + 4);

        // Apply sigmoid to ALL values (based on your debug showing values in 0-1 range)
        float cx = sigmoid(cx_raw);
        float cy = sigmoid(cy_raw);
        float w = sigmoid(w_raw);      // Changed from expf to sigmoid
        float h = sigmoid(h_raw);      // Changed from expf to sigmoid
        float conf = sigmoid(conf_raw);

        if (conf > conf_threshold) {
            valid_boxes.push_back({cx, cy, w, h, conf});
        }
    }

    ESP_LOGI(TAG_INF, "Detections before NMS: %d", (int)valid_boxes.size());

    // Sort by confidence (highest to lowest) for NMS
    std::sort(valid_boxes.begin(), valid_boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.confidence > b.confidence;
    });

    // Apply Non-Maximum Suppression (NMS)
    const float nms_iou_threshold = 0.45f;
    std::vector<BoundingBox> final_boxes;
    for (const auto& box : valid_boxes) {
        bool keep = true;
        for (const auto& kept_box : final_boxes) {
            if (calculate_iou(box, kept_box) > nms_iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            final_boxes.push_back(box);
        }
    }

    ESP_LOGI(TAG_INF, "Detections after NMS: %d", (int)final_boxes.size());
    
    out_boxes = final_boxes;
    return true;
}