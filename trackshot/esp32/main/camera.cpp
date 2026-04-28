// ESP includes
#include "esp_camera.h"
#include "esp_log.h"

// Project includes
#include "camera.h"

static const char *TAG = "CAMERA";

static camera_config_t get_camera_config()
{
    camera_config_t config = {};

    // LEDC (for XCLK)
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    // Data pins
    config.pin_d0 = 15; // Y2_GPIO_NUM
    config.pin_d1 = 17; // Y3_GPIO_NUM
    config.pin_d2 = 18; // Y4_GPIO_NUM
    config.pin_d3 = 16; // Y5_GPIO_NUM
    config.pin_d4 = 14; // Y6_GPIO_NUM
    config.pin_d5 = 12; // Y7_GPIO_NUM
    config.pin_d6 = 11; // Y8_GPIO_NUM
    config.pin_d7 = 48; // Y9_GPIO_NUM

    // Control pins
    config.pin_xclk  = 10; // XCLK_GPIO_NUM
    config.pin_pclk  = 13; // PCLK_GPIO_NUM
    config.pin_vsync = 38; // VSYNC_GPIO_NUM
    config.pin_href  = 47; // HREF_GPIO_NUM

    // SCCB (I2C) pins
    config.pin_sccb_sda = 40; // SIOD_GPIO_NUM
    config.pin_sccb_scl = 39; // SIOC_GPIO_NUM

    // Power-down and reset (not used on XIAO)
    config.pin_pwdn  = -1; // PWDN_GPIO_NUM
    config.pin_reset = -1; // RESET_GPIO_NUM

    // Clock
    config.xclk_freq_hz = 16000000; // 16 MHz

    // Frame buffer settings
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 1;
    config.grab_mode   = CAMERA_GRAB_LATEST;

    // Pixel format: RGB565 is efficient to convert to RGB888
    config.pixel_format = PIXFORMAT_RGB565;

    // Resolution: QVGA (320x240)
    config.frame_size = FRAMESIZE_QVGA;

    // JPEG-specific fields (ignored for RGB565 but set to sane defaults)
    config.jpeg_quality = 12;

    return config;
}

/**
 * Convert an RGB565 frame buffer to a model input tensor.
 * @param fb Pointer to the camera frame buffer in RGB565 format.
 * @param frame_tensor Pointer to the output tensor buffer (must be at least (TENSOR_W * TENSOR_H * TENSOR_C) bytes).
 * @return true on success, false on failure.
 */
static bool fb_to_tensor_rgb(const camera_fb_t *fb, int8_t *frame_tensor)
{
    // Validate input
    if (!fb || fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Frame buffer is null or not RGB565 (fmt=%d)", fb ? fb->format : -1);
        return false;
    }

    // Get source dimensions
    const int src_w = static_cast<int>(fb->width);
    const int src_h = static_cast<int>(fb->height);
    if (src_w <= 0 || src_h <= 0) {
        ESP_LOGE(TAG, "Invalid fb size: %dx%d", src_w, src_h);
        return false;
    }

    // Pointer to RGB565 pixel data
    const uint8_t* src = fb->buf;
    const int min_dim = (src_w < src_h) ? src_w : src_h;
    const int offset_x = (src_w  - min_dim) / 2;
    const int offset_y = (src_h  - min_dim) / 2;

    // Iterate over output tensor pixels in y direction
    for (int oy = 0; oy < TENSOR_H; ++oy) {
        // Map output y to input y within the square crop
        int sy = offset_y + (oy * min_dim) / TENSOR_H;
        if (sy < 0) sy = 0;
        if (sy >= src_h) sy = src_h - 1;

        // Iterate over output tensor pixels in x direction
        for (int ox = 0; ox < TENSOR_W; ++ox) {
            // Map output x to input x within the square crop
            int sx = offset_x + (ox * min_dim) / TENSOR_W;
            if (sx < 0) sx = 0;
            if (sx >= src_w) sx = src_w - 1;

            // Convert RGB565 pixel from source to RGB888 then center to int8 [-128, 127]
            const int i = 2 * (sy * src_w + sx);
            const uint16_t p = static_cast<uint16_t>(src[i]) |
                               (static_cast<uint16_t>(src[i + 1]) << 8);
            const uint8_t r = static_cast<uint8_t>((p >> 11) & 0x1F) << 3;
            const uint8_t g = static_cast<uint8_t>((p >> 5) & 0x3F) << 2;
            const uint8_t b = static_cast<uint8_t>(p & 0x1F) << 3;

            const int8_t r8 = static_cast<int8_t>(static_cast<int>(r) - 128);
            const int8_t g8 = static_cast<int8_t>(static_cast<int>(g) - 128);
            const int8_t b8 = static_cast<int8_t>(static_cast<int>(b) - 128);

            // Store in output tensor (HWC format)
            const int dst_index = (oy * TENSOR_W + ox) * TENSOR_C;
            frame_tensor[dst_index + 0] = r8;
            frame_tensor[dst_index + 1] = g8;
            frame_tensor[dst_index + 2] = b8;
        }
    }

    return true;
}

/**
 * Initialize the camera.
 * @return true on success, false on failure.
 */
bool camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera...");

    camera_config_t camera_config = get_camera_config();
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x.", err);
        return false;
    }

    ESP_LOGI(TAG, "Camera initialized: QVGA RGB565 -> %dx%dx%d tensor.",
             TENSOR_W, TENSOR_H, TENSOR_C);

    return true;
}

/**
 * Capture a frame and convert to tensor of size TENSOR_W x TENSOR_H x TENSOR_C.
 * @param frame_tensor Pointer to the output tensor buffer (must be at least TENSOR_W x TENSOR_H x TENSOR_C bytes).
 * @return true on success, false on failure.
 */
bool camera_capture_frame(int8_t *frame_tensor)
{
    // Capture a frame from the camera
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to get frame buffer.");
        return false;
    }

    // Convert the frame buffer to a model input tensor
    bool success = fb_to_tensor_rgb(fb, frame_tensor);
    if (!success) {
        ESP_LOGE(TAG, "Failed to convert frame to tensor.");
    }

    // Return the frame buffer to the driver
    esp_camera_fb_return(fb);

    return success;
}
