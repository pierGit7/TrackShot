import cv2
import glob
import os
import numpy as np


def check_bbox_sizes():
    image_dir = "data/raw/tracking-soccerball-yolov8/dataset/Train/images"
    label_dir = "data/raw/tracking-soccerball-yolov8/dataset/Train/labels"

    # Get first image to determine original dimensions
    img_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not img_files:
        print("No images found.")
        return

    img = cv2.imread(img_files[0])
    h_orig, w_orig = img.shape[:2]
    print(f"Original Image Dimensions: {w_orig}x{h_orig}")

    target_size = 96

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    widths = []
    heights = []

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    nw = float(parts[3])
                    nh = float(parts[4])

                    # Calculate size in 96x96 pixels
                    w_96 = nw * target_size
                    h_96 = nh * target_size

                    widths.append(w_96)
                    heights.append(h_96)

    if not widths:
        print("No labels found.")
        return

    print(f"Stats for {target_size}x{target_size} resolution:")
    print(f"Min Width: {min(widths):.2f} px")
    print(f"Max Width: {max(widths):.2f} px")
    print(f"Avg Width: {np.mean(widths):.2f} px")
    print(f"Min Height: {min(heights):.2f} px")
    print(f"Max Height: {max(heights):.2f} px")
    print(f"Avg Height: {np.mean(heights):.2f} px")

    # Count how many are smaller than 1 pixel
    tiny_count = sum(1 for w in widths if w < 1.0)
    print(f"Boxes narrower than 1px: {tiny_count}/{len(widths)}")


if __name__ == "__main__":
    check_bbox_sizes()
