import os
import cv2
import glob
import random
from pathlib import Path

def process_split(split_name, img_size=128):
    in_img_dir = f"data/raw/tracking-soccerball-yolov8/dataset/{split_name}/images"
    in_lbl_dir = f"data/raw/tracking-soccerball-yolov8/dataset/{split_name}/labels"
    
    out_img_dir = f"data/processed/tiles/{split_name}/images"
    out_lbl_dir = f"data/processed/tiles/{split_name}/labels"
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    images = glob.glob(os.path.join(in_img_dir, "*.jpg"))
    print(f"Found {len(images)} images in {split_name} split.")
    
    crop_count = 0
    bg_count = 0
    
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        base_name = Path(img_path).stem
        lbl_path = os.path.join(in_lbl_dir, base_name + ".txt")
        
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            # Background
            x_start = random.randint(0, w - img_size)
            y_start = random.randint(0, h - img_size)
            crop = img[y_start:y_start+img_size, x_start:x_start+img_size]
            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_bg.jpg"), crop)
            open(os.path.join(out_lbl_dir, f"{base_name}_bg.txt"), 'w').close()
            bg_count += 1
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls_id = parts[0]
            cx, cy, bw, bh = map(float, parts[1:])
            
            px_cx, px_cy = int(cx * w), int(cy * h)
            px_bw, px_bh = int(bw * w), int(bh * h)
            
            # Determine crop bounds (add random jitter so it is not always dead center)
            jitter_x = random.randint(-img_size//4, img_size//4)
            jitter_y = random.randint(-img_size//4, img_size//4)
            
            x_start = px_cx - img_size//2 + jitter_x
            y_start = px_cy - img_size//2 + jitter_y
            
            # Clamp to image limits
            x_start = max(0, min(x_start, w - img_size))
            y_start = max(0, min(y_start, h - img_size))
            
            crop = img[y_start:y_start+img_size, x_start:x_start+img_size]
            
            # Relabel
            obj_x1, obj_y1 = px_cx - px_bw/2, px_cy - px_bh/2
            obj_x2, obj_y2 = px_cx + px_bw/2, px_cy + px_bh/2
            
            crop_x1, crop_y1 = x_start, y_start
            crop_x2, crop_y2 = x_start + img_size, y_start + img_size
            
            inter_x1 = max(obj_x1, crop_x1)
            inter_y1 = max(obj_y1, crop_y1)
            inter_x2 = min(obj_x2, crop_x2)
            inter_y2 = min(obj_y2, crop_y2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                new_cx = ((inter_x1 + inter_x2) / 2.0 - x_start) / img_size
                new_cy = ((inter_y1 + inter_y2) / 2.0 - y_start) / img_size
                new_w = (inter_x2 - inter_x1) / img_size
                new_h = (inter_y2 - inter_y1) / img_size
                
                crop_name = f"{base_name}_crop_{i}"
                cv2.imwrite(os.path.join(out_img_dir, f"{crop_name}.jpg"), crop)
                with open(os.path.join(out_lbl_dir, f"{crop_name}.txt"), 'w') as lf:
                    lf.write(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
                crop_count += 1
                
    print(f"Done {split_name}. Created {crop_count} foreground crops and {bg_count} backgrounds.")

random.seed(42)
process_split("Train", 128)
process_split("Validation", 128)
