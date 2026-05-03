import sys
import numpy as np

# A quick script to list operators
with open("checkpoints/best_saved_model/best_full_integer_quant.tflite", "rb") as f:
    content = f.read()
    print(f"model size: {len(content)}")
