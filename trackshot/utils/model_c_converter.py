import torch
import torchvision
import litert_torch
import os 

def write_model_c_file(path: str, tflite_model):
    # Ensure that the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write source file
    with open(path, "w") as c_file:
        c_file.write("const unsigned char model_binary[] = {\n")
        for i, byte in enumerate(tflite_model):
            c_file.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                c_file.write("\n")
        c_file.write("\n};\n")


with open("checkpoints/best_saved_model/best_full_integer_quant.tflite", "rb") as f:
    tflite_model = f.read()

write_model_c_file("trackshot/esp32/model/model.c", tflite_model)