import struct
import json
import urllib.request
import os

def read_tflite_shape(file_path):
    print("Reading", file_path)
    os.system(f"python3 -c \"import sys; sys.path.append('/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages'); import tensorflow as tf; interpreter = tf.lite.Interpreter(model_path='{file_path}'); print('IN:', interpreter.get_input_details()[0]['shape']); print('OUT:', interpreter.get_output_details()[0]['shape'])\"")

read_tflite_shape("checkpoints/best_saved_model/best_int8.tflite")
