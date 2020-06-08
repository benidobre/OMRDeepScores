import os
import tensorflow as tf

from tensorflow_core.lite.python.lite import TFLiteConverter

keras_file = "linear.h5"

converter = TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("omr5.tflite", "wb").write(tflite_model)
