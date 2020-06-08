import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("export_dir")
lite_model = converter.convert()
open("model.tflite", "wb").write(lite_model)