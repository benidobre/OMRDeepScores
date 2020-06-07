"""Convolutional Neural Network Estimator for DeepScores Classification, built with Tensorflow
"""

import argparse
import sys

import tensorflow as tf
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.lite.python.lite import TFLiteConverter

import Classification_BatchDataset


FLAGS = None


def main(unused_argv):
    print("Setting up image reader...")
    data_reader = Classification_BatchDataset.class_dataset_reader(FLAGS.data_dir)
    data_reader.read_images()

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(data_reader.tile_size[0], data_reader.tile_size[1],1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(data_reader.nr_classes, activation=tf.nn.softmax)
    ])
    keep_prob = tf.placeholder(tf.float32)
    drop = 0.25
    downsample = 7*4*256
    # Define the model architecture
    model2 = keras.Sequential([
        keras.layers.InputLayer(input_shape=(data_reader.tile_size[0], data_reader.tile_size[1],1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(drop),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(drop),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(drop),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(drop),
        keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(drop),
        keras.layers.Flatten(),
        keras.layers.Dense(downsample, activation=tf.nn.relu),
        keras.layers.Dense(data_reader.nr_classes)
    ])
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        shear_range=0.25,
        zoom_range=0.2
    )
    # Define how to train the model
    model2.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the digit classification model
    train_images, train_labels = data_reader.get_records()
    test_images, test_labels = data_reader.get_test_records()
    del data_reader

    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(np.squeeze(train_images[i], axis=2), cmap=plt.cm.gray)
    #     plt.xlabel(train_labels[i])
    # plt.show()

    train_generator = datagen.flow(train_images, train_labels, seed=4, batch_size=100)
    del train_images
    del train_labels

    # augmented_images, augmented_labels = next(train_generator)
    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(np.squeeze(augmented_images[70+i], axis=2), cmap=plt.cm.gray)
    #     plt.xlabel('Label: %d' % augmented_labels[i])
    # plt.show()
    # train_labels = np.array([np.where(r == 1)[0][0] for r in a])

    test_generator = datagen.flow(test_images, test_labels, seed=4)
    del test_images
    del test_labels

    model2.fit_generator(train_generator, epochs=5, validation_data=test_generator)
    # test_labels = np.array([np.where(r == 1)[0][0] for r in b])
    # test_loss, test_acc = model.evaluate(test_images, test_labels)

    # print('Test accuracy:', test_acc)

    keras_file = "linear.h5"
    keras.models.save_model(model2, keras_file)

    # converter = TFLiteConverter.from_keras_model_file(keras_file)
    # tflite_model = converter.convert()
    # open("linear.tflite", "wb").write(tflite_model)


    # Convert Keras model to TF Lite format.
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_quantized_model = converter.convert()
    #
    # f = open('mnist.tflite', "wb")
    # f.write(tflite_quantized_model)
    # f.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/home/beni/Downloads/DeepScores2017_classification',
                      help='Directory for storing input data')
  parser.add_argument("--batch_size", type=int, default=2, help="batch size for training")
  parser.add_argument("--test_batch_size", type=int, default=200, help="batch size for training")
  parser.add_argument("--model_path", type=str, default="Models/deepscores_class.ckpt",
                      help="where to store the trained model")

  FLAGS, unparsed = parser.parse_known_args()
  main([sys.argv[0]] + unparsed)
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)