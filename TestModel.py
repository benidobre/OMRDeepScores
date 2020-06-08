import argparse
import sys

import tensorflow as tf
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_core.lite.python.lite import TFLiteConverter

import Classification_BatchDataset


def pp():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Add a color dimension to the images in "train" and "validate" dataset to
    # leverage Keras's data augmentation utilities later.
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.2
    )

    # Generate augmented data from MNIST dataset
    train_generator = datagen.flow(train_images, train_labels)
    test_generator = datagen.flow(test_images, test_labels)

    model.fit(train_generator, epochs=1, validation_data=test_generator)


def main(unused_argv):
    print("Setting up image reader...")
    data_reader = Classification_BatchDataset.class_dataset_reader(FLAGS.data_dir)
    data_reader.read_images()

    test_images, test_labels = data_reader.get_test_records()

    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(np.squeeze(test_images[i], axis=2), cmap=plt.cm.gray)
    #     plt.xlabel(test_images[i])
    # plt.show()
    from PIL import Image
    im = Image.fromarray(test_images[0])
    im.show()
    im.save("test.png", "PNG")
    print(test_labels[0])
    t = test_images[0]
    rez = ""
    for i in range(220):
        for j in range(120):
            if(i == 0 and j==0):
                rez += str(t[i][j])
            else:
                rez += "," + str(t[i][j])
    open("test.txt", "w").write(rez)

    # np.savetxt('data.csv', test_images[0], delimiter=',')
    # model = keras.models.load_model("linear.h5")
    # prediction = model.predict(test_images[0:24])
    # a = 0



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
