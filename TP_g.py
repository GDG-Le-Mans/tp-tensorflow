from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist

class_names = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau','Sandales',
'Chemise', 'Baskets', 'Sac', 'Bottines']

train_images = train_images / 255.0
test_images = test_images / 255.0

save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = save_dir + "/cp.ckpt.keras"
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path)

model = keras.Sequential([
keras.Input(shape=(28,28)),
keras.layers.Flatten(),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])








