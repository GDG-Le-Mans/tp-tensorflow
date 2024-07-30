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

print("train_images.shape")
print(train_images.shape)
print("len(train_labels)")
print(len(train_labels))
print("train_labels")
print(train_labels)
print("test_images.shape")
print(test_images.shape)
print("len(test_labels)")
print(len(test_labels))







