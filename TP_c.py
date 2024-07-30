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

image_to_show = train_images[0]

plt.figure()
plt.imshow(image_to_show, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()







