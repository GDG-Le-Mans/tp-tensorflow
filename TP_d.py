from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist

class_names = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau','Sandales', 'Chemise', 'Baskets', 'Sac', 'Bottines']

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()







