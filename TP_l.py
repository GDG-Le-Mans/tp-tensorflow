from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

LOAD_SAVED_MODEL = True

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

if LOAD_SAVED_MODEL :
    model.load_weights(checkpoint_path)
else :
    model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
	100*np.max(predictions_array),
	class_names[true_label]),
	color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks(range(10), [str.lower(cn[:4])+'.' for cn in class_names], rotation=45)
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

i = 2
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()








