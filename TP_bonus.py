from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

LOAD_SAVED_MODEL = False

##########################################################################
######################### GESTION IMAGE IMPORTEE #########################
##########################################################################

# installation PIL : pip install python-resize-image
from PIL import Image

#my_image = "mon-tshirt.jpg"
my_image = sys.argv[1]
file_name = '.'.join(my_image.split('.')[:-1])

# conversion png necessaire pour la suite
img = Image.open(my_image)
img.save(file_name+'.png')
img.close()

# passage en noir et blanc
img = Image.open(file_name+'.png').convert('LA')
# resize format 28x28
img = img.resize((28,28))
# sauvegarde
img.save(file_name+'.resize.png')
# recuperation valeur des pixels (liste de 728 couples de 2 int)
pix_val = list(img.getdata())
img.close()

# recuperation de la premiere valeur des couples (et inversion pour corespondance image MNIST)
pix_val_flat0 = [255-sets[0] for sets in pix_val]
# recuperation de la seconde valeur des couples
pix_val_flat1 = [sets[1] for sets in pix_val]
# verrification : la premiere valeur c'est celle du pixel
assert(min(pix_val_flat0)>=0 and max(pix_val_flat0)<=255)
# verrification : la seconde valeur c'est toujours 255
assert(pix_val_flat1 == 28*28*[255])

# conversion en un numpy array de (1,28,28) comme dans le MNIST
pix_array = np.array(pix_val_flat0)
pix_array.resize((28,28))
pix_array = (np.expand_dims(pix_array,0))

##########################################################################
##########################################################################
########################################################################## 

class_names = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau','Sandales',
'Chemise', 'Baskets', 'Sac', 'Bottines']

pix_array = pix_array / 255.0

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
    fashion_mnist = keras.datasets.fashion_mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])

predictions_single = model.predict(pix_array)

def plot_image(predictions_array, img):
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions_array)
	plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],100*np.max(predictions_array)))

def plot_value_array(predictions_array):
	plt.grid(False)
	plt.xticks(range(10), class_names, rotation=45)
	plt.yticks([])
	plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(predictions_single[0], pix_array[0])
plt.subplot(1,2,2)
plot_value_array(predictions_single[0])
plt.show()









