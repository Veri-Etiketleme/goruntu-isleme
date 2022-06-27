from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import numpy as np


image = load_img('dataset/all_open/5.jpg', target_size=(224, 224))
image = img_to_array(image)

print(image)

image = preprocess_input(image)

print(image)

simpan = tf.keras.preprocessing.image.array_to_img(image)
simpan.save('./cekhasil/hasil.png')

input()



images = np.array([image], dtype="float32")

# print(images)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# i=0
# for a in aug.flow(images, batch_size=32, save_prefix='test', save_to_dir='./preprocess'):
#     i += 1
#     if i >32:
#         break


