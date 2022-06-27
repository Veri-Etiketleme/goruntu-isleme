from tensorflow.keras.models import load_model
import tensorflow as tf
import pathlib
import argparse
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from imutils import paths
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
                help="path to input dataset")

args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")



model = load_model('./mask_detector.model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_models_dir = pathlib.Path("./ls_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# tflite_model_file = tflite_models_dir/"fm_model.tflite"
# tflite_model_file.write_bytes(tflite_model)


images = tf.cast(data, tf.float32) / 255.0
facedetect_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)


def representative_data_gen():
    for input_value in facedetect_ds.take(100):
        yield [input_value]


converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir/"fm_model_quant_int8.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
