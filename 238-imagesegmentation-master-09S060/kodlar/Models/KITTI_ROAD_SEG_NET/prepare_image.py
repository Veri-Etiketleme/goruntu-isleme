import scipy.io as sio
import numpy as np
import gzip
import shutil
import argparse
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image

parser = argparse.ArgumentParser(description='convert mnist to pytorch tensor')
parser.add_argument('--src_gz_train_data', dest='src_gz_train_data', default='./data/Train', help='folder containing train data')
parser.add_argument('--src_gz_test_data', dest='src_gz_test_data', default='./data/Test', help='folder containing test data')
parser.add_argument('--train_input', dest='train_input', default='train-images-idx3-ubyte.gz', help='train input data file name')
parser.add_argument('--test_input', dest='test_input', default='t10k-images-idx3-ubyte.gz', help='test input data file name')
parser.add_argument('--train_target_output', dest='train_target_output', default='train-labels-idx1-ubyte.gz', help='train target data file name')
parser.add_argument('--test_target_output', dest='test_target_output', default='t10k-labels-idx1-ubyte.gz', help='test target data file name')
parser.add_argument('--image_size', dest='image_size', type=np.uint8, default=28, help='size of the image')
parser.add_argument('--num_images', dest='num_images', type=np.uint8, default=60000, help='number of the images')
parser.add_argument('--num_test_images', dest='num_test_images', type=np.uint8, default=10000, help='number of test images')
parser.add_argument('--num_classes', dest='num_classes', type=np.uint8, default=10, help='number of the classes')
args = parser.parse_args()

## unzip the files
def unzip_inputs():
    with gzip.open(os.path.join(args.src_gz_train_data, args.train_input)) as bytestream:
        ## Read file content
        bytestream.read(16)
        buf = bytestream.read(args.image_size * args.image_size * args.num_images)
        data_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        ## Normalize the data to be between 0-1
        data_train = data_train / 255.0
        data_train = data_train.reshape(args.num_images, 1, args.image_size, args.image_size)
        bytestream.close()
    with gzip.open(os.path.join(args.src_gz_test_data, args.test_input)) as bytestream:
        ## Read file content
        bytestream.read(16)
        buf = bytestream.read(args.image_size * args.image_size * args.num_test_images)
        data_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        ## Normalize the data to be between 0-1
        data_test = data_test / 255.0
        data_test = data_test.reshape(args.num_test_images, 1, args.image_size, args.image_size)
        bytestream.close()
    return data_train, data_test

def unzip_targets():
    with gzip.open(os.path.join(args.src_gz_train_data, args.train_target_output)) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * args.num_images)
        data_train = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        bytestream.close()
    with gzip.open(os.path.join(args.src_gz_test_data, args.test_target_output)) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * args.num_images)
        data_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        bytestream.close()
    return data_train, data_test

def plot_images(images):
    f, axis = plt.subplots(5, 5)
    num_img_plot = np.linspace(0, 24, num=25)
    num_img_plot = num_img_plot.astype(np.int)
    for i, ax in zip(num_img_plot, np.ravel(axis)):
        ax.imshow(images[i, :, :, 0], cmap=plt.cm.gray)
    plt.show()

def np_to_torch(np_data):
    torch_data = torch.from_numpy(np_data)
    return torch_data

def one_hot_label(labels):
    ## create an zero mat for one hot encoding
    one_hot = np.zeros((args.num_images, args.num_classes))
    for i in range(len(labels)):
        one_hot[i, labels[i] - 1] = 1
    return one_hot

def resize_img(image, newSize):
    img_resized = image.resize(newSize, resample=Image.ANTIALIAS)
    return img_resized

def gz_to_tensor():
    images_train, images_test = unzip_inputs()
    labels_train, labels_test = unzip_targets()
    #print(labels[0:25])
    #plot_images(images)
    torch_train_imgs = np_to_torch(images_train)
    torch_test_imgs = np_to_torch(images_test)
    print(torch_train_imgs.size())
    print(torch_test_imgs.size())
    torch_train_labels = np_to_torch(one_hot_label(labels_train).astype(np.int64))
    torch_test_labels = np_to_torch(one_hot_label(labels_test).astype(np.int64))
    print(torch_train_labels.size())
    print(torch_test_labels.size())
    #print(torch_labels[0:25, :])

    return torch_train_imgs, torch_train_labels, torch_test_imgs, torch_test_labels
'''
if __name__ == "__main__":
    images = unzip_inputs()
    labels = unzip_targets()
    print(labels[0:25])
    plot_images(images)
    torch_imgs = np_to_torch(images)
    print(torch_imgs.size())
    torch_labels = np_to_torch(one_hot_label(labels))
    print(torch_labels.size())
    print(torch_labels[0:25, :])
'''
