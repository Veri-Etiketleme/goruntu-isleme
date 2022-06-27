from ann.losses import MSE
from ann.sequential_model import Sequential
from ann.linear import Linear
from ann.activations import Sigmoid
import json
from ann.image_operations import *
import numpy as np


# global variables
window_shape = (3, 3)
row_margin = window_shape[0] // 2
col_margin = window_shape[1] // 2


def train(model):
    x_data = []
    y_data = []
    with open('./dataset/mountain_photos_info.json', 'r') as fo:
        dataset = json.loads(fo.read())
        for i, image_key in enumerate(dataset):
            print(f"Processing Img: {i}")
            try:
                img = get_image_from_url(dataset[image_key]['url'])
            except Exception:
                continue
            # img.show()
            # input()
            generate_data_set_from_image(img, x_data, y_data, window_shape)
            model.train(np.array(x_data), np.array(y_data), loss=MSE(), epoch=2, lr=5e-6, batch_size=100)
            x_data, y_data = [], []
            if (i+1) % 10 == 0:
                validate(model, dataset, image_key)


def validate(model, dataset, image_key):
    # test image
    img = get_image_from_url(dataset[image_key]['url'])
    np_img = np.asarray(img)
    img.show()

    # gray scale
    gray_img = 0.21 * np_img[:, :, 0] + 0.72 * np_img[:, :, 1] + 0.07 * np_img[:, :, 2]
    gray_img = Image.fromarray(gray_img.astype(np.uint8))
    gray_img.show()
    # generate test data set
    x_test, y_test = [], []
    generate_data_set_from_image(img, x_test, y_test, window_shape)

    out = model.forward(np.array(x_test))
    np_out = np.zeros(np_img.shape - np.array([2, 2, 0]))
    k = 0
    for i in range(np_out.shape[0]):
        for j in range(np_out.shape[1]):
            np_out[i, j, ] = out[k] * 255
            k += 1
    out_img = Image.fromarray(np_out.astype(np.uint8))
    out_img.show()


def main():
    model = Sequential()

    model.add(Linear(in_dim=9, out_dim=18))
    model.add(Sigmoid())
    model.add(Linear(in_dim=18, out_dim=36))
    model.add(Sigmoid())
    model.add(Linear(in_dim=36, out_dim=18))
    model.add(Sigmoid())
    model.add(Linear(in_dim=18, out_dim=9))
    model.add(Sigmoid())
    model.add(Linear(in_dim=9, out_dim=3))

    train(model)
    validate(model)


if __name__ == '__main__':
    main()
