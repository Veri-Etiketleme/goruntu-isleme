import os
import numpy as np
from PIL import Image
from neural_net import NN, sigmoid, relu, linear
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
from PIL import Image
import json
import requests
from io import BytesIO
import numpy as np


def get_image_from_url(url):
    """
    returns PIL Image object from response
    :param url:
    :return:
    """
    response = requests.get(url, stream=True)
    img_loc = Image.open(BytesIO(response.content))
    return img_loc


def rgbToGray(rgb):
    """
    :param rgb: 1D np array consisting of RGB values of the pixel
    :return: gray = 0.21r + 0.72g  + 0.07b
    """
    return np.dot(np.array([0.21, 0.72, 0.07]), rgb)


def generate_data_set_from_image(image: Image, x_data, y_data, window_shape = (3, 3)):
    """
    Window dimensions are always assumed to be odd, i.e. we always have a middle element
    """
    arr = np.asarray(image)
    n, m = arr.shape[0], arr.shape[1]
    print("size", n, m)

    # base case: image must be larger than the window size
    if n < window_shape[0] or m < window_shape[1]:
        return None

    row_margin = window_shape[0] // 2
    col_margin = window_shape[1] // 2
    for i in range(0 + row_margin, n - row_margin):
        for j in range(0 + col_margin, m - col_margin):
            data_pt = []
            for x in range(i - row_margin, i + row_margin + 1):
                for y in range(j - col_margin, j + col_margin + 1):
                    rgb = arr[x, y, ] / 255
                    # middle element of filter, to be used as output
                    if x == i and y == j:
                        y_data.append(rgb)
                    data_pt.append(rgbToGray(rgb) / 255)
            x_data.append(np.array(data_pt))
    return x_data, y_data


def load_data(data_folder: str):

    x_data = []
    y_data = []
    for root, dir, files in os.walk(data_folder):
        for file in files:
            print(file)
            img = Image.open(f"{root}/{file}")
            img = img.resize((200,200))
            y_data.append(np.asarray(img)[:, :, 0].reshape((-1)))
            img = img.convert('L')
            x_data.append(np.asarray(img).reshape((-1)))
            img.close()
    return np.array(x_data), np.array(y_data)


# if __name__ == '__main__':
#     # X, y = make_circles(n_samples=1000, factor=.3, noise=.10)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     #
#     # plt.figure(figsize=(10, 10))
#     # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), s=50, cmap=plt.cm.Spectral, edgecolors='black')
#     # plt.show()
#
#     x_data,y_data = load_data("../../colorization/data1/")
#     x_train,y_train = (1.0/255) * x_data[:100], (1.0/255) * y_data[:100]
#
#     new_x = x_train.transpose()
#     new_y = y_train.transpose()


    # model = NN(40000, [200, 100, 50], 40000, [relu, relu, relu, linear])


if __name__ == '__main__':
    #  TODO - figure out why is this getting loaded from inside the ann folder
    # x_data = []
    # y_data = []
    # with open('../dataset/ocean_photos_info.json', 'r') as fo:
    #     print(fo)
    #     oceans = json.loads(fo.read())
    #     for i, image_key in enumerate(oceans):
    #         print(f"Processing Img: {image_key}")
    #         img = get_image_from_url(oceans[image_key]['url'])
    #         generate_data_set_from_image(img, x_data, y_data, (5,5))
    #         # img.show()
    #         # input()
    #
    #         if i == 10:
    #             print("Stopping after ", i, "images")
    #             break
    #
    # x_data = np.array(x_data).transpose()
    # y_data = np.array(y_data).transpose()
    # np.save("x_data.npy", x_data)
    # np.save("y_data.npy", y_data)


    #Download data above and load below
    # x_data = np.load("x_data.npy")
    # y_data = np.load("y_data.npy")

    model = NN(9, [30, 15], 3, [sigmoid, sigmoid, linear])

    # model = Sequential()
    #
    # model.add(Dense(12, input_dim=9, activation='sigmoid'))
    # model.add(Dense(8, activation='sigmoid'))
    # model.add(Dense(3))
    #
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])