from PIL import Image
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
                    rgb = arr[x, y, ]
                    # middle element of filter, to be used as output
                    if x == i and y == j:
                        # scale down by 255 to get all values in 0 - 1 range
                        y_data.append(rgb/255)
                    data_pt.append(rgbToGray(rgb)/255)
            x_data.append(np.array(data_pt))
    return x_data, y_data