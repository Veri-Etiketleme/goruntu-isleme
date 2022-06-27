"""
Dense Sequential Neural Network Model
"""
from ann.activations import Sigmoid, ReLU
from ann.linear import Linear
from ann.losses import MSE, Loss, BCE
import numpy as np
from tqdm import tqdm
import sys
import logging
import time

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
log = logging.getLogger(__name__)


class Sequential:
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

    def add(self, layer):        
        self.layers.append(layer)

    def forward(self, _input):
        for layer in self.layers:
            _input = layer.forward(_input)
        return _input

    def back_prop(self, dl_dy, lr):
        for layer in self.layers[::-1]:
#             print(layer)
            dl_dy = layer.back_prop(dl_dy, lr)
#             print(f"dl : {dl_dy.shape}")
        return dl_dy

    def update(self, lr):
        """
        :param lr: learning rate
        """
        for layer in self.layers[::-1]:
            try:
                getattr(layer, 'W')
                layer.update(lr)
            except AttributeError:
                continue

    def train(self, x_train, y_train, loss: Loss, epoch=100, batch_size=64, lr=1e-4):
        """
        Train the neural network model using Batch Gradient Descent
        :param epoch:
        :param lr: learning rate
        :param x_train: (n, input_features)
        :param y_train: (last_layer_size, 1)
        :param loss: loss function used to train the model
        :param batch_size: No. of data points used in for batch gradient descent
        :return: None
        """
        if batch_size == 0:  # use entire dataset
            batch_size = x_train.shape[0]
        iterations = x_train.shape[0] // batch_size
        with tqdm(total=epoch * iterations) as pbar:
            # log.info(f"Training iterations: {iterations}")
            for ep in range(epoch):
                for i in range(iterations):
#                     time.sleep(0.1)
                    batch = x_train[i * batch_size: (i + 1) * batch_size]                
                    y = y_train[i * batch_size: (i + 1) * batch_size]
                    y_hat = self.forward(batch)
#                     print(f"batch: {batch}, y_hat: {y_hat}, y: {y} ")
                    # dL/dy vector at output layers
                    dl_dy = loss.grad(y_hat, y)
                    # back propagate the error
                    self.back_prop(dl_dy, lr)
        
                    pbar.update(1)
                    pbar.set_description(
                        f"{ep},{i} => Avg loss: {np.mean(loss.loss(y_hat, y))}")

    def __str__(self):
        return f"{__class__}: \n {[str(layer) for layer in self.layers]}"


if __name__ == '__main__':
    model = Sequential()
    model.add(Linear(in_dim=2, out_dim=2))
    model.add(Sigmoid())
    # model.add(Linear(in_dim=2, out_dim=2))
    # model.add(Sigmoid())
    model.add(Linear(in_dim=2, out_dim=1))
    model.add(Sigmoid())

    data = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    x_data = data
    y_data = np.array([1, -1, 1, -1]).reshape(4, 1)

    print(model)
    # print("grad", model.back_prop(np.array([0.5, 0.4, 0.3])))
    bce = BCE()
    model.train(x_data, y_data, loss=bce, epoch=2, batch_size=4)
    print("out", model.forward(x_data))
