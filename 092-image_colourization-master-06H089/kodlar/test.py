
from ann.activations import Sigmoid, ReLU, Softmax,Tanh
from ann.linear import Linear
from ann.losses import MSE, Loss, BCE, CCE
import numpy as np
from ann.sequential_model import Sequential
from tqdm import tqdm
import sys
import logging
import time

model = Sequential()
model.add(Linear(in_dim=2, out_dim=2))
model.add(Tanh())
# model.add(Linear(in_dim=2, out_dim=2))
# model.add(Sigmoid())
model.add(Linear(in_dim=2, out_dim=2))
model.add(Tanh())

data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
x_data = data
y_data = np.array([[[1, 0]], [[0, 1]], [[1, 0]], [[0,  1]]])

print(model)
# print("grad", model.back_prop(np.array([0.5, 0.4, 0.3])))
cce = MSE()
model.train(x_data, y_data, loss=cce, epoch=1000, batch_size=1 , lr=0.1)
print("out", model.forward(x_data))