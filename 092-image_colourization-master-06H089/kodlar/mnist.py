import keras
from ann.linear import Linear
from ann.activations import Sigmoid, Softmax, ReLU

from keras.datasets import mnist

from ann.losses import BCE
from ann.sequential_model import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Linear(in_dim=image_vector_size, out_dim=32))
model.add(Sigmoid())
model.add(Linear(in_dim=32, out_dim=10))
model.add(Softmax())

if __name__ == '__main__':
    bce = BCE()
    model.train(x_train, y_train, loss=bce, batch_size=128, epoch=5, lr=1e-10)