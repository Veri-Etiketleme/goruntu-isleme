import numpy as np

# Implementation based on https://sudeepraja.github.io/Neural/

def loss_function(y_hat, y):
    # loss = 0
    # for i in range(len(y[0])):
    #     loss += np.sum((y_hat[:, i] - y[:, i]) ** 2)
    return np.sum((y_hat - y) ** 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(X):
    return np.maximum(0, X)


def linear(z):
    return z


def activation_derivative(activation_layer):
    def sigmoid_derivative(z):
        return sigmoid(z) * (1 - sigmoid(z))

    def relu_derivative(z):
        return (z > 0) * 1

    def linear_derivative(z):
        return np.ones_like(z)

    if activation_layer.__name__ == "sigmoid":
        return sigmoid_derivative
    elif activation_layer.__name__ == "relu":
        return relu_derivative
    else:
        return linear_derivative


class NN:
    def __init__(self, input_layer_size: int, hidden_layers: list, output_layer_size: int, activation_layer: list,
                 verbose=False):

        assert len(hidden_layers) > 0, "The number of hidden layers should be greater than 0"
        assert all(layer > 0 for layer in hidden_layers), "Hidden layers can't be non-positive"
        assert len(activation_layer) == len(hidden_layers) + 1, "Not all activation layers are given"

        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers
        self.activation_layer = activation_layer
        self.bias_matrix = []
        self.weight_matrix = []
        self.x_array = []
        self.loss_array = []
        self.build_model()

    def build_model(self):
        layers = [self.input_layer_size] + self.hidden_layers + [self.output_layer_size]
        print(f"layers {layers}")
        for i in range(1, len(layers)):
            weight = np.random.normal(0, 1, (layers[i], layers[i - 1]))
            bias = np.random.normal(0, 1, (layers[i], 1))
            self.weight_matrix.append(weight)
            self.bias_matrix.append(bias)

    def forward_pass(self, input_array: np.array):
        # reinitialize
        self.x_array = []

        x = input_array.copy()
        self.x_array.append(x)
        for i in range(len(self.weight_matrix)):
            x = self.activation_layer[i](np.matmul(self.weight_matrix[i], x) + self.bias_matrix[i])
            self.x_array.append(x)
        return x

    def backward_pass(self, y, alpha):
        # reverse_x = self.x_array.copy()
        # reverse_x.reverse()

        delta_mat = [0] * (len(self.weight_matrix))
        # delta_mat.append( np.matmul((reverse_x[0] - y),self.activation

        for i in range(len(self.weight_matrix), 0, -1):

            print(f"Layer {i}")
            if i == len(self.weight_matrix):

                print(f"i : {i}")
                print(f"x - y : {self.x_array[i] - y}")
                # print(y.shape)
                # print(self.weight_matrix[i-1].shape)
                # print(self.x_array[i-1].shape)
                # print(self.bias_matrix[i-1].shape)
                # print(f"activation function : {activation_derivative(self.activation_layer[i - 1]).__name__}")
                print((np.matmul(self.weight_matrix[i - 1], self.x_array[i - 1]) + self.bias_matrix[i - 1]).shape)
                # print(f"activation function input : {type(np.matmul(self.weight_matrix[i - 1],self.x_array[i - 1]) + self.bias_matrix[i - 1])}")

                delta_mat[i - 1] = 2 * np.multiply((self.x_array[i] - y),
                                                   activation_derivative(self.activation_layer[i - 1])(
                                                       np.matmul(self.weight_matrix[i - 1],
                                                                 self.x_array[i - 1]) +
                                                       self.bias_matrix[i - 1]))
            else:
                delta_mat[i - 1] = 2 * np.multiply(
                    np.matmul(np.transpose(self.weight_matrix[i]), delta_mat[i]),
                    activation_derivative(self.activation_layer[i - 1])(
                        np.matmul(self.weight_matrix[i - 1],
                                  self.x_array[i - 1]) +
                        self.bias_matrix[i - 1]))
            # print(f"i:{i}")
            # print(delta_mat[i - 1])
            dedw = np.matmul(delta_mat[i - 1], np.transpose(self.x_array[i - 1]))
            dedb = delta_mat[i - 1]
            # print()
            # print(f"dedb {dedb}")
            # print(f"dedw {dedw}")
            # print()

            self.weight_matrix[i - 1] = self.weight_matrix[i - 1] - alpha * dedw
            self.bias_matrix[i - 1] = self.bias_matrix[i - 1] - alpha * dedb
        print(f"Weight matrix shape {[weight.shape for weight in self.weight_matrix]}")
        print(f"Bias matrix shape {[bias.shape for bias in self.bias_matrix]}")

    def train(self, x_data, y_data, iterations, alpha):
        count = 0
        while count <= iterations:
            for j in range(len(x_data)):
                yout = self.forward_pass(np.expand_dims(x_data[:, j], 1))

                # print(f"yout {yout.shape}")
                # print(f"y_data {np.expand_dims(y_data[:, j], 1)}")

                loss = loss_function(yout, np.expand_dims(y_data[:, j], 1))
                # self.loss_array.append(loss)
                print(f"Loss {loss} Iteration {count}")
                self.backward_pass(np.expand_dims(y_data[:, j], 1), alpha)
                # print(f"Weight {self.weight_matrix[-1]}")
                # print()
                # print(f"Bias {self.bias_matrix[-1]}")
                # input()
            count += 1


