from __future__ import division, print_function, absolute_import
from typing import Tuple
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd


def autoencoder(learning_rate: float,
                training_epochs: int,
                display_step: int,
                alpha: int,
                n_hidden_nodes: int,
                n_input_nodes: int,
                data_path=None,
                concat_features=None) -> Tuple:
    """Training autoencoder either from features taken from .csv file or concatenated features. Should specify either
    data_path or concat_features object not both
    Arguments:
        learning_rate: autoencoder learning rate
        training_epochs: autoencoders number of training epochs
        alpha: regularization parameter for weight decay
        n_hidden_nodes: number of hidden neurons in inner layer
        n_input_nodes: number of input features passed to autoencoder
        data_path: path to data file
        concat_features: numpy array with concatenated features
    """
    if data_path is not None:
        df = pd.read_csv(data_path, sep=',', header=None)
        FT = df.values
        FT_norm = preprocessing.normalize(FT, axis=1)
    else:
        FT = concat_features
        FT_norm = preprocessing.normalize(FT, axis=1)
    # Setting input node od autoencoder
    X = tf.placeholder("float", [None, n_input_nodes])
    # Setting up weight and biases dictionary
    weights = {
        'encoder_w': tf.Variable(tf.random_normal([n_input_nodes, n_hidden_nodes])),
        'decoder_w': tf.Variable(tf.random_normal([n_hidden_nodes, n_input_nodes]))
    }
    biases = {
        'encoder_b': tf.Variable(tf.random_normal([n_hidden_nodes])),
        'decoder_b': tf.Variable(tf.random_normal([n_input_nodes])),
    }

    # Setting up encoding phase
    def encoder(x):
        # Get hidden layer activation values
        hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w']),
                                            biases['encoder_b']))
        return hidden_layer

    # Setting decoding phase
    def decoder(x):
        # Decoding input layer with sigmoid activation 
        output_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w']),
                                            biases['decoder_b']))
        return output_layer

    # Assembling autoencoder
    encoding = encoder(X)
    decoding = decoder(encoding)
    # Estimating input data
    X_est = decoding
    # True data to compare with
    X_true = X
    # Cost and optimization
    summ_square_errors = tf.reduce_mean(tf.pow(X_true - X_est, 2))
    # Weight in encoder part of autoencoder
    weight_decay_1 = tf.nn.l2_loss(weights['encoder_w'])
    # Weight in decoder part of autoencoder
    weight_decay_2 = tf.nn.l2_loss(weights['decoder_w'])
    cost = summ_square_errors + alpha * (weight_decay_1 + weight_decay_2)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Start the network
    sess = tf.InteractiveSession()
    sess.run(init)
    # Training
    for epoch in range(training_epochs):
        # Running backpropagation optimization
        _, c = sess.run([optimizer, cost], feed_dict={X: FT_norm})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Network optimized")
    # Getting output and hidden layer values
    AE_output = sess.run(decoding, feed_dict={X: FT_norm})
    AE_hidden_representation = sess.run(encoding, feed_dict={X: FT_norm})
    return AE_output, AE_hidden_representation


# Features concatination
def features_comb(lbpf_hidden_representation: np.array,
                  gf_hidden_representation: np.array) -> np.array:
    """
    Concatenating geometric and lbp features in one vector
    :param lbpf_hidden_representation: hidden representation (autoencoder) of lbp features
    :param gf_hidden_representation: hidden representation (autoencoder) of geometric features
    :return: concatenated vector of 2 feature types
    """
    _ = []
    for x, y in zip(lbpf_hidden_representation, gf_hidden_representation):
        cond = np.concatenate([x, y])
        _.append(cond)
        concat_fetures = np.array(_)
    return concat_fetures


class SOM(object):
    """
    Implementation of Self-Organizing Map classifier
    """
    # Check if SOM has been trained or not
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        SOM intialization:
        - m, n: dimensions of SOM 
        - 'n_iterations': number of iterations during training
        - 'dim': dimensionality of the training data 
        - 'alpha': is a number denoting the initial time(iteration no)-based
                   learning rate. Default value is 0.3
        - 'sigma': is the the initial neighbourhood value, denoting
                   the radius of influence of the BMU while training. By default, its
                   taken to be half of max(m, n).
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        # INITIALIZE GRAPH
        self._graph = tf.Graph()
        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
            # VARIABLES AND CONSTANT OPS FOR DATA STORAGE
            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m * n, dim]))
            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
            # PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training
            # The training vector
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.placeholder("float")

            # CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.sub(self._weightage_vects, tf.pack([self._vect_input for i in range(m * n)])), 2), 1)), 0)
            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]))), [2])
            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.sub(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.mul(alpha, learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)
            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.sub(
                self._location_vects, tf.pack(
                    [bmu_loc for i in range(m * n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_multiplier = tf.pack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                for i in range(m * n)])
            weightage_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(tf.pack([self._vect_input for i in range(m * n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)
            # INITIALIZE SESSION
            self._sess = tf.Session()
            # INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
        return to_return