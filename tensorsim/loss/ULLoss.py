import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K

""" Loss function with hard constraints 
"""
class Jacobi_layer(tf.keras.layers.Layer):
    def __init__(self, nx=21, length=0.1, bcs=None):
        super().__init__()
        self.length = length
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = tf.constant([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]], dtype=tf.float32)
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return Conv2D(filters=1, kernel_size=(3, 3), padding='valid', strides=1, use_bias=False, 
                      kernel_initializer=tf.constant_initializer(self.weight), trainable=False)(x)

    def call(self, inputs):
        print('TF Loss Call')
        layout, heat, n_iter = inputs
        # Source item
        f = self.cof * layout
        # The nodes which are not in boundary
        G = tf.ones_like(heat, dtype=tf.float32)

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1].assign(tf.zeros_like(G[..., idx_start:idx_end, :1]))
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:].assign(tf.zeros_like(G[..., idx_start:idx_end, -1:]))
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end].assign(tf.zeros_like(G[..., :1, idx_start:idx_end]))
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end].assign(tf.zeros_like(G[..., -1:, idx_start:idx_end]))
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = tf.pad(heat * G, [1, 1, 1, 1], "REFLECT")
            else:
                x = tf.pad(x, [1, 1, 1, 1], "REFLECT")
            x = G * (self.jacobi(x) + f)
        return x

class OHEMF12d(tf.keras.losses.Loss):
    def __init__(self, loss_fun, weight=None):
        super(OHEMF12d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def call(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction=tf.keras.losses.Reduction.NONE)
        diff = tf.stop_gradient(diff)
        min, max = tf.reduce_min(tf.reshape(diff, [diff.shape[0], -1]), axis=1), tf.reduce_max(tf.reshape(diff, [diff.shape[0], -1]), axis=1)
        if inputs.ndim == 4:
            min, max = tf.reshape(min, [diff.shape[0], 1, 1, 1]), tf.reshape(max, [diff.shape[0], 1, 1, 1])
            min, max = tf.broadcast_to(min, diff.shape), tf.broadcast_to(max, diff.shape)
        elif inputs.ndim == 3:
            min, max = tf.reshape(min, [diff.shape[0], 1, 1]), tf.reshape(max, [diff.shape[0], 1, 1])
            min, max = tf.broadcast_to(min, diff.shape), tf.broadcast_to(max, diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return tf.reduce_mean(tf.abs(diff * (inputs - targets)))