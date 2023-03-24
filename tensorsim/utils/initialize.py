import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense
from tf.keras.initializers import GlorotNormal

def initialize_weights(*models):
    for model in models:
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                GlorotNormal()(layer.weights[0])
                if layer.use_bias:
                    layer.bias.assign(tf.zeros_like(layer.bias))
            elif isinstance(layer, BatchNormalization):
                layer.gamma.assign(tf.ones_like(layer.gamma))
                layer.beta.assign(tf.zeros_like(layer.beta))
