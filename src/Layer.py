import tensorflow as tf
from math import sqrt


############# Usefull functions #################
def weight_variable(shape):
    nOut = shape[-1]  # nOut
    std = 1 / sqrt(nOut)

    initial = tf.random_uniform(shape, minval=-std, maxval=std, dtype=tf.float32, seed=None)
    return tf.Variable(initial, trainable=True, name="weights")


def bias_variable(shape):
    return tf.Variable(tf.zeros(shape),trainable=True, name="bias")


class Layer:
    def __init__(self, inputSize, ouputSize, name):
        with tf.name_scope('name'):
            self.w = weight_variable([inputSize, ouputSize])  # warning regarding line/column
            self.b = bias_variable([ouputSize])


def buildOutput(layers, placeholder):
    y = placeholder
    for layer in layers[:-1]:
        y = tf.matmul(y, layer.w) + layer.b
        y = tf.nn.tanh(y)

    y = tf.matmul(y, layers[-1].w) + layers[-1].b
    return y
