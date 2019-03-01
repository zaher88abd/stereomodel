import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, \
    Input, Reshape, multiply, add, Dropout, concatenate, Conv2DTranspose
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from itertools import product
from skimage.morphology import label
import numpy as np


def down_sampling(input, n_filters, kernel=5):
    l = Conv2D(filters=n_filters, kernel_size=kernel, strides=2, padding='same')(input)
    return l


def up_sample(input, n_filters, kernel_size=2, stride=2):
    l = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=stride, padding='same')(input)
    return l


def bottleneck(input, n_filters, kernel=1):
    l = Conv2D(filters=n_filters, kernel_size=kernel, strides=1, padding='valid')(input)
    return l


def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.3):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(filters=n_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l


def dens_block(input):
    l = bottleneck(input, n_filters=8)
    l1 = BN_ReLU_Conv(l, n_filters=4)
    c1 = concatenate([input, l1])
    l = bottleneck(c1, n_filters=8)
    l2 = BN_ReLU_Conv(l, n_filters=4)
    c2 = concatenate([c1, l2])
    l = bottleneck(c2, n_filters=8)
    l3 = BN_ReLU_Conv(l, n_filters=4)
    c3 = concatenate([l1, l2, l3])
    return c3
