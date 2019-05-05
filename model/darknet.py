import tensorflow as tf
from tensorflow.keras import layers


def darknetconv2d(x, filters, kernel_size, strides=(1, 1)):
    padding = 'valid' if strides == (2, 2) else 'same'
    x = layers.Conv2D(filters, kernel_size, strides,
                      padding=padding,
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    return x


def darknetconv2d_bn_leaky(x, filters, kernel_size, strides=(1, 1)):
    padding = 'valid' if strides == (2, 2) else 'same'
    x = layers.Conv2D(filters, kernel_size, strides,
                      padding=padding,
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def resblock_body(x, num_filters, num_blocks):
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknetconv2d_bn_leaky(x, num_filters, (3, 3), strides=(2, 2))
    for i in range(num_blocks):
        y = darknetconv2d_bn_leaky(x, num_filters//2, (1, 1))
        y = darknetconv2d_bn_leaky(y, num_filters, (3, 3))
        x = layers.Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = darknetconv2d_bn_leaky(x, 32, (3, 3))
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


