import tensorflow as tf


class DarkNetConv2DBNLeaky(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1), **kwargs):
        super(DarkNetConv2DBNLeaky, self).__init__(**kwargs)
        padding = 'valid' if strides == (2, 2) else 'same'
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides,
                                             padding=padding,
                                             use_bias=False,
                                             kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_normalization(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x
