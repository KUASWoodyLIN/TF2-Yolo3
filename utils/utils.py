import tensorflow as tf
import numpy as np


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers_list = []
    conv_name = 'conv2d'
    norm_name = 'batch_normalization'
    for i in range(52):
        if i > 0:
            layers_list.append(model.layers[1].get_layer(conv_name + '_{}'.format(i)))
            layers_list.append(model.layers[1].get_layer(norm_name + '_{}'.format(i)))
        else:
            layers_list.append(model.layers[1].get_layer(conv_name))
            layers_list.append(model.layers[1].get_layer(norm_name))
    for i in range(52, 58):
        layers_list.append(model.get_layer(conv_name + '_{}'.format(i)))
        layers_list.append(model.get_layer(norm_name + '_{}'.format(i)))
    layers_list.append(model.get_layer('conv2d_last_layer1_80'))
    for i in range(58, 65):
        layers_list.append(model.get_layer(conv_name + '_{}'.format(i)))
        layers_list.append(model.get_layer(norm_name + '_{}'.format(i)))
    layers_list.append(model.get_layer('conv2d_last_layer2_80'))
    for i in range(65, 72):
        layers_list.append(model.get_layer(conv_name + '_{}'.format(i)))
        layers_list.append(model.get_layer(norm_name + '_{}'.format(i)))
    layers_list.append(model.get_layer('conv2d_last_layer3_80'))

    for i, layer in enumerate(layers_list):
        if not layer.name.startswith('conv2d'):
            continue
        batch_norm = None
        if i + 1 < len(layers_list) and \
                layers_list[i + 1].name.startswith('batch_normalization'):
            batch_norm = layers_list[i + 1]

        print("{}/{} {}".format(model.name, layer.name, 'bn' if batch_norm else 'bias'))

        filters = layer.filters
        size = layer.kernel_size[0]
        in_dim = layer.input.shape[-1]

        if batch_norm is None:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
        else:
            # darknet [beta, gamma, mean, variance]
            bn_weights = np.fromfile(
                wf, dtype=np.float32, count=4 * filters)
            # tf [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, size, size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(
            conv_shape).transpose([2, 3, 1, 0])

        if batch_norm is None:
            layer.set_weights([conv_weights, conv_bias])
        elif batch_norm:
            layer.set_weights([conv_weights])
            batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def trainable_model(model, trainable=False):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            trainable_model(l, trainable)
        else:
            l.trainable = trainable
