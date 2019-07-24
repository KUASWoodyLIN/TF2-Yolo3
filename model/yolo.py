import config
from layers import YoloOutputLayer, YoloOutputBoxLayer, NMSLayer
from model.darknet import darknet_body, darknetconv2d_bn_leaky, darknetconv2d
from tensorflow.python.keras import layers, Model, Input


yolo_anchors = config.yolo_anchors
yolo_tiny_anchors = config.yolo_tiny_anchors


def make_last_layers(x, num_filters, num_anchors, num_classes):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    out_filters = num_anchors * (num_classes + 5)
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknetconv2d_bn_leaky(x, num_filters * 2, (3, 3))
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknetconv2d_bn_leaky(x, num_filters * 2, (3, 3))
    x = darknetconv2d_bn_leaky(x, num_filters, (1, 1))
    y = darknetconv2d_bn_leaky(x, num_filters*2, (3, 3))
    y = darknetconv2d(y, out_filters, (1, 1), num_classes=num_classes)
    y = YoloOutputLayer(num_anchors, num_classes)(y)
    return x, y


def yolov3_tiny(input_size, anchors=yolo_tiny_anchors, num_classes=80, iou_threshold=0.5, score_threshold=0.5, training=False):
    num_anchors = len(anchors) // 3
    inputs = Input(input_size)
    x1 = darknetconv2d_bn_leaky(inputs, 16, (3, 3))
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = darknetconv2d_bn_leaky(x1, 32, (3, 3))
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = darknetconv2d_bn_leaky(x1, 64, (3, 3))
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = darknetconv2d_bn_leaky(x1, 128, (3, 3))
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = darknetconv2d_bn_leaky(x1, 256, (3, 3))

    x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x2 = darknetconv2d_bn_leaky(x2, 512, (3, 3))
    x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x2)
    x2 = darknetconv2d_bn_leaky(x2, 1024, (3, 3))
    x2 = darknetconv2d_bn_leaky(x2, 256, (1, 1))
    # make last layer 1
    y1 = darknetconv2d_bn_leaky(x2, 512, (3, 3))
    y1 = darknetconv2d(y1, num_anchors*(num_classes+5), (1, 1))
    y1 = YoloOutputLayer(num_anchors, num_classes)(y1)

    # Up Sampling
    x2 = darknetconv2d_bn_leaky(x2, 128, (1, 1))
    x2 = layers.UpSampling2D(2)(x2)

    # make last layer 2
    y2 = layers.Concatenate()([x2, x1])
    y2 = darknetconv2d_bn_leaky(y2, 256, (3, 3))
    y2 = darknetconv2d(y2, num_anchors*(num_classes+5), (1, 1))
    y2 = YoloOutputLayer(num_anchors, num_classes)(y2)

    h, w, _ = input_size
    y1 = YoloOutputBoxLayer(anchors[3:], 1, num_classes, training)(y1)
    y2 = YoloOutputBoxLayer(anchors[:3], 2, num_classes, training)(y2)
    if training:
        return Model(inputs, (y1, y2), name='Yolo-V3')
    outputs = NMSLayer(num_classes, iou_threshold, score_threshold)([y1, y2])
    return Model(inputs, outputs, name='Yolo-V3')


def yolov3(input_size, anchors=yolo_anchors, num_classes=80, iou_threshold=0.5, score_threshold=0.5, training=False):
    """Create YOLO_V3 model CNN body in Keras."""
    num_anchors = len(anchors) // 3
    inputs = Input(input_size)
    x_26, x_43, x = darknet_body(name='Yolo_DarkNet')(inputs)
    x, y1 = make_last_layers(x, 512, num_anchors, num_classes)

    x = darknetconv2d_bn_leaky(x, 256, (1, 1))
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, x_43])
    x, y2 = make_last_layers(x, 256, num_anchors, num_classes)

    x = darknetconv2d_bn_leaky(x, 128, (1, 1))
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, x_26])
    x, y3 = make_last_layers(x, 128, num_anchors, num_classes)
    h, w, _ = input_size
    y1 = YoloOutputBoxLayer(anchors[6:], 1, num_classes, training)(y1)
    y2 = YoloOutputBoxLayer(anchors[3:6], 2, num_classes, training)(y2)
    y3 = YoloOutputBoxLayer(anchors[0:3], 3, num_classes, training)(y3)
    if training:
        return Model(inputs, (y1, y2, y3), name='Yolo-V3')
    outputs = NMSLayer(num_classes, iou_threshold, score_threshold)([y1, y2, y3])
    return Model(inputs, outputs, name='Yolo-V3')


if __name__ == "__main__":
    from tensorflow.python.keras.callbacks import TensorBoard

    # test yolo_v3
    model = yolov3((416, 416, 3), training=True)
    model.summary(line_length=250)

    # test tiny yolo_v3
    model = yolov3_tiny((384, 384, 3), training=True)
    model.summary(line_length=250)

    # Save model graph on tensorboard
    model_tb = TensorBoard('../logs-2')
    model_tb.set_model(model)

