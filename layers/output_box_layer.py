import tensorflow as tf


class YoloOutputBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, num_classes, **kwargs):
        super(YoloOutputBoxLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes

    def build(self, input_shape):
        self.grid_h, self.grid_w = input_shape[1:3]

    def call(self, inputs, **kwargs):
        # inputs (batch, grid_h, grid_w, anchors, (x, y, w, h, obj, ...classes))
        box_xy, box_wh, box_confidence, box_class_probs = tf.split(inputs, (2, 2, 1, self.num_classes), axis=-1)
        box_xy = tf.sigmoid(box_xy)     # scale to 0~1
        box_confidence = tf.sigmoid(box_confidence)     # scale to 0~1
        box_class_probs = tf.sigmoid(box_class_probs)   # scale to 0~1
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original x,y,w,h for loss function

        grid = tf.meshgrid(tf.range(self.grid_w), tf.range(self.grid_h))
        grid = tf.stack(grid, axis=-1)          # (gx, gy, 2)
        grid = tf.expand_dims(grid, axis=2)     # (gx, gy, 1, 2)

        # box_xy: (batch, grid_h, grid_w, anchors, (x, y))
        # each box (x, y)
        box_xy = (box_xy + tf.cast(grid, box_xy.dtype)) / tf.cast((self.grid_w, self.grid_h), box_xy.dtype)
        box_wh = tf.exp(box_wh) * self.anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        return bbox, box_confidence, box_class_probs, pred_box
