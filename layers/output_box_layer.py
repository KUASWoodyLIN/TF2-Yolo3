import tensorflow as tf


class YoloOutputBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, output_layer=1, num_classes=80, training=False, **kwargs):
        """

        :param anchors: Yolo anchors box setting
        :param output_layer:
        :param num_classes:
        :param training:
        :param kwargs:
        """
        super(YoloOutputBoxLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self.training = training
        if output_layer == 1:
            self.grid_to_img_scale = 32
        elif output_layer == 2:
            self.grid_to_img_scale = 16
        else:
            self.grid_to_img_scale = 8

    def build(self, input_shape):
        self.grid_h, self.grid_w = input_shape[1:3]

    def call(self, inputs, **kwargs):
        """
        :param inputs:  (batch, grid_h, grid_w, anchors, (x, y, w, h, obj, ...classes))
        :param kwargs: None
        :return:
                bbox: (batch, grid_h, grid_w, anchors, (x1, y1, x2, y2))
                box_confidence: (batch, grid_h, grid_w, anchors, 1)
                box_class_probs: (batch, grid_h, grid_w, anchors, classes)
                 pred_box: (batch, grid_h, grid_w, anchors, (tx, ty, tw, th)) for Calculated loss function
        """
        if self.grid_h is None:
            grid_h, grid_w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        else:
            grid_h, grid_w = self.grid_h, self.grid_w

        box_xy, box_wh, box_confidence, box_class_probs = tf.split(inputs, (2, 2, 1, self.num_classes), axis=-1)
        box_xy = tf.sigmoid(box_xy)     # scale to 0~1
        box_confidence = tf.sigmoid(box_confidence)     # scale to 0~1
        box_class_probs = tf.sigmoid(box_class_probs)   # scale to 0~1
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original x,y,w,h for loss function

        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.stack(grid, axis=-1)          # (gx, gy, 2)
        grid = tf.expand_dims(grid, axis=2)     # (gx, gy, 1, 2)

        # box_xy: (batch, grid_h, grid_w, anchors, (x, y))
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast((grid_w, grid_h), tf.float32)
        # Calculate input image size
        img_w, img_h = (grid_w * self.grid_to_img_scale, grid_h * self.grid_to_img_scale)
        # box_wh: (batch, grid_h, grid_w, anchors, (w, h))
        box_wh = self.anchors * tf.exp(box_wh) / (img_w, img_h)

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        if self.training:
            return tf.concat([bbox, box_confidence, box_class_probs, pred_box], axis=-1)
        return bbox, box_confidence, box_class_probs