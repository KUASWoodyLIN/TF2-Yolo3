import tensorflow as tf


class NMSLayer(tf.keras.layers.Layer):
    """
    Non maximum suppression Layer
    """
    def __init__(self, num_classes, iou_threshold, score_threshold, **kwargs):
        super(NMSLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def call(self, inputs, **kwargs):
        bboxes, box_conf, box_class = [], [], []
        for pred in inputs:
            bboxes.append(tf.reshape(pred[0], (tf.shape(pred[0])[0], -1, 4)))
            box_conf.append(tf.reshape(pred[1], (tf.shape(pred[1])[0], -1, 1)))
            box_class.append(tf.reshape(pred[2], (tf.shape(pred[2])[0], -1, self.num_classes)))

        bboxes = tf.concat(bboxes, axis=1)
        box_conf = tf.concat(box_conf, axis=1)
        box_class = tf.concat(box_class, axis=1)

        scores = box_conf * box_class
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1, self.num_classes)),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )
        return boxes, scores, classes, valid_detections
