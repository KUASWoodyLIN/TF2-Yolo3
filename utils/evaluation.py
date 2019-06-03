import tensorflow as tf


# TODO:
class MeanAveragePrecision(tf.python.keras.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, score_threshold=0.5, name='mean_average_precision', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        bboxes, box_conf, box_class = [], [], []
        for pred in y_pred:
            pred_box, pred_obj, pred_class, _ = tf.split(pred, (4, 1, self.num_classes, 4), axis=-1)
            bboxes.append(tf.reshape(pred_box, (tf.shape(pred_box)[0], -1, 4)))
            box_conf.append(tf.reshape(pred_obj, (tf.shape(pred_obj)[0], -1, 1)))
            box_class.append(tf.reshape(pred_class, (tf.shape(pred_class)[0], -1, self.num_classes)))

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


        self.total.assign_add(values_sum)  # 更新正確預測的總數
        self.count.assign_add(num_values)  # 更新資料量的總數

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        # 每一次Epoch結束後會重新初始化變數
        self.total.assign(0.)
        self.count.assign(0.)
