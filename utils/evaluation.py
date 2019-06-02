import tensorflow as tf


# TODO:
class MeanAveragePrecision(tf.python.keras.metrics.Metric):
    def __init__(self, name='mean_average_precision', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        boxes, scores, classes, nums = y_pred
        self.total.assign_add(values_sum)  # 更新正確預測的總數
        self.count.assign_add(num_values)  # 更新資料量的總數

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        # 每一次Epoch結束後會重新初始化變數
        self.total.assign(0.)
        self.count.assign(0.)
