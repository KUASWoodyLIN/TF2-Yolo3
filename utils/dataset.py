import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# @tf.function
def parse_aug_fn(dataset, input_size=(416, 416)):
    """
    Image Augmentation function
    """
    ih, iw = input_size
    # (None, None, 3)
    x = tf.cast(dataset['image'], tf.float32) / 255.
    # (y1, x1, y2, x2, class)
    bbox = dataset['objects']['bbox']
    label = tf.expand_dims(tf.cast(dataset['objects']['label'], tf.float32), axis=-1)
    x, bbox = resize(x, bbox, input_size)

    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 觸發影像翻轉機率50%
    x, bbox = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: flip(x, bbox), lambda: (x, bbox))
    # 觸發影像縮放機率50%
    x, bbox = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x, bbox), lambda: (x, bbox))
    # 將[y1, x1, y2, x2]合為shape=(x, 4)的Tensor
    bbox = tf.stack(bbox, axis=1)
    bbox = tf.divide(bbox, [ih, iw, ih, iw])

    y = tf.concat([bbox, label], axis=-1)
    paddings = [[0, 100 - tf.shape(bbox)[0]], [0, 0]]
    y = tf.pad(y, paddings)
    y = tf.ensure_shape(y, (100, 5))
    return x, y


def parse_fn(dataset, anchors, anchor_masks, input_size=(416, 416)):
    ih, iw = input_size
    # (None, None, None, 3)
    x = tf.cast(dataset['image'], tf.float32) / 255.
    # (None, [y1, x1, y2, x2])
    bbox = dataset['objects']['bbox']
    # (None, classes, 1)
    label = tf.expand_dims(tf.cast(dataset['objects']['label'], tf.float32), axis=-1)
    # 將影像縮放到模型輸入大小
    x, bbox = resize(x, bbox, input_size)

    y = tf.concat([bbox, label], axis=-1)
    paddings = [[0, 100 - tf.shape(bbox)[0]], [0, 0]]
    y = tf.pad(y, paddings)
    y = tf.ensure_shape(y, (100, 5))
    x, y = transform_targets(x, y, anchors, anchor_masks)
    return x, y


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    batch = tf.shape(y_true)[0]
    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((batch, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(batch):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(x_train, y_train, anchors, anchor_masks):
    """
            transform y_label to training target label,
        (None, 100, [y1, x1, y2, x2, class]) -> (N, grid_size, grid_size, anchor, [x, y, w, h, obj, class])
        :param x_train: (None, 416, 416, 3)
        :param y_train: (None, 100, [y1, x1, y2, x2, class])
        :param anchors: yolo anchors setting
        :param anchor_masks: yolo anchors mask
        :return:
    """
    y_outs = []
    grid_size = 13

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return x_train, tuple(y_outs)


def resize(x, bboxes, input_size):
    """
        Resize Image(影像縮放)

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs, shape(x, 4) "y1, x1, y2, x2", 0~1
        :param input_size: 網路輸入大小
        :return: 返回(images, bboxes), images: scale0~1, bboxes: return list [y1, x1, y2, x2], scale 0~w, h
        """
    ih, iw = input_size
    # image resize
    x = tf.image.resize(x, (ih, iw))

    # bounding box resize
    bboxes = tf.multiply(bboxes, [ih, iw, ih, iw])
    y1 = bboxes[..., 0]
    x1 = bboxes[..., 1]
    y2 = bboxes[..., 2]
    x2 = bboxes[..., 3]
    return x, [y1, x1, y2, x2]


def flip(x, bboxes):
    """
        flip image(翻轉影像)

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs list [y1, x1, y2, x2], 0~w or h
        :return: 返回(images, bboxes), images: scale0~1, bboxes: return list [y1, x1, y2, x2], scale 0~w, h
        """
    h, w, c = x.shape
    x = tf.image.flip_left_right(x)  # 隨機左右翻轉影像
    y1 = bboxes[0]
    x1 = w - bboxes[3]
    y2 = bboxes[2]
    x2 = w - bboxes[1]
    return x, [y1, x1, y2, x2]


def zoom(x, bboxes, scale_min=0.6, scale_max=1.4):
    """
        Zoom Image(影像縮放)

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs list [y1, x1, y2, x2], 0~w or h
        :param scale_min: 縮放最小倍數
        :param scale_max: 縮放最大倍數
        :return: 返回(images, bboxes), images: scale0~1, bboxes: return list [y1, x1, y2, x2], scale 0~w, h
        """
    h, w, _ = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)  # 隨機縮放比例
    # 等比例縮放
    nh = tf.cast(h * scale, tf.int32)  # 縮放後影像長度
    nw = tf.cast(w * scale, tf.int32)  # 縮放後影像寬度

    # 如果將影像縮小執行以下程式
    def scale_less_then_one():
        resize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (h - nh), tf.int32)
        dx = tf.random.uniform([], 0, (w - nw), tf.int32)
        indexes = tf.meshgrid(tf.range(dy, dy+nh), tf.range(dx, dx+nw), indexing='ij')
        indexes = tf.stack(indexes, axis=-1)
        output = tf.scatter_nd(indexes, resize_x, (h, w, 3))
        return output, dx, dy

    # 如果將影像放大執行以下以下程式
    def scale_greater_then_one():
        resize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (nh - h), tf.int32)
        dx = tf.random.uniform([], 0, (nw - w), tf.int32)
        return resize_x[dy:dy + h, dx:dx + w], -dx, -dy

    def scale_equal_zero():
        return x, 0, 0

    output, dx, dy = tf.case([(tf.less(scale, 1), scale_less_then_one),
                              (tf.greater(scale, 1), scale_greater_then_one)],
                             default=scale_equal_zero)
    # [(tf.less(x, y), f1)]

    # 重新調整bounding box位置
    y1 = bboxes[0] * scale + tf.cast(dy, dtype=tf.float32)
    x1 = bboxes[1] * scale + tf.cast(dx, dtype=tf.float32)
    y2 = bboxes[2] * scale + tf.cast(dy, dtype=tf.float32)
    x2 = bboxes[3] * scale + tf.cast(dx, dtype=tf.float32)
    # 如果座標超出範圍將其限制在邊界上
    y1 = tf.where(y1 < 0, tf.zeros_like(y1), y1)
    x1 = tf.where(x1 < 0, tf.zeros_like(x1), x1)
    y2 = tf.where(y2 > h, h * tf.ones_like(y2), y2)
    x2 = tf.where(x2 > w, w * tf.ones_like(x2), x2)
    # 找出不存在影像上的bounding box並剔除
    box_w = x2 - x1
    box_h = y2 - y1
    bboxes_filter = tf.logical_and(box_w > 1, box_h > 1)
    y1 = y1[bboxes_filter]
    x1 = x1[bboxes_filter]
    y2 = y2[bboxes_filter]
    x2 = x2[bboxes_filter]
    return output, [y1, x1, y2, x2]


def color(x):
    """
         Color change(改變顏色)

        :param x:  image inputs, 0~1
        :return: 返回images
        """
    x = tf.image.random_hue(x, 0.08)  # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0.6, 1.6)  # 隨機調整影像飽和度
    x = tf.image.random_brightness(x, 0.05)  # 隨機調整影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3)  # 隨機調整影像對比度
    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式

    def test_augmentation():
        # 取得訓練數據，並順便讀取data的資訊
        train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
        train_data = train_data.shuffle(1000)  # 打散資料集
        train_data = train_data.map(lambda dataset: parse_aug_fn(dataset),
                                    num_parallel_calls=AUTOTUNE)
        _h, _w = (416, 416)
        images = np.zeros((_h * 4, _w * 4, 3))
        for i in range(4):
            for j, [img, bboxes] in enumerate(train_data.take(4)):
                img = img.numpy()
                for box in bboxes:
                    _y1 = tf.cast(box[0], tf.int16).numpy()
                    _x1 = tf.cast(box[1], tf.int16).numpy()
                    _y2 = tf.cast(box[2], tf.int16).numpy()
                    _x2 = tf.cast(box[3], tf.int16).numpy()
                    cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)
                images[_h * i:_h * (i + 1), _w * j:_w * (j + 1)] = img
        plt.figure(figsize=(12, 12))
        plt.imshow(images)
        plt.show()

    def test_label_transform():
        yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                 (59, 119), (116, 90), (156, 198), (373, 326)],
                                np.float32) / 416
        yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        # 取得訓練數據，並順便讀取data的資訊
        train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
        train_data = train_data.shuffle(1000)  # 打散資料集
        train_data = train_data.map(lambda dataset: parse_aug_fn(dataset),
                                    num_parallel_calls=AUTOTUNE)
        train_data = train_data.batch(1)
        train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks),
                                    num_parallel_calls=AUTOTUNE)
        train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    # Augmentation test
    test_augmentation()
    # Targets transform test
    test_label_transform()
