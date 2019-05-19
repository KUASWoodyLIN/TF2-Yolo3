import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# FeaturesDict({
#     'image': Image(shape=(None, None, 3), dtype=tf.uint8),
#     'image/filename': Text(shape=(), dtype=tf.string, encoder=None),
#     'objects': SequenceDict({
#         'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
#         'is_crowd': Tensor(shape=(), dtype=tf.bool),
#         'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=80),
#     }),
# })

# FeaturesDict({
#     'image': Image(shape=(None, None, 3), dtype=tf.uint8),
#     'image/filename': Text(shape=(), dtype=tf.string, encoder=None),
#     'labels': Sequence(shape=(None,), dtype=tf.int64, feature=ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
#     'labels_no_difficult': Sequence(shape=(None,), dtype=tf.int64, feature=ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
#     'objects': SequenceDict({
#         'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
#         'is_difficult': Tensor(shape=(), dtype=tf.bool),
#         'is_truncated': Tensor(shape=(), dtype=tf.bool),
#         'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=20),
#         'pose': ClassLabel(shape=(), dtype=tf.int64, num_classes=5),
#     }),
# })
def parse_aug_fn(dataset):
    """
    Image Augmentation function
    """
    # (None, None, 3)
    x = tf.cast(dataset['image'], tf.float32) / 255.
    # (y1, x1, y2, x2, class)
    y = dataset['objects']['bbox']
    x, y = resize(x, y)

    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 觸發影像翻轉機率50%
    x, y = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: flip(x, y), lambda: (x, y))
    # 觸發影像縮放機率50%
    x, y = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x, y), lambda: (x, y))
    # 將[y1, x1, y2, x2]合為shape=(x, 4)的Tensor
    y = tf.stack(y, axis=1)
    return x, y


def parse_fn(dataset):
    x = tf.cast(dataset['image'], tf.float32) / 255.  # 影像標準化
    # 將輸出標籤轉乘One-hot編碼
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def transform_targets(y_train, anchors, anchor_masks, classes):
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
            y_train, grid_size, anchor_idxs, classes))
        grid_size *= 2

    return tuple(y_outs)


def resize(x, bboxes, input_size=(416, 416)):
    """
        Resize Image(影像縮放)

        :param x:  image inputs, 0~1
        :param bboxes: bounding boxes inputs, shape(x, 4) "y1, x1, y2, x2", 0~1
        :param input_size: 網路輸入大小
        :return: 返回(images, bboxes), images: scale0~1, bboxes: return list [y1, x1, y2, x2], scale 0~w, h
        """
    # h, w, _ = x.shape
    ih, iw = input_size
    # scale = min(ih / h, iw / w)
    # nh = int(h * scale)
    # nw = int(w * scale)
    # image resize
    x = tf.image.resize(x, (ih, iw))
    # x = tf.image.resize_image_with_crop_or_pad(x, ih, iw)  # 影像裁減和填補

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
        rsize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (h - nh), tf.int32)
        dx = tf.random.uniform([], 0, (w - nw), tf.int32)
        indexes = tf.meshgrid(tf.range(dy, dy+nh), tf.range(dx, dx+nw), indexing='ij')
        indexes = tf.stack(indexes, axis=-1)
        output = tf.scatter_nd(indexes, rsize_x, (h, w, 3))
        return output, dx, dy

    # 如果將影像放大執行以下以下程式
    def scale_greater_then_one():
        rsize_x = tf.image.resize(x, (nh, nw))  # 影像縮放
        dy = tf.random.uniform([], 0, (nh - h), tf.int32)
        dx = tf.random.uniform([], 0, (nw - w), tf.int32)
        return rsize_x[dy:dy + h, dx:dx + w], -dx, -dy

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
    # 取得訓練數據，並順便讀取data的資訊
    train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
    # 取得驗證數據
    valid_data = tfds.load("voc2007", split=tfds.Split.VALIDATION)
    # 取得測試數據
    test_data = tfds.load("voc2007", split=tfds.Split.TEST)

    # h, w = (416, 416)
    # images = np.zeros((h * 4, w * 4, 3))
    # for i in range(4):
    #     for j, data in enumerate(train_data.take(4)):
    #         img = data['image']
    #         bboxes = data['objects']['bbox']  # (ymin, xmin, ymax, xmax)
    #         img = tf.cast(img, tf.float32) / 255.
    #         img, bboxes = resize(img, bboxes)
    #         img, bboxes = zoom(img, bboxes)
    #         bboxes = tf.stack(bboxes, axis=1).numpy()
    #         img = img.numpy()
    #         for box in bboxes:
    #             y1 = tf.cast(box[0], tf.int16).numpy()
    #             x1 = tf.cast(box[1], tf.int16).numpy()
    #             y2 = tf.cast(box[2], tf.int16).numpy()
    #             x2 = tf.cast(box[3], tf.int16).numpy()
    #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         images[h * i:h * (i + 1), w * j:w * (j + 1)] = img
    # plt.figure(figsize=(12, 12))
    # plt.imshow(images)
    # plt.show()


    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
    train_data = train_data.shuffle(1000)  # 打散資料集
    train_data = train_data.map(map_func=parse_aug_fn, num_parallel_calls=AUTOTUNE)
    h, w = (416, 416)
    images = np.zeros((h * 4, w * 4, 3))
    for i in range(4):
        for j, [img, bboxes] in enumerate(train_data.take(4)):
            # img = data['image']
            # bboxes = data['objects']['bbox']  # (ymin, xmin, ymax, xmax)
            # img = tf.cast(img, tf.float32) / 255.
            # img, bboxes = resize(img, bboxes)
            # img, bboxes = flip(img, bboxes)
            # img, bboxes = zoom(img, bboxes)
            # bboxes = tf.stack(bboxes, axis=1)
            img = img.numpy()
            for box in bboxes:
                y1 = tf.cast(box[0], tf.int16).numpy()
                x1 = tf.cast(box[1], tf.int16).numpy()
                y2 = tf.cast(box[2], tf.int16).numpy()
                x2 = tf.cast(box[3], tf.int16).numpy()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            images[h * i:h * (i + 1), w * j:w * (j + 1)] = img
    plt.figure(figsize=(12, 12))
    plt.imshow(images)
    plt.show()
