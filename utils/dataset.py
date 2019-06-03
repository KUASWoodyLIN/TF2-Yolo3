import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np


# @tf.function
def parse_aug_fn(dataset, input_size=(416, 416)):
    """
    Image Augmentation function
    """
    ih, iw = input_size
    # (None, None, 3)
    x = tf.cast(dataset['image'], tf.float32)
    # (y1, x1, y2, x2, class)
    bbox = dataset['objects']['bbox']
    label = tf.cast(dataset['objects']['label'], tf.float32)

    x, bbox = resize(x, bbox, input_size)

    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: color(x), lambda: x)
    # 觸發影像翻轉機率50%
    x, bbox = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: flip(x, bbox), lambda: (x, bbox))
    # 觸發影像縮放機率50%
    x, bbox, label = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x, bbox, label), lambda: (x, bbox, label))
    # 觸發影像旋轉機率50%
    x, bbox, label = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: rotate(x, bbox, label), lambda: (x, bbox, label))

    # normalization
    x = tf.clip_by_value(x, 0, 255)
    x = x / 255.

    # 將[x1, y1, x2, y2, classes]合為shape=(x, 5)的Tensor
    y = tf.stack([bbox[1], bbox[0], bbox[3], bbox[2], label], axis=-1)
    y = tf.divide(y, [ih, iw, ih, iw, 1])
    paddings = [[0, 100 - tf.shape(y)[0]], [0, 0]]
    y = tf.pad(y, paddings)
    y = tf.ensure_shape(y, (100, 5))
    return x, y


def parse_fn(dataset, input_size=(416, 416)):
    ih, iw = input_size
    # (None, None, None, 3)
    x = tf.cast(dataset['image'], tf.float32)
    # (None, [y1, x1, y2, x2])
    bbox = dataset['objects']['bbox']
    # (None, classes, 1)
    label = tf.cast(dataset['objects']['label'], tf.float32)
    # 將影像縮放到模型輸入大小
    x, bbox = resize(x, bbox, input_size)

    # normalization
    x = tf.clip_by_value(x, 0, 255)
    x = x / 255.

    # 將[x1, y1, x2, y2, classes]合為shape=(x, 5)的Tensor
    y = tf.stack([bbox[1], bbox[0], bbox[3], bbox[2], label], axis=-1)
    y = tf.divide(y, [ih, iw, ih, iw, 1])
    paddings = [[0, 100 - tf.shape(y)[0]], [0, 0]]
    y = tf.pad(y, paddings)
    y = tf.ensure_shape(y, (100, 5))
    return x, y


def parse_fn_test(dataset, input_size=(416, 416)):
    # (None, None, None, 3)
    x = tf.cast(dataset['image'], tf.float32)
    # (None, [y1, x1, y2, x2])
    bbox = dataset['objects']['bbox']
    # (None, classes, 1)
    label = tf.cast(dataset['objects']['label'], tf.float32)

    x, bbox = resize(x, bbox, input_size)

    # normalization
    x = tf.clip_by_value(x, 0, 255)
    x = x / 255.

    # 將[x1, y1, x2, y2, classes]合為shape=(x, 5)的Tensor
    y = tf.stack([bbox[1], bbox[0], bbox[3], bbox[2], label], axis=-1)
    return x, y


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """
        產生一個輸出層的訓練標籤
        (batch, 100, [x1, y1, x2, y2, class, best_anchor]) -> (batch, grid, grid, anchor, [x, y, w, h, obj, class])
        :param y_true: shape: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
        :param grid_size: grid cell 的大小
        :param anchor_idxs: 該輸出層anchor boxes的索引值(每層有3個anchor boxes所以有三個索引值)
        :return:(batch, grid, grid, anchor, [x, y, w, h, obj, class])
        """
    batch = tf.shape(y_true)[0]
    # 創建訓練標籤 y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((batch, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(batch):
        for j in tf.range(tf.shape(y_true)[1]):
            # 如果w=0代表沒有物件
            if tf.equal(y_true[i][j][2], 0):
                continue
            # 該層使用的anchor boxes是否是與真實物件框IoU最高的anchor box
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                # box: (x1, y1, x2, y2)
                box = y_true[i][j][0:4]
                # box中心點
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                # 找出最高IoU的anchor box索引值
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                # 計算該anchor box位於哪個grid cell中
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # 紀錄要填入y_true_out的索引值
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                # 紀錄要填入y_true_out的數值(x, y, w, h, 1(有物件), class)
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1
    # 將剛紀錄的y_true數值跟新到y_true_out中
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(x_train, y_train, anchors, anchor_masks, grid_size=13):
    """
        transform y_label to training target label,
        (batch, 100, [y1, x1, y2, x2, class])→(batch, grid, grid, anchor, [x, y, w, h, obj, class])
        :param x_train: shape: (None, 416, 416, 3)
        :param y_train: shape: (None, 100, [x1, y1, x2, y2, class])
        :param anchors: 9個預設的anchors box，shape: (9,2)
        :param anchor_masks: anchors box的遮罩
        :return:
            x_train: 訓練影像，shape: (batch, img_h, img_w, 3)
            y_outs: 返回三個不同層輸出的訓練資料
                    ((batch, grid, grid, 3, [x, y, w, h, obj, class, best_anchor]),
                     (batch, grid, grid, 3, [x, y, w, h, obj, class, best_anchor]),
                     (batch, grid, grid, 3, [x, y, w, h, obj, class, best_anchor]))
        """
    y_outs = []

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
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
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


def zoom(x, bboxes, label, scale_min=0.6, scale_max=1.6):
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

    output, dx, dy = tf.case([(tf.logical_or(tf.less(nh - h, 0), tf.less(nw - w, 0)), scale_less_then_one),
                              (tf.logical_or(tf.greater(nh - h, 0), tf.greater(nw - w, 0)), scale_greater_then_one)],
                             default=scale_equal_zero)

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
    label = label[bboxes_filter]
    output = tf.ensure_shape(output, x.shape)
    return output, [y1, x1, y2, x2], label


def rotate(img, bboxes, label, angle=(-45, 45)):
    h, w, c = img.shape
    cx, cy = w // 2, h // 2
    angle = tf.random.uniform([], angle[0], angle[1], tf.float32)
    # angle = random.uniform(*angle)
    # angle = np.random.uniform(*angle)

    theta = np.pi * angle / 180
    output = tfa.image.rotate(img, theta)

    # convert (ymin, xmin, ymax, xmax) to corners
    width = bboxes[3] - bboxes[1]
    height = bboxes[2] - bboxes[0]
    x1 = bboxes[1]
    y1 = bboxes[0]
    x2 = x1 + width
    y2 = y1
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[3]
    y4 = bboxes[2]
    corners = tf.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=-1)

    # calculate the rotate bboxes
    corners = tf.reshape(corners, (-1, 2))
    corners = tf.concat((corners, tf.ones((tf.shape(corners)[0], 1), dtype=corners.dtype)), axis=-1)

    alpha = tf.cos(theta)
    beta = tf.sin(theta)
    M = tf.reshape(tf.stack([alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy]),
                   (2, 3))
    # M = cv2.getRotationMatrix2D((cx, cy), tf.cast(angle, np.float), 1.0)# .astype(np.float32)
    corners = tf.matmul(corners, M, transpose_b=True)
    corners = tf.reshape(corners, (-1, 8))

    # convert corners to (xmin, ymin, xmax, ymax)
    x_ = corners[:, ::2]
    y_ = corners[:, 1::2]
    x1 = tf.reduce_min(x_, axis=-1)
    y1 = tf.reduce_min(y_, axis=-1)
    x2 = tf.reduce_max(x_, axis=-1)
    y2 = tf.reduce_max(y_, axis=-1)

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
    label = label[bboxes_filter]
    return output, [y1, x1, y2, x2], label


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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    def test_augmentation():
        # 取得訓練數據，並順便讀取data的資訊
        train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
        train_data = train_data.shuffle(1000)  # 打散資料集
        train_data = train_data.map(lambda dataset: parse_aug_fn(dataset, (416, 416)),
                                    num_parallel_calls=AUTOTUNE)
        classes_list = info.features['labels'].names

        h, w = (416, 416)
        images = np.zeros((h * 4, w * 4, 3))
        for count, [img, bboxes] in enumerate(train_data.take(16)):
            bboxes = tf.multiply(bboxes, [h, w, h, w, 1])
            img = img.numpy()
            box_indices = tf.where(tf.reduce_sum(bboxes, axis=-1))
            bboxes = tf.gather_nd(bboxes, box_indices)
            for box in bboxes:
                x1 = tf.cast(box[0], tf.int16).numpy()
                y1 = tf.cast(box[1], tf.int16).numpy()
                x2 = tf.cast(box[2], tf.int16).numpy()
                y2 = tf.cast(box[3], tf.int16).numpy()
                label = classes_list[box[4]]
                print(label)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1, 0), 2)
                cv2.putText(img,
                            label,
                            (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 1, 0), 2)

            i = count // 4
            j = count % 4
            images[h * i:h * (i + 1), w * j:w * (j + 1)] = img
        plt.figure(figsize=(12, 12))
        plt.imshow(images)
        plt.show()


    def test_label_transform():
        # 取得訓練數據，並順便讀取data的資訊
        train_data = tfds.load("voc2007", split=tfds.Split.TRAIN)
        train_data = train_data.shuffle(1000)  # 打散資料集
        train_data = train_data.map(lambda dataset: parse_aug_fn(dataset, (416, 416)),
                                    num_parallel_calls=AUTOTUNE)
        train_data = train_data.batch(32)
        train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks, 19),
                                    num_parallel_calls=AUTOTUNE)
        train_data = train_data.prefetch(buffer_size=AUTOTUNE)

        for x_batch, y_batch in train_data.take(500):
            print(True)

    def test():
        import os
        import psutil
        import gc
        memory_used = []
        for i in range(30):
            # 取得訓練數據，並順便讀取data的資訊
            train_data = tfds.load("voc2007", split=tfds.Split.TRAIN)
            train_data = train_data.shuffle(1000)  # 打散資料集
            train_data = train_data.map(lambda dataset: parse_aug_fn(dataset, (608, 608)),
                                        num_parallel_calls=AUTOTUNE)
            train_data = train_data.batch(32)
            # train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks, 19),
            #                             num_parallel_calls=AUTOTUNE)
            train_data = train_data.prefetch(buffer_size=AUTOTUNE)

            for x_batch, y_batch in train_data.take(5):
                print(i, True)
            memory_used.append(psutil.virtual_memory().used / 2 ** 30)
            gc.collect()

        plt.plot(memory_used)
        plt.title('Evolution of memory')
        plt.xlabel('iteration')
        plt.ylabel('memory used (GB)')
        plt.show()

    # test augmentation
    # test()
    # Augmentation test
    test_augmentation()
    # Targets transform test
    # test_label_transform()
