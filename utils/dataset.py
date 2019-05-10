import tensorflow as tf
import tensorflow_datasets as tfds

train_data, info = tfds.load("coco2014", split=tfds.Split.TRAIN, with_info=True)
valid_data = tfds.load("coco2014", split=tfds.Split.VALIDATION)
test_data = tfds.load("coco2014", split=tfds.Split.TEST)
labels = info.features['labels'].names


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
    # (x1, y1, x2, y2, class)
    y = dataset['objects']['bbox']
    x, y = flip(x, y)

    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 觸發影像旋轉機率0.25%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(x), lambda: x)
    # 觸發影像縮放機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    # 將輸出標籤轉乘One-hot編碼
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


def flip(x, y):
    """
    flip image(翻轉影像)
    """
    x = tf.image.random_flip_left_right(x)  # 隨機左右翻轉影像
    y[..., 2::-2] = 1 - y[..., 0::2]
    return x, y

def color(x):
    """
     Color change(改變顏色)
    """
    x = tf.image.random_hue(x, 0.08)  # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0.6, 1.6)  # 隨機調整影像飽和度
    x = tf.image.random_brightness(x, 0.05)  # 隨機調整影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3)  # 隨機調整影像對比度
    return x

def rotate(x):
    """
    Rotation image(影像旋轉)
    """
    # 隨機選轉n次(通過minval和maxval設定n的範圍)，每次選轉90度
    x = tf.image.rot90(x,tf.random.uniform(shape=[],minval=1,maxval=4,dtype=tf.int32))
    return x

def zoom(x, scale_min=0.6, scale_max=1.4):
    """
    Zoom Image(影像縮放)
    """
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)  # 隨機縮放比例
    sh = h * scale  # 縮放後影像長度
    sw = w * scale  # 縮放後影像寬度
    x = tf.image.resize(x, (sh, sw))  # 影像縮放
    x = tf.image.resize_image_with_crop_or_pad(x, h, w)  # 影像裁減和填補

    box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
    box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
    return x


def parse_aug_fn(dataset):
    """
    Image Augmentation(影像增強) function
    """
    x = tf.cast(dataset['image'], tf.float32) / 255.  # 影像標準化
    x = tf.image.resize(x, input_shape)
    x = flip(x)  # 隨機水平翻轉
    # 觸發顏色轉換機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 觸發影像旋轉機率0.25%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(x), lambda: x)
    # 觸發影像縮放機率50%
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    return x, dataset['label']

def parse_fn(dataset):
    x = tf.cast(dataset['image'], tf.float32) / 255.  # 影像標準化
    x = tf.image.resize(x, input_shape)
    return x, dataset['label']