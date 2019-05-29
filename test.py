import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from model import yolov3
from utils import parse_fn_test, trainable_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
# Load dataset
test_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
test_data = test_data.map(lambda dataset: parse_fn_test(dataset), num_parallel_calls=AUTOTUNE)
# Dataset info
classes_list = info.features['labels'].names
num_classes = info.features['labels'].num_classes


def test_and_show_result(model, test_number=10):
    for img, bboxes in test_data.take(test_number):
        #  Predict
        boxes, scores, classes, nums = model.predict(tf.expand_dims(img, axis=0))
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], int(nums[0])
        img = img.numpy()
        h, w, _ = img.shape
        # Draw predict bounding box
        print(nums)
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * (w, h)).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * (w, h)).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img,
                              '{} {:.4f}'.format(classes_list[int(classes[i])], scores[i]),
                              x1y1,
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255, 0, 0), 2)

        # draw ground truth bounding box
        for box in bboxes:
            x1 = tf.cast(box[0], tf.int16).numpy()
            y1 = tf.cast(box[1], tf.int16).numpy()
            x2 = tf.cast(box[2], tf.int16).numpy()
            y2 = tf.cast(box[3], tf.int16).numpy()
            label = classes_list[box[4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img,
                        label,
                        (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        plt.figure()
        plt.imshow(img)
    plt.show()


def main():

    # Create model & Load weights
    model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, iou_threshold=0.3, score_threshold=0.3, training=False)
    model.summary()

    # Freeze layers
    darknet = model.get_layer('Yolo_DarkNet')
    trainable_model(darknet, trainable=False)

    # Load weights
    print('weights loaded ', config.yolo_voc_weights)
    # model.load_weights(config.yolo_voc_weights, by_name=True)
    model.load_weights('logs-yolo-add-rotate/models/best-model-ep087.h5')

    test_and_show_result(model)


if __name__ == '__main__':
    main()