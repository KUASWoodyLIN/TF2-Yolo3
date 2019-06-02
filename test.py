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


# Load dataset
test_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)
# Dataset info
classes_list = info.features['labels'].names
num_classes = info.features['labels'].num_classes


def test_and_show_result(model, test_number=10):
    for data in test_data.take(test_number):
        org_img = data['image'].numpy()
        h, w, _ = data['image'].shape
        img, bboxes = parse_fn_test(data)
        #  Predict
        boxes, scores, classes, nums = model.predict(tf.expand_dims(img, axis=0))
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], int(nums[0])
        # Draw predict bounding box
        print(nums)
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * (w, h)).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * (w, h)).astype(np.int32))
            cv2.rectangle(org_img, x1y1, x2y2, (255, 0, 0), 2)
            cv2.putText(org_img,
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
            cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(org_img,
                        label,
                        (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        plt.figure()
        plt.imshow(org_img)
    plt.show()


def main():

    # Create model & Load weights
    model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, training=False)
    model.summary()

    # Freeze layers
    darknet = model.get_layer('Yolo_DarkNet')
    trainable_model(darknet, trainable=False)

    # Load weights
    print('weights loaded ', config.yolo_voc_weights)
    # model.load_weights(config.yolo_voc_weights, by_name=True)
    model.load_weights('logs-yolo-add-rotate/models/best-model-ep087.h5')

    test_and_show_result(model, test_number=1)


if __name__ == '__main__':
    main()
