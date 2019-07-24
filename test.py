import os
import config
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from model import yolov3
from utils import parse_fn_test, trainable_model


# Load dataset
test_data = tfds.load("voc2007", split=tfds.Split.TEST)
# weight_file = 'model_data/yolo_weights.h5'      # or 'logs_yolo/models/best_xxx.h5'
weight_file = 'logs_yolo/models/best_100.h5'

if weight_file == 'model_data/yolo_weights.h5':
    # COCO weights
    classes_list = config.coco_classes
    num_classes = len(config.coco_classes)
    freeze = False
else:
    # VOC2007 weights
    classes_list = config.voc_classes
    num_classes = len(config.voc_classes)
    if int(os.path.splitext(weight_file)[0].split('_')[-1]) <= 100:
        freeze = True
# 每個類別使用不同顏色做標記
colors = (plt.cm.hsv(np.linspace(0, 1, 80)) * 255).astype(np.int).tolist()


def test_and_show_result(model, test_number=10):
    for img_count, data in enumerate(test_data.take(test_number)):
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
            cv2.rectangle(org_img, x1y1, x2y2, colors[int(classes[i])], 2)
            cv2.putText(org_img,
                        '{} {:.4f}'.format(classes_list[int(classes[i])], scores[i]),
                        x1y1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, colors[int(classes[i])], 2)
        plt.figure()
        plt.imshow(org_img)
        plt.imsave('output_images/output_{}.png'.format(img_count), org_img)
    plt.show()


def main():

    # Create model
    model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, training=False)
    model.summary()

    if freeze:
        # Freeze all layers in except last layer
        trainable_model(model, trainable=False)
        model.get_layer('conv2d_last_layer1_20').trainable = True
        model.get_layer('conv2d_last_layer2_20').trainable = True
        model.get_layer('conv2d_last_layer3_20').trainable = True

    # Load weights
    model.load_weights(weight_file)

    # Detect Object
    test_and_show_result(model, test_number=10)


if __name__ == '__main__':
    main()
