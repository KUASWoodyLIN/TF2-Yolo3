import tensorflow as tf
import numpy as np
from model import yolov3, yolov3_tiny
from utils.utils import load_darknet_weights
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tiny_model = False


def main():
    input_size = (416, 416, 3)
    if tiny_model:
        yolo = yolov3_tiny(input_size)
        yolo_darknet_weights = 'model_data/yolov3_tiny.weights'
    else:
        yolo = yolov3(input_size)
        yolo_darknet_weights = 'model_data/yolov3.weights'

    yolo.summary()
    print('model created')

    load_darknet_weights(yolo, yolo_darknet_weights)
    print('weights loaded')

    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    print('sanity check passed')

    yolo.save_weights('model_data/yolo_weights.h5')
    print('weights saved')


if __name__ == '__main__':
    main()
