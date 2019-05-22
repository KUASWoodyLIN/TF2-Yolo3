import tensorflow as tf
from absl import flags, logging
from absl.flags import FLAGS
import numpy as np
from model import yolov3, yolov3_tiny
from utils.utils import load_darknet_weights
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tiny_model = False


def main():
    inputs = tf.keras.Input((416, 416, 3))
    if tiny_model:
        yolo = yolov3_tiny(inputs)
        yolo_darknet_weights = 'model_data/yolov3-tiny.weights'
    else:
        yolo = yolov3(inputs)
        yolo_darknet_weights = 'model_data/yolov3.weights'

    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, yolo_darknet_weights)
    logging.info('weights loaded')

    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights('model_data/yolo_weights.h5')
    logging.info('weights saved')


if __name__ == '__main__':
    main()
