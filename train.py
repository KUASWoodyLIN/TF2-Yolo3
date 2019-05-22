import config
import tensorflow as tf
import tensorflow_datasets as tfds
from losses import YoloLoss
from model.yolo import yolov3
from utils import parse_aug_fn, parse_fn, transform_targets, trainable_model

# Anchors setting
anchors = config.yolo_anchors
anchor_masks = config.yolo_anchor_masks

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def training_model(model, callbacks, step=1):
    if step == 1:
        batch_size = config.step1_batch_size
        learning_rate = config.step1_learning_rate
        epochs = config.step1_epochs
    else:
        batch_size = config.step2_batch_size
        learning_rate = config.step2_learning_rate
        epochs = config.step2_epochs

    # Training dataset setting
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
    train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)    # 取得訓練數據
    valid_data = tfds.load("voc2007", split=tfds.Split.VALIDATION)    # 取得驗證數據
    train_data = train_data.shuffle(1000)  # 打散資料集
    train_data = train_data.map(lambda dataset: parse_aug_fn(dataset), num_parallel_calls=AUTOTUNE)
    train_data = train_data.batch(batch_size)
    train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks),
                                num_parallel_calls=AUTOTUNE)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    # Validation dataset setting
    valid_data = valid_data.batch(batch_size)
    valid_data = valid_data.map(lambda dataset: parse_fn(dataset, anchors, anchor_masks),
                                num_parallel_calls=AUTOTUNE)
    valid_data = valid_data.prefetch(buffer_size=AUTOTUNE)

    # Training
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=[YoloLoss(anchors[mask]) for mask in anchor_masks], run_eagerly=False)
    model.fit(train_data,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=valid_data)


def main():
    # Create model
    inputs_ = tf.keras.Input((config.size_h, config.size_w, 3))
    model = yolov3(inputs_, training=True)
    model.summary()

    # Load Weights
    darknet = model.get_layer('Yolo_DarkNet')
    darknet.load_weights(config.yolo_weights, by_name=True)

    # Callbacks function
    log_dir = 'logs-yolo'
    model_tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = tf.keras.callbacks.ModelCheckpoint(log_dir + 'models/best-model.hdf5',
                                                    monitor='val_categorical_accuracy',  # TODO: mAP
                                                    save_best_only=True,
                                                    mode='max')
    model_ep = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)
    mdoel_rlr = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)

    # # Freeze layers
    # trainable_model(darknet, trainable=False)
    #
    # # 1) Training model step1
    # print("Start teraining Step1")
    # training_model(model,
    #                callbacks=[model_tb, model_mckp],
    #                step=1)

    # Unfreeze layers
    darknet = model.get_layer('Yolo_DarkNet')
    trainable_model(darknet, trainable=True)

    # 2) Training model step2
    print("Start teraining Step2")
    training_model(model,
                   callbacks=[model_tb, model_mckp],
                   step=2)


if __name__ == '__main__':
    main()
