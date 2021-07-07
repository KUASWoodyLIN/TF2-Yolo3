import os
import config
import tensorflow as tf
import tensorflow_datasets as tfds
from losses import YoloLoss
from model import yolov3
from utils import parse_aug_fn, parse_fn, transform_targets, trainable_model

# Anchors setting
anchor_masks = config.yolo_anchor_masks


def training_model(model, callbacks, num_classes=80, step=1):
    if step == 1:
        batch_size = config.step1_batch_size
        learning_rate = config.step1_learning_rate
        start_epochs = config.step1_start_epochs
        end_epochs = config.step1_end_epochs

    else:
        batch_size = config.step2_batch_size
        learning_rate = config.step2_learning_rate
        start_epochs = config.step2_start_epochs
        end_epochs = config.step2_end_epochs
    anchors = config.yolo_anchors / 416

    # Training dataset setting
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
    combined_split = 'train+validation'
    train_data = tfds.load("voc", split=combined_split)    # 取得訓練數據
    train_data = train_data.shuffle(1000)  # 打散資料集
    train_data = train_data.map(lambda dataset: parse_aug_fn(dataset), num_parallel_calls=AUTOTUNE)
    train_data = train_data.batch(batch_size)
    train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks),
                                num_parallel_calls=AUTOTUNE)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    # Validation dataset setting
    val_data = tfds.load("voc", split='test')
    val_data = val_data.map(lambda dataset: parse_fn(dataset), num_parallel_calls=AUTOTUNE)
    val_data = val_data.batch(batch_size)
    val_data = val_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks), num_parallel_calls=AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=AUTOTUNE)

    # Training
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=[YoloLoss(anchors[mask], num_classes=num_classes) for mask in anchor_masks],
                  run_eagerly=False)
    model.fit(train_data,
              epochs=end_epochs,
              callbacks=callbacks,
              validation_data=val_data,
              initial_epoch=start_epochs)


def main():
    # Dataset Info
    num_classes = len(config.voc_classes)

    # Create model
    model = yolov3((config.size_h, config.size_w, 3), num_classes=num_classes, training=True)
    model.summary()

    # Load Weights
    model.load_weights(config.yolo_weights, by_name=True)

    # Callbacks function
    log_dir = 'logs_yolo'
    model_dir = log_dir + '/models'
    os.makedirs(model_dir, exist_ok=True)
    model_tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = tf.keras.callbacks.ModelCheckpoint(model_dir + '/best_{epoch:03d}.h5',
                                                    monitor='val_loss',  # TODO: mAP
                                                    save_best_only=True,
                                                    mode='min')
    model_ep = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1)
    mdoel_rlr = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)

    # Freeze all layers in except last layer
    trainable_model(model, trainable=False)
    model.get_layer('conv2d_last_layer1_20').trainable = True
    model.get_layer('conv2d_last_layer2_20').trainable = True
    model.get_layer('conv2d_last_layer3_20').trainable = True

    # 1) Training model step1
    print("Start teraining Step1")
    training_model(model,
                   callbacks=[model_tb, model_mckp, mdoel_rlr, model_ep],
                   num_classes=num_classes,
                   step=1)

    # Unfreeze layers
    trainable_model(model, trainable=True)

    # 2) Training model step2
    print("Start teraining Step2")
    training_model(model,
                   callbacks=[model_tb, model_mckp, mdoel_rlr, model_ep],
                   num_classes=num_classes,
                   step=2)


if __name__ == '__main__':
    main()
