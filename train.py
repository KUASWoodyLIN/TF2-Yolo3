import config
import tensorflow as tf
import tensorflow_datasets as tfds
from losses import YoloLoss
from model.yolo import yolo_body
from utils.dataset import parse_aug_fn, parse_fn, transform_targets



def main():
    inputs_ = tf.keras.Input((416, 416, 3))
    model = yolo_body(inputs_)
    anchors = config.yolo_anchors
    anchor_masks = config.yolo_anchor_masks

    # dataset setting
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
    train_data, info = tfds.load("voc2007", split=tfds.Split.TRAIN, with_info=True)    # 取得訓練數據
    valid_data = tfds.load("voc2007", split=tfds.Split.VALIDATION)    # 取得驗證數據
    train_data = train_data.shuffle(1000)  # 打散資料集
    train_data = train_data.map(lambda dataset: parse_aug_fn(dataset), num_parallel_calls=AUTOTUNE)
    train_data = train_data.batch(config.batch_size)
    train_data = train_data.map(lambda x, y: transform_targets(x, y, anchors, anchor_masks),
                                num_parallel_calls=AUTOTUNE)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    valid_data = valid_data.batch(config.batch_size)
    valid_data = valid_data.map(lambda dataset: parse_fn(dataset), num_parallel_calls=AUTOTUNE)
    valid_data = valid_data.prefetch(buffer_size=AUTOTUNE)

    # Callbacks function
    log_dir = 'yolo-logs'
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(log_dir + 'models/best-model.hdf5',
                                           monitor='val_categorical_accuracy',  # TODO: mAP
                                           save_best_only=True,
                                           mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1),
    ]

    # TODO: Load weights

    # TODO: freeze layers

    # Training
    optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)
    loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(train_data,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        validation_data=valid_data)


if __name__ == '__main__':
    print('Start training')