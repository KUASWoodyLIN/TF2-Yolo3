import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy, sparse_categorical_crossentropy


def broadcast_iou(pred_box, true_box):
    # pred_box: (b, gx, gy, 3, 4)
    # true_box: (n, 4)

    # broadcast boxes
    pred_box = tf.expand_dims(pred_box, -2)   # (b, gx, gy, 3, 1, 4)
    true_box = tf.expand_dims(true_box, 0)    # (1, n, 4)
    # new_shape: (b, gx, gy, 3, n, 4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(pred_box), tf.shape(true_box))
    pred_box = tf.broadcast_to(pred_box, new_shape)
    true_box = tf.broadcast_to(true_box, new_shape)

    # (b, gx, gy, 3, n)
    int_w = tf.maximum(tf.minimum(pred_box[..., 2], true_box[..., 2]) -
                       tf.maximum(pred_box[..., 0], true_box[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(pred_box[..., 3], true_box[..., 3]) -
                       tf.maximum(pred_box[..., 1], true_box[..., 1]), 0)
    int_area = int_w * int_h

    # w * h
    box_1_area = (pred_box[..., 2] - pred_box[..., 0]) * \
        (pred_box[..., 3] - pred_box[..., 1])
    box_2_area = (true_box[..., 2] - true_box[..., 0]) * \
        (true_box[..., 3] - true_box[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def YoloLoss(anchors, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = y_pred
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        # true_box: (batch_size, grid, grid, anchors, (x1, y1, x2, y2))     # 0~1
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_h, grid_w = tf.shape(y_true)[1:3]
        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        # ex: (0.5, 0.5) * (13, 13) - (6, 6)
        true_xy = true_xy * (grid_h, grid_w) - tf.cast(grid, true_xy.dtype)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)     # (batch_size, grid, grid, anchors)
        # ignore false positive when iou is over threshold
        # true_box_flat (N, (x1, y1, x2, y2))
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        confidence_loss = obj_mask * binary_crossentropy(true_obj, pred_obj) + \
                          (1 - obj_mask) * ignore_mask * binary_crossentropy(true_obj, pred_obj)
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        confidence_loss = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + confidence_loss + class_loss
    return yolo_loss


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, y_true[0].dtype)
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], y_true[0].dtype) for l in range(num_layers)]
    loss = 0
    m = tf.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = tf.cast(m, yolo_outputs[0].dtype)

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = tf.math.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = tf.where(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, true_box.dtype))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                     from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                  from_logits=True) * ignore_mask
        class_loss = object_mask * binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, tf.reduce_sum(ignore_mask)])
    return loss