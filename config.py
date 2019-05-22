import numpy as np

# Yolo parameter
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

# Yolo setting
size_h = 416
size_w = 416

# Training setting
step1_batch_size = 30
step1_learning_rate = 1e-3
step1_epochs = 50
step2_batch_size = 10
step2_learning_rate = 1e-4
step2_epochs = 100


# Pre-Train weights
yolo_weights = 'model_data/yolo_weights.h5'
yolo_tiny_weights = 'model_data/yolo_tiny_weights.h5'
