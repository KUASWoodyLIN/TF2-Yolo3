import numpy as np

# Yolo parameter
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32)
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

# Yolo setting
size_h = 416
size_w = 416

# Training setting
tiny = False
if tiny:
    step1_batch_size = 32
    step1_learning_rate = 1e-3
    step1_start_epochs = 0
    step1_end_epochs = 50
    step2_batch_size = 14
    step2_learning_rate = 1e-4
    step2_start_epochs = 50
    step2_end_epochs = 100
else:
    step1_batch_size = 32
    step1_learning_rate = 1e-3
    step1_start_epochs = 0
    step1_end_epochs = 100
    step2_batch_size = 8
    step2_learning_rate = 1e-4
    step2_start_epochs = step1_end_epochs
    step2_end_epochs = step1_end_epochs + 100

# Pre-Train weights
yolo_weights = 'model_data/yolo_weights.h5'
yolo_tiny_weights = 'model_data/yolo_tiny_weights.h5'

# Our Yolo weights
yolo_voc_weights = 'logs_yolo/models/best_011.h5'
yolo_coco_weights = 'logs_yolo/models/best_001.h5'

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']