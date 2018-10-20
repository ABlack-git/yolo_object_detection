import numpy as np


def prediction_to_boxes(labels, grid_size, img_size, o_img_size):
    # print(labels[0, :, 0])
    w_ratio = o_img_size[0] / img_size[0]
    h_ratio = o_img_size[1] / img_size[1]
    g_i = np.zeros([grid_size[0] * grid_size[1]])
    g_j = np.zeros([grid_size[0] * grid_size[1]])
    counter_i = 0
    counter_j = 0
    for i in range(grid_size[0] * grid_size[1]):
        g_i[i] = counter_i
        g_j[i] = counter_j
        counter_i += 1
        if (i + 1) % grid_size[0] == 0:
            counter_i = 0
            counter_j += 1
    # print(Gi)
    print(img_size[0] / grid_size[0])
    labels[:, :, 0] = (labels[:, :, 0] + g_i) * img_size[0] / grid_size[0] * w_ratio
    labels[:, :, 1] = (labels[:, :, 1] + g_j) * img_size[1] / grid_size[1] * h_ratio
    labels[:, :, 2] = np.power(labels[:, :, 2] * img_size[0], 2) * w_ratio
    labels[:, :, 3] = np.power(labels[:, :, 3] * img_size[1], 2) * h_ratio
    for row in labels[0, :, :]:
        print(row)


def convert_topleft_to_centre(bboxes):
    if isinstance(bboxes, (list,)):
        bboxes = np.array(bboxes)
    bboxes[:, 0] = bboxes[:, 0] + np.round(np.divide(bboxes[:, 2], 2))
    bboxes[:, 1] = bboxes[:, 1] + np.round(np.divide(bboxes[:, 3], 2))
    return bboxes


def convert_2points_to_center(bboxes):
    if isinstance(bboxes, (list,)):
        bboxes = np.array(bboxes)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + np.round(np.divide(bboxes[:, 2], 2))
    bboxes[:, 1] = bboxes[:, 1] + np.round(np.divide(bboxes[:, 3], 2))
    return bboxes


def convert_topleft_to_2points(bboxes):
    if isinstance(bboxes, (list,)):
        bboxes = np.array(bboxes)
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


def convert_center_to_2points(bboxes):
    if isinstance(bboxes, (list,)):
        bboxes = np.array(bboxes)
    if bboxes.size == 0:
        return bboxes
    if len(bboxes.shape) < 2:
        bboxes = np.array(bboxes, ndmin=2)
    bboxes[:, 0] = bboxes[:, 0] - np.round(np.divide(bboxes[:, 2], 2))
    bboxes[:, 1] = bboxes[:, 1] - np.round(np.divide(bboxes[:, 3], 2))
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


def resize_boxes(boxes, old_size, new_size):
    # size[0] is height
    # size[1] is width
    if old_size[1] > new_size[1] or old_size[0] > new_size[0]:
        w_ratio = new_size[1] / old_size[1]
        h_ratio = new_size[0] / old_size[0]
        boxes = np.array(boxes)
        boxes[:, [0, 2]] = np.round(boxes[:, [0, 2]] * w_ratio)
        boxes[:, [1, 3]] = np.round(boxes[:, [1, 3]] * h_ratio)
    return boxes


def iou(boxes_a, boxes_b, epsilon=1e-5):
    """
    Computes intersection over union of boxes_a and boxes_b
    :param boxes_a:
    :param boxes_b:
    :return:
    """
    if len(boxes_a.shape) < 2:
        boxes_a = np.array(boxes_a, ndmin=2)
    if len(boxes_b.shape) < 2:
        boxes_b = np.array(boxes_b, ndmin=2)
    x_a = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    y_a = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    x_b = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    y_b = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter_area = np.multiply(np.maximum((x_b - x_a + 1), 0), np.maximum((y_b - y_a + 1), 0))
    a_area = np.multiply(boxes_a[:, 2] - boxes_a[:, 0] + 1, boxes_a[:, 3] - boxes_a[:, 1] + 1)
    b_area = np.multiply(boxes_b[:, 2] - boxes_b[:, 0] + 1, boxes_b[:, 3] - boxes_b[:, 1] + 1)
    union_area = np.maximum(a_area + b_area - inter_area, epsilon)
    return np.divide(inter_area, union_area)
