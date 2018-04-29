import cv2
import numpy as np


def draw_bbox(data, img):
    for rect in data:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 139, 0), 2)


def draw_grid(img, grid_size):
    grid_w = int(img.shape[1] / grid_size[0])
    grid_h = int(img.shape[0] / grid_size[1])
    for n in range(grid_size[0]):
        cv2.line(img, (n * grid_w, 0), (n * grid_w, img.shape[1]), (0, 0, 0), 1)
    for m in range(grid_size[1]):
        cv2.line(img, (0, m * grid_h), (img.shape[1], m * grid_h), (0, 0, 0), 1)


def draw_centers(img, boxes):
    for box in boxes:
        cv2.circle(img, (box[0], box[1]), 2, (0, 0, 255), -1)


def covert_to_centre(boxes):
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] = boxes[:, 0] + np.round(np.divide(boxes[:, 2], 2))
    boxes[:, 1] = boxes[:, 1] + np.round(np.divide(boxes[:, 3], 2))
    return boxes


def prediction_to_boxes(labels, grid_size, img_size, o_img_size):
    # print(labels[0, :, 0])
    w_ratio = o_img_size[0] / img_size[0]
    h_ratio = o_img_size[1] / img_size[1]
    Gi = np.zeros([grid_size[0] * grid_size[1]])
    Gj = np.zeros([grid_size[0] * grid_size[1]])
    counter_i = 0
    counter_j = 0
    for i in range(grid_size[0] * grid_size[1]):
        Gi[i] = counter_i
        Gj[i] = counter_j
        counter_i += 1
        if (i + 1) % grid_size[0] == 0:
            counter_i = 0
            counter_j += 1
    # print(Gi)
    print(img_size[0] / grid_size[0])
    labels[:, :, 0] = (labels[:, :, 0] + Gi) * img_size[0] / grid_size[0] * w_ratio
    labels[:, :, 1] = (labels[:, :, 1] + Gj) * img_size[1] / grid_size[1] * h_ratio
    labels[:, :, 2] = np.power(labels[:, :, 2] * img_size[0], 2) * w_ratio
    labels[:, :, 3] = np.power(labels[:, :, 3] * img_size[1], 2) * h_ratio
    for row in labels[0, :, :]:
        print(row)


def iou(boxes_a, boxes_b):
    x_a = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    y_a = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    x_b = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    y_b = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter_area = np.multiply(np.maximum((x_b - x_a + 1), 0), np.maximum((y_b - y_a + 1), 0))
    a_area = np.multiply(boxes_a[:, 2] - boxes_a[:, 0] + 1, boxes_a[:, 3] - boxes_a[:, 1] + 1)
    b_area = np.multiply(boxes_b[:, 2] - boxes_b[:, 0] + 1, boxes_b[:, 3] - boxes_b[:, 1] + 1)
    union_area = a_area + b_area - inter_area
    return np.divide(inter_area, union_area)
