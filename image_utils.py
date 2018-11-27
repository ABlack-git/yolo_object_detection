import cv2
import numpy as np
import os
import bbox_utils as bbu
import stats_utils as su


def draw_bbox(data, img, color=(0, 139, 0), thickness=2):
    data = np.array(data)
    # data = bbu.convert_center_to_2points(data)
    for rect in data:
        if rect.size != 0:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)


def draw_grid(img, grid_size):
    grid_w = int(img.shape[1] / grid_size[0])
    grid_h = int(img.shape[0] / grid_size[1])
    for n in range(grid_size[0]):
        cv2.line(img, (n * grid_w, 0), (n * grid_w, img.shape[1]), (0, 0, 0), 1)
    for m in range(grid_size[1]):
        cv2.line(img, (0, m * grid_h), (img.shape[1], m * grid_h), (0, 0, 0), 1)


def draw_centers(img, boxes):
    for _ in boxes:
        for box in boxes:
            if len(box) > 0:
                cv2.circle(img, (box[0], box[1]), 2, (0, 0, 255), -1)


def draw_text(text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (20, 40), font, 1.2, (0, 0, 200), 3, cv2.LINE_AA)


def list_of_images(path):
    imgs = []
    if isinstance(path, list):
        for folder in path:
            imgs += [os.path.join(folder, f) for f in os.listdir(folder) if
                     f.endswith('.jpg') and not f.startswith('.')]
    elif isinstance(path, str):
        imgs = [f for f in os.listdir(path) if f.endswith('.jpg') and not f.startswith('.')]
    else:
        print('input argument should be of type list or string')
    return imgs


def resize_img(img, h_new, w_new, keep_asp_ratio=True):
    height, width = img.shape[:2]
    if keep_asp_ratio:
        as_nom, as_denom = su.compute_aspectratio(width, height)  # 16:9,4:3,3:2, etc..
        as_nom_new, as_denom_new = su.compute_aspectratio(w_new, h_new)
        if (as_nom == as_nom_new) and (as_denom == as_denom_new):
            pass
        else:
            tmp_h = int(w_new * (as_denom / as_nom))
            if tmp_h > h_new:
                w_new = int(h_new * (as_nom / as_denom))
            else:
                h_new = tmp_h

    if height > h_new or width > w_new:
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return img


def pad_img(img, h_new, w_new):
    h, w = img.shape[:2]
    if h == h_new and w == w_new:
        return img
    if h > h_new or w > w_new:
        raise ValueError('Height and width of the input image should be less than height and with of the output.')

    top = int(np.ceil((h_new - h) / 2))
    bottom = int(np.floor((h_new - h) / 2))
    left = int(np.ceil((w_new - w) / 2))
    right = int(np.floor((w_new - w) / 2))

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
