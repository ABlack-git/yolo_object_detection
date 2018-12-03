import os
import numpy as np

DATA_EXTENSIONS = ('.MOV', '.mov', '.MP4', '.mp4', '.mpg', '.jpg')
LABELS_EXTENSIONS = ('.txt', '.xgtf')


def list_dirs(paths, file_ext):
    if isinstance(paths, str):
        paths = [paths]
    labels = []
    for p in paths:
        if os.path.isdir(p):
            for file in os.listdir(p):
                if file.endswith(file_ext) and not file.startswith('.'):
                    labels.append(os.path.join(p, file))
        else:
            raise ValueError('Path should point to existing directory')
    return labels


def match_imgs_with_labels(imgs: list, labels: list):
    imgs.sort()
    labels.sort()
    return imgs, labels


def get_boxes(txt_name):
    with open(txt_name, 'r') as file:
        coords = file.read().splitlines()
    if coords[0] == 'None':
        return None
    new_coords = []
    for i, line in enumerate(coords):
        tmp = line.split()
        new_coords.append([int(x) for x in tmp])
    return np.array(new_coords)
