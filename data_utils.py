import os
import numpy as np


def list_dir(path, file_ext):
    labels = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(file_ext) and not file.startswith('.'):
                labels.append(file)
    else:
        raise ValueError('Path should point to existing directory')
    return labels


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
