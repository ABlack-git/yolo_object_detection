from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import numpy as np


# loss_scale, training_set_imgs, training_set_labels, batch_size, learning_rate

def first_run():
    params = {'coord_scale': 5,
              'noobj_scale': 0.5,
              'training_set_imgs': '/Volumes/TRANSCEND/Data Sets/another_testset/imgs',
              'training_set_labels': '/Volumes/TRANSCEND/Data Sets/another_testset/labels',
              'batch_size': 10,
              'learning_rate': 0.01,
              'optimizer': 'Adam',
              'threshold': 0.5}
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size, params)
    net.set_logger_verbosity()
    net.optimize(1)
    net.close_sess()


def restore_and_run():
    params = {'coord_scale': 5,
              'noobj_scale': 0.5,
              'training_set_imgs': '/Volumes/TRANSCEND/Data Sets/another_testset/imgs',
              'training_set_labels': '/Volumes/TRANSCEND/Data Sets/another_testset/labels',
              'batch_size': 10,
              'learning_rate': 0.01,
              'optimizer': 'Adam',
              'threshold': 0.5}
    model_path = ''
    meta_path = ''
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size, params, restore=True)
    net.restore(model_path, meta_path)


if __name__ == '__main__':
    first_run()
