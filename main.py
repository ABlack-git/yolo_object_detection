from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import numpy as np


# loss_scale, training_set_imgs, training_set_labels, batch_size, learning_rate

def first_run():
    params = {'coord_scale': 5,
              'noobj_scale': 0.15,
              'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
              'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
              'batch_size': 32,
              'learning_rate': 0.0001,
              'optimizer': 'Adam',
              'threshold': 0.25,
              'save_path': "E:\Andrew\Model_checkpoints\model_01"}
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size, params)
    net.set_logger_verbosity()
    net.optimize(5)
    net.close_sess()


def restore_and_run():
    params = {'coord_scale': 5,
              'noobj_scale': 0.15,
              'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
              'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
              'batch_size': 32,
              'learning_rate': 0.0001,
              'optimizer': 'Adam',
              'threshold': 0.25,
              'save_path': "E:\Andrew\Model_checkpoints\model_02"}
    model_path ="E:\Andrew\Model_checkpoints\model_02\model-12504"
    meta_path ="E:\Andrew\Model_checkpoints\model_02\model-12504.meta"
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size, params, restore=True)
    net.restore(model_path, meta_path)
    net.optimize(10)
    net.save(params.get('save_path'), 'model')

if __name__ == '__main__':
    restore_and_run()
