from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import numpy as np

# loss_scale, training_set_imgs, training_set_labels, batch_size, learning_rate
if __name__ == '__main__':
    params = {'coord_scale': 5, 'noobj_scale': 0.5,
              'training_set_imgs': '/Volumes/TRANSCEND/Data Sets/another_testset/imgs',
              'training_set_labels': '/Volumes/TRANSCEND/Data Sets/another_testset/labels', 'batch_size': 10,
              'learning_rate': 0.01, 'optimizer': 'Adam', 'threshold': 0.5}
    img_size = (720, 480)
    grid_size = (36, 24)
    # dataset = DatasetGenerator(params.get('training_set_imgs'), params.get('training_set_labels'), img_size, grid_size,
    #                            1)
    # batch = dataset.get_minibatch(10)
    net = YoloV0(grid_size, img_size, params)
    net.set_logger_verbosity()
    net.open_sess()
    net.optimize(1)
    # img, _ = next(batch)
    # preds = net.predictions_to_boxes(x=img)
    # print('Size of prediction vector is ' + str(np.shape(preds)))
    # for item in preds[0, :, :]:
    #     print(item)
    # # net.optimize(1)
    # net.close_sess()
