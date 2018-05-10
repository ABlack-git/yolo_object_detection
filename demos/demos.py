from network.yolo_v0_1 import YoloV01
import cv2
import utils
import os


def pretrained_model_test():
    path = ''
    imgs = utils.list_of_images(path)
    params = None
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV01(grid_size, img_size, params, restore=True)
    meta_path = ''
    weight_path = ''
    net.restore(weight_path, meta_path)
    for img_path in imgs:
        img = cv2.imread(os.path.join(path, img_path))
        img = utils.resize_img(img, img_size[0], img_size[1])
        preds = net.get_predictions(img)
        coords = net.predictions_to_cells(preds)
        utils.draw_centers(img, coords)
        utils.draw_grid(grid_size)
        cv2.imshow('Network test', img)
        k = cv2.waitKey(0)


if __name__ == '__main__':
    pretrained_model_test()
