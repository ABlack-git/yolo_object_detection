from network.yolo_v0_1 import YoloV01
from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import cv2
import utils
import os
import random
import time
import argparse


def pretrained_model_test():
    path = 'E:\Andrew\Dataset\Testing set\Images'
    imgs = utils.list_of_images(path)
    random.shuffle(imgs)
    params = None
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV01(grid_size, img_size, params, restore=True)
    meta_path = "E:\Andrew\Ann_models\pretrain_model_1_0\weights\model-30600.meta"
    weight_path = "E:\Andrew\Ann_models\pretrain_model_1_0\weights\model-30600"
    net.restore(weight_path, meta_path)
    for img_path in imgs:
        img = cv2.imread(os.path.join(path, img_path))
        img = utils.resize_img(img, img_size[1], img_size[0])
        preds = net.get_predictions([img])
        coords = net.predictions_to_cells(preds)
        utils.draw_centers(img, coords)
        utils.draw_grid(img, grid_size)
        cv2.imshow('Network test', img)
        k = cv2.waitKey(0)
        if k == 27:
            break


def test_model(cfg, testing_set, path_to_parameters):
    net = YoloV0(cfg)
    net.restore(path=path_to_parameters)
    ts = DatasetGenerator(testing_set[0], testing_set[1], net.img_size, net.grid_size, net.no_boxes, sqrt=net.sqrt)
    batch = ts.get_minibatch(net.batch_size, resize_only=True)

    for _ in range(ts.get_number_of_batches(net.batch_size)):
        imgs, labels = next(batch)
        preds = net.get_predictions(imgs)


def show_images_with_boxes(cfg, testing_set, path_to_parameters, draw_centre=True, draw_grid=False, delay=0,
                           print_time=True):
    net = YoloV0(cfg)
    net.restore(path=path_to_parameters)
    list_of_imgs = utils.list_of_images(testing_set)
    compute_time = []
    for img_path in list_of_imgs:
        t0 = time.time()
        img = cv2.imread(os.path.join(testing_set, img_path))
        t_read = time.time() - t0
        img = utils.resize_img(img, net.img_size[1], net.img_size[0])
        t_resize = time.time() - t_read
        preds = net.get_predictions([img])
        t_preds = time.time() - t_resize
        if draw_centre:
            pass
        if draw_grid:
            utils.draw_grid(img, net.grid_size)
        utils.draw_bbox(preds[0], img)
        t_draw = time.time() - t_preds
        cv2.imshow(net.model_version, img)
        k = cv2.waitKey(delay)
        compute_time.append([t_read, t_resize, t_preds, t_draw])
        if print_time:
            print('Read time: %.3f, Resize time: %.3f, Prediction time: %.3f, Draw time: %.3f' % (t_read, t_resize,
                                                                                                  t_preds, t_draw))
        if k == 27:
            break
    return compute_time


def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-stats', action='store_true')
    parser.add_argument('-bboxes', action='store_true')
    # common args
    parser.add_argument('-cfg', action='store', type=str, help='Path to .cfg file')
    parser.add_argument('-images', action='store', type=str, help='Path to folder that contains images', default='')
    parser.add_argument('-labels', action='store', type=str, help='Path folder that contains labels', default='')
    parser.add_argument('-weights', action='store', type=str, help='Path to weights. Note that path to weights should '
                                                                   'not contain extension, e.g. .meta or .index.')
    # add args that go with -stats

    # args that go with -bboxes
    parser.add_argument('-draw_centers', action='store_true', help='With this flag program will draw centers of '
                                                                   'bounding boxes')
    parser.add_argument('-draw_grid', action='store_true', help='With this flag program will draw grid over image')
    parser.add_argument('-no_time', action='store_false', dest='print_time', help='With this flag program will not '
                                                                                  'output processing time')
    parser.add_argument('-delay', action='store', type=int, default=0, help='The amount of time in ms for how long '
                                                                            'single image will be shown')
    args = parser.parse_args()
    if not args.stats and not args.bboxes:
        parser.error('Either -stats or -bboxes flag should be specified.')

    if not os.path.isfile(args.cfg):
        parser.error("argument -cfg: invalid path to .cfg file.")
    if not os.path.exists(args.images):
        parser.error("argument -images: invalid path to image directory.")
    if not (args.labels == ''):
        if not os.path.exists(args.labels):
            parser.error("argument -labels: invalid path to labels directory.")
    if not os.path.isfile(args.weights + '.data-00000-of-00001'):
        parser.error('argument -weights: invalid path to weights.')
    return args


def main():
    args = fetch_args()
    if args.bboxes:
        show_images_with_boxes(args.cfg, args.images, args.weights, args.draw_centers, args.draw_grid, args.delay)

    if args.stats:
        test_model(args.cfg, [args.images, args.labels], args.weights)


if __name__ == '__main__':
    main()
