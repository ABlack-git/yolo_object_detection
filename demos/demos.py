from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import cv2
import image_utils
import stats_utils as su
import os
import time
import argparse
import json


def test_model(net_cfg, path_to_parameters, images, labels, iou_threshold):
    net = YoloV0(net_cfg)
    net.restore(path=path_to_parameters)
    # construct json
    img_size = {"width": net.img_size[0], "height": net.img_size[1]}
    grid_size = {"width": net.grid_size[0], "height": net.grid_size[1]}
    params = {"img_size": img_size, "grid_size": grid_size, "no_boxes": net.no_boxes, "shuffle": True, "sqrt": net.sqrt}
    conf = {"images": images, "annotations": labels, "configuration": params}
    conf = json.dumps(conf)

    ts = DatasetGenerator(conf)
    batch = ts.get_minibatch(net.batch_size, resize_only=True)
    stats = []
    for _ in range(ts.get_number_of_batches(net.batch_size)):
        # labels are only resized
        imgs, labels = next(batch)
        preds = net.get_predictions(imgs)
        # compute stats for batch
        stats = su.compute_stats(preds, labels, iou_threshold, stats)
    final_stats = su.process_stats(stats)
    print('Average precision: {0[0]}, Average recall: {0[1]}, Average iou: {0[2]}, Average confidence of TP: {0[3]}, '
          'Average confidence of FP: {0[4]}, Total num of TP: {0[5]}, Total num of FP: {0[6]}, '
          'Total num of FN: {0[7]}'.format(final_stats))


def show_images_with_boxes(cfg, testing_set, path_to_parameters, draw_centre=True, draw_grid=False, delay=0,
                           print_time=True):
    net = YoloV0(cfg)
    net.restore(path=path_to_parameters)
    list_of_imgs = image_utils.list_of_images(testing_set)
    compute_time = []
    for img_path in list_of_imgs:
        t0_read = time.time()
        img = cv2.imread(os.path.join(testing_set, img_path))
        t_read = time.time() - t0_read
        t0_resize = time.time()
        img = image_utils.resize_img(img, net.img_size[1], net.img_size[0])
        t_resize = time.time() - t0_resize
        t0_pred = time.time()
        preds = net.get_predictions([img])
        t_preds = time.time() - t0_pred
        t0_draw = time.time()
        if draw_centre and preds:
            image_utils.draw_centers(img, preds)
        if draw_grid:
            image_utils.draw_grid(img, net.grid_size)
        image_utils.draw_bbox(preds[0], img)
        t_draw = time.time() - t0_draw
        cv2.imshow(net.model_version, img)
        k = cv2.waitKey(delay)
        compute_time.append([t_read, t_resize, t_preds, t_draw])
        t_total = t_read + t_resize + t_preds + t_draw
        if print_time:
            print('Read time: %.3f, Resize time: %.3f, Prediction time: %.3f, Draw time: %.3f, Total time: %.3f' % (
                t_read, t_resize, t_preds, t_draw, t_total))
        if k == 27:
            break
    return compute_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net_cfg', action='store', type=str, help='Path to configuration file of the network')
    parser.add_argument('-demos_cfg', action='store', type=str, help='Path to configuration file for demos')
    args = parser.parse_args()
    if not os.path.isfile(args.net_cfg):
        parser.error('Path to configuration file of the network should point to existing file.')
    if not os.path.isfile(args.demos_cfg):
        parser.error('Path to configuration file for demos should point to existing file.')
    return args


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
    args = parse_args()
    with open(args.demos_cfg, 'r') as file:
        config = json.load(file)
    
    if config['configuration']['modes']['images']:
        show_images_with_boxes(args.net_cfg, config['images'], config['weights'], config['draw_centers'],
                               config['draw_grid'], config['delay'])

    if config['configuration']['modes']['stats']:
        test_model(args.net_cfg, config['weights'], config['images'], config['annotations'],
                   config['configuration']['iou_threshold'])

    if __name__ == '__main__':
        main()
