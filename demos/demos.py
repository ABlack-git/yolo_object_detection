from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import cv2
import image_utils
import stats_utils as su
import os
import time
import argparse
import json
import bbox_utils as bbu


def test_model(net, images, labels, iou_threshold, save_stats, path=''):
    # construct json
    img_size = {"width": net.img_size[0], "height": net.img_size[1]}
    grid_size = {"width": net.grid_size[0], "height": net.grid_size[1]}
    params = {"img_size": img_size, "grid_size": grid_size, "no_boxes": net.no_boxes, "shuffle": True, "sqrt": net.sqrt,
              'keep_asp_ratio': net.keep_asp_ratio}
    conf = {"images": images, "annotations": labels, "configuration": params}
    conf = json.dumps(conf)

    ts = DatasetGenerator(conf)
    batch = ts.get_minibatch(net.batch_size, resize_only=True)
    stats = []
    num_batches = ts.get_number_of_batches(net.batch_size)
    for i in range(num_batches):
        # labels are only resized and returned in center based coords
        imgs, labels = next(batch)
        true_boxes = []
        for img_labesl in labels:
            true_boxes.append(bbu.convert_center_to_2points(img_labesl))
        preds = net.get_predictions(imgs)
        # compute stats for batch
        stats = su.compute_stats(preds, true_boxes, iou_threshold, stats)
        su.progress_bar(i, num_batches)
    final_stats = su.process_stats(stats)
    stats.append(final_stats)
    if save_stats:
        su.save_stats(stats, path, net.model_version)
    print('Average precision: {0[0]}, Average recall: {0[1]}, Average iou: {0[2]}, Average confidence of TP: {0[3]}, '
          'Average confidence of FP: {0[4]}, Total num of TP: {0[5]}, Total num of FP: {0[6]}, '
          'Total num of FN: {0[7]}'.format(final_stats))


def show_images_with_boxes(net, testing_set, draw_centre=True, draw_grid=False, delay=0,
                           print_time=True):
    list_of_imgs = image_utils.list_of_images(testing_set)
    compute_time = []
    for img_path in list_of_imgs:
        t0_read = time.time()
        if isinstance(testing_set, list):
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(os.path.join(testing_set, img_path))
        t_read = time.time() - t0_read
        t0_resize = time.time()
        img = image_utils.resize_img(img, net.img_size[1], net.img_size[0], keep_asp_ratio=net.keep_asp_ratio)

        # if net.keep_asp_ratio: pad

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


def main():
    args = parse_args()
    with open(args.demos_cfg, 'r') as file:
        config = json.load(file)

    net = YoloV0(args.net_cfg)
    net.restore(path=config['weights'])
    modes = config['configuration']['modes']
    if modes['images']:
        show_images_with_boxes(net, config['images'], config['configuration']['draw_centers'],
                               config['configuration']['draw_grid'], config['configuration']['delay'],
                               config['configuration']['print_time'])

    if modes['stats']['enable']:
        test_model(net, config['images'], config['annotations'],
                   config['configuration']['iou_threshold'], modes['stats']['save'], modes['stats']['path'])
    net.close_sess()


if __name__ == '__main__':
    main()
