# https://docs.python.org/3/library/unittest.html
from network.yolo_v0 import YoloV0
from data.dataset_generator import DatasetGenerator
import tensorflow as tf
import numpy as np


def convert(labels, grid_size, img_size, o_img_size):
    # print(labels[0, :, 0])
    w_ratio = o_img_size[0] / img_size[0]
    h_ratio = o_img_size[1] / img_size[1]
    Gi = np.zeros([grid_size[0] * grid_size[1]])
    Gj = np.zeros([grid_size[0] * grid_size[1]])
    counter_i = 0
    counter_j = 0
    for i in range(grid_size[0] * grid_size[1]):
        Gi[i] = counter_i
        Gj[i] = counter_j
        counter_i += 1
        if (i + 1) % grid_size[0] == 0:
            counter_i = 0
            counter_j += 1
    # print(Gi)
    print(img_size[0] / grid_size[0])
    labels[:, :, 0] = (labels[:, :, 0] + Gi) * img_size[0] / grid_size[0] * w_ratio
    labels[:, :, 1] = (labels[:, :, 1] + Gj) * img_size[1] / grid_size[1] * h_ratio
    labels[:, :, 2] = np.power(labels[:, :, 2] * img_size[0], 2) * w_ratio
    labels[:, :, 3] = np.power(labels[:, :, 3] * img_size[1], 2) * h_ratio
    for row in labels[0, :, :]:
        print(row)


def covert_to_centre(boxes):
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] = boxes[:, 0] + np.round(np.divide(boxes[:, 2], 2))
    boxes[:, 1] = boxes[:, 1] + np.round(np.divide(boxes[:, 3], 2))

    return boxes


def test_tf_iou():
    img_size = (720, 480)
    orgn_size = (2160, 3840)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size)
    # data for first test. all ious should be 1
    print('Prepare first set')
    dataset = DatasetGenerator('/Volumes/TRANSCEND/Data Sets/another_testset/tf_iou_test/imgs',
                               '/Volumes/TRANSCEND/Data Sets/another_testset/tf_iou_test/labels',
                               img_size, grid_size, 1, False)
    batch = dataset.get_minibatch(5)
    _, labels = next(batch)
    labels = np.reshape(labels, [-1, grid_size[0] * grid_size[1], 5])
    # data for second test. For some boxes it should be 0, since truth and pred are in different boxes
    print('Prepare second set')
    boxes_truth = np.array([[2518, 713, 39, 101], [1395, 1484, 126, 118], [1060, 896, 87, 79],
                            [1055, 856, 82, 73], [1496, 632, 59, 106], [2013, 70, 31, 68],
                            [2115, 51, 36, 65], [2297, 53, 37, 62], [2016, 177, 31, 85]])
    boxes_pred = np.array([[2630, 680, 45, 111], [1352, 1400, 131, 100], [801, 890, 115, 79],
                           [1045, 860, 41, 36], [1496, 632, 118, 212], [2000, 56, 31, 68],
                           [2100, 66, 72, 32], [2311, 67, 37, 62], [2016, 169, 15, 15]])
    bbox_t = dataset.resize_and_adjust_labels(orgn_size, boxes_truth)
    bbox_t = np.reshape(bbox_t, [-1, grid_size[0] * grid_size[1], 5])
    bbox_p = dataset.resize_and_adjust_labels(orgn_size, boxes_pred)
    bbox_p = np.reshape(bbox_p, [-1, grid_size[0] * grid_size[1], 5])
    # data for third test. IoUs should mostly be greater that 0.5
    print('Prepare third set')
    orgn_size = (540, 960)
    bbox2_truth = np.array(
        [[348, 301, 63, 80], [439, 217, 45, 64], [441, 336, 47, 67], [500, 262, 36, 61], [603, 257, 39, 61]],
        dtype=np.float64)
    bbox2_pred = np.array(
        [[347, 304, 68, 75], [431, 223, 49, 60], [443, 330, 45, 70], [515, 270, 30, 66], [601, 256, 37, 64]],
        dtype=np.float64)
    bbox2_t = dataset.resize_and_adjust_labels(orgn_size, bbox2_truth)
    bbox2_t = np.reshape(bbox2_t, [-1, grid_size[0] * grid_size[1], 5])
    bbox2_p = dataset.resize_and_adjust_labels(orgn_size, bbox2_pred)
    bbox2_p = np.reshape(bbox2_p, [-1, grid_size[0] * grid_size[1], 5])

    # prepare IOU
    y_pred = tf.placeholder(tf.float32, [None, grid_size[0] * grid_size[1], 5])
    y_true = tf.placeholder(tf.float32, [None, grid_size[0] * grid_size[1], 5])
    ious = net.tf_iou(y_true, y_pred)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res1 = sess.run(ious, feed_dict={y_true: labels, y_pred: labels})
        res2 = sess.run(ious, feed_dict={y_true: bbox_t, y_pred: bbox_p})
        res3 = sess.run(ious, feed_dict={y_true: bbox2_t, y_pred: bbox2_p})
        print('Printing first results')
        for row in res1[1, :]:
            print(row, end=' ')
        print('')
        print('Printing second results')
        for i, element in enumerate(res2[0, :]):
            if element != 1:
                print('%f[%d]' % (element, i), end=' ')
        print('')
        print('Printing third results')
        for j, e, in enumerate(res3[0, :]):
            if e != 1:
                print('%f[%d]' % (e, j), end=' ')
    del net


def test_nms():
    boxes_pred = np.array([
        [[2630, 680, 45, 111, 0.78], [2640, 690, 40, 101, 0.7], [2623, 682, 42, 109, 0.71], [2651, 672, 45, 111, 0.6],

         [1352, 1400, 131, 100, 0.79], [1342, 1395, 127, 100, 0.71], [1330, 1420, 141, 110, 0.68],
         [1329, 1390, 120, 95, 0.6]],

        [[801, 890, 115, 79, 0.9], [1340, 1400, 115, 79, 0.2], [809, 895, 107, 74, 0.6], [815, 894, 115, 79, 0.5],
         [1045, 860, 41, 36, 0.7], [1040, 865, 47, 39, 0.1], [1045, 860, 41, 36, 0.6], [1045, 860, 41, 36, 0.56]]])
    img_size = (720, 480)
    orgn_size = (2160, 3840)
    grid_size = (36, 24)
    print(np.shape(boxes_pred))
    net = YoloV0(grid_size, img_size)
    nms = net.nms(boxes_pred)
    print(nms)
    for batch_n in range(len(boxes_pred)):
        nms[batch_n] = covert_to_centre(nms[batch_n])
        print(nms[batch_n])


def argsort():
    boxes = np.array([[[2518, 713, 39, 101, 0.51], [1395, 1484, 126, 118, 0.72]],
                      [[1060, 896, 87, 79, 0.31], [1055, 856, 82, 73, 0.56]],
                      [[1496, 632, 59, 106, 0.76], [2013, 70, 31, 68, 0.87]],
                      [[2115, 51, 36, 65, 0.9], [2297, 53, 37, 62, 0.55]],
                      [[2016, 177, 31, 85, 0.78], [2297, 53, 37, 62, 0.31]]])
    print(np.shape(boxes))
    indxs = np.argsort(boxes[:, :, 4])
    print(len(indxs))


def argmax_test():
    boxes_pred = np.array([
        [2630, 680, 45, 111, 0.78], [2640, 690, 40, 101, 0.7], [2623, 682, 42, 109, 0.71], [2651, 672, 45, 111, 0.6],

        [1352, 1400, 131, 100, 0.79], [1342, 1395, 127, 100, 0.71], [1330, 1420, 141, 110, 0.68],
        [1329, 1390, 120, 95, 0.6],

        [801, 890, 115, 79, 0.9], [1340, 1400, 115, 79, 0.2], [809, 895, 107, 74, 0.6], [815, 894, 115, 79, 0.5],
        [1045, 860, 41, 36, 0.7], [1040, 865, 47, 39, 0.1], [1045, 860, 41, 36, 0.6], [1045, 860, 41, 36, 0.56]])
    maxes = np.argmax(boxes_pred, axis=0)
    print(maxes)
    # print(boxes_pred)
    print([boxes_pred[m, i] for i, m in enumerate(maxes)])
    maxes1 = np.argmax(boxes_pred, axis=1)
    print(maxes1)
    # print(boxes_pred)
    print([boxes_pred[i, m] for i, m in enumerate(maxes1)])
    b = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    print(np.argmax(b, axis=0))
    print(np.argmax(b, axis=1))


def test_compute_stats():
    # check basic functioning
    boxes_true = [np.array([[2630, 680, 45, 111, 1], [1352, 1400, 131, 100, 1]]),
                  np.array([[441, 336, 47, 67, 1]])]
    boxes_pred = [np.array([[2630, 680, 45, 111, 0.9], [1352, 1400, 131, 100, 0.9]]),
                  np.array([[441, 336, 47, 67, 0.9]])]
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size)
    b_true = net.convert_coords(boxes_true, nump=False)
    b_pred = net.convert_coords(boxes_pred, nump=False)
    stats = net.compute_stats(b_pred, b_true)
    print('First test results:')
    print(stats)
    # second test
    boxes_true = [np.array([[2630, 680, 45, 111, 1], [1352, 1400, 131, 100, 1]]),
                  np.array([[441, 336, 47, 67, 1], [2115, 51, 36, 65, 1]])]
    boxes_pred = [np.array([[2630, 680, 45, 111, 0.9], [1352, 1400, 131, 100, 0.9], [2115, 51, 36, 65, 0.3]]),
                  np.array([[441, 336, 47, 67, 0.9]])]
    b_true = net.convert_coords(boxes_true, nump=False)
    b_pred = net.convert_coords(boxes_pred, nump=False)
    stats = net.compute_stats(b_pred, b_true)
    print('Second test results:')
    print(stats)
    # third test
    boxes_true = [np.array([[2518, 713, 39, 101, 1], [1395, 1484, 126, 118, 1], [1060, 896, 87, 79, 1],
                            [1055, 856, 82, 73, 1], [1040, 890, 104, 81, 1]]),
                  np.array([[1496, 632, 59, 106, 1], [2013, 70, 31, 68, 1],
                            [2115, 51, 36, 65, 1], [2297, 53, 37, 62, 1], [2016, 177, 31, 85, 1]])]
    boxes_pred = [np.array([[2525, 707, 43, 108, 0.64], [1401, 1479, 131, 109, 0.82], [1040, 890, 104, 81, 0.57],
                            [1049, 860, 70, 56, 0.65], [2510, 707, 43, 108, 0.61], [1043, 887, 95, 81, 0.56],
                            [2518, 713, 39, 101, 1]]),
                  np.array([[1496, 632, 65, 110, 0.87], [2003, 64, 33, 64, 0.46],
                            [2105, 66, 43, 55, 0.40], [2307, 67, 37, 62, 0.4], [2016, 169, 27, 79, 0.72]])]
    b_true = net.convert_coords(boxes_true, nump=False)
    b_pred = net.convert_coords(boxes_pred, nump=False)
    stats = net.compute_stats(b_pred, b_true)
    tmp = np.sum(stats, axis=0)
    no_tp = tmp[0]
    avg_prec = tmp[1] / len(stats)
    avg_recall = tmp[2] / len(stats)
    avg_conf = tmp[3] / len(stats)
    avg_iou = tmp[4] / len(stats)
    print('Third test results:')
    print(stats)
    print([no_tp, avg_prec, avg_recall, avg_conf, avg_iou])


def test_io():
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloV0(grid_size, img_size)
    dataset = DatasetGenerator('/Volumes/TRANSCEND/Data Sets/another_testset/tf_iou_test/imgs',
                               '/Volumes/TRANSCEND/Data Sets/another_testset/tf_iou_test/labels',
                               img_size, grid_size, 1, False)
    net.open_sess()
    batch = dataset.get_minibatch(10)
    imgs, labels = next(batch)
    preds = net.get_predictions(imgs)
    preds = net.predictions_to_boxes(preds)
    preds = net.nms(preds)

    print(preds)
    truth = net.predictions_to_boxes(labels)
    truth = net.convert_coords(truth)
    print(truth)
    true_boxes = []
    for t_boxes in truth:
        true_boxes.append(np.delete(t_boxes, np.where(t_boxes[:, 4] != 1.0), axis=0))
    print(true_boxes)
    stats = net.compute_stats(preds, true_boxes)
    print(stats)
    print(np.shape(stats))
    net.close_sess()


if __name__ == '__main__':
    test_io()