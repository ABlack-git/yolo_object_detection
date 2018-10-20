import numpy as np
import bbox_utils as bbu
import pandas as pd
import os


def compute_stats(pred_boxes, true_boxes, iou_threshold, stats=None):
    """

    :param pred_boxes: list of np.arrays
    :param true_boxes: list of np.arrays
    :param iou_threshold:
    :param stats
    :return:
    """
    if stats is None:
        stats = []

    for i, i_boxes in enumerate(pred_boxes):
        no_fp = 0
        no_fn = 0
        no_tp = 0
        iou_sum = 0
        conf_tp_sum = 0
        conf_fp_sum = 0
        # check if preds and trues are not empty
        if i_boxes.size == 0 and true_boxes[i].size == 0:
            stats.append([1, 1, iou_sum, conf_tp_sum, conf_fp_sum, no_tp, no_fp, no_fn])
            continue

        if i_boxes.size == 0 and true_boxes[i].size != 0:
            no_fn = len(true_boxes[i])
            stats.append([0, 0, iou_sum, conf_tp_sum, conf_fp_sum, no_tp, no_fp, no_fn])
            continue

        if i_boxes.size != 0 and true_boxes[i].size == 0:
            no_fp = i_boxes.size
            conf_fp_sum += np.sum(i_boxes[:, 4])
            stats.append([0, 0, 0, conf_tp_sum, conf_fp_sum, no_tp, no_fp, no_fn])
            continue

        # compute ious
        ious = bbu.iou(np.tile(i_boxes, [len(true_boxes[i]), 1]), np.repeat(true_boxes[i], len(i_boxes), axis=0))
        # create matrix of ious where columns are predicted boxes and rows are true boxes
        ious = np.reshape(ious, (len(true_boxes[i]), len(i_boxes)))

        while ious.size != 0:
            max_ind = np.unravel_index(np.argmax(ious), ious.shape)
            max_iou = ious[max_ind]
            if max_iou >= iou_threshold:
                # assign predictor to ground truth
                no_tp += 1
                iou_sum += max_iou
                conf_tp_sum += i_boxes[max_ind[1], 4]
                # delete rows and cols
                ious = np.delete(ious, max_ind[1], axis=1)
                ious = np.delete(ious, max_ind[0], axis=0)
                if ious.size == 0:
                    break
            # check for false positives(background classified as person => low iou values for column)
            fp_ind = np.where(np.amax(ious, axis=0) < iou_threshold)[0]
            no_fp += fp_ind.size
            conf_fp_sum += np.sum(i_boxes[fp_ind, 4])
            # check for false negatives(not detected person => low iou values for entire row)
            fn_ind = np.where(np.amax(ious, axis=1) < iou_threshold)[0]
            no_fn += fn_ind.size
            # delete rows and columns with false predictions
            # deletion along axis=0 is deletion of rows, axis=1 deletion of column
            ious = np.delete(ious, fp_ind, axis=1)
            if ious.size == 0:
                break
            ious = np.delete(ious, fn_ind, axis=0)

        precision = no_tp / (no_tp + no_fp) if (no_tp + no_fp) > 0 else 1
        recall = no_tp / (no_tp + no_fn) if (no_tp + no_fn) > 0 else 1
        # avg_iou=(iou_sum / no_tp) if no_tp > 0 else None
        # conf_tp_avg = (conf_tp_sum / no_tp) if no_tp > 0 else None
        # conf_fp_avg = (conf_fp_sum / no_fp) if no_fp > 0 else None
        stats.append([precision, recall, iou_sum, conf_tp_sum, conf_fp_sum, no_tp, no_fp, no_fn])

    return stats


def process_stats(stats):
    if len(stats) <= 1:
        return stats
    stats = np.array(stats)
    precisions = stats[:, 0]
    recalls = stats[:, 1]
    avg_prec = np.sum(precisions) / precisions.size if precisions.size > 0 else 1
    avg_recall = np.sum(recalls) / recalls.size if recalls.size > 0 else 1
    sum_tp = np.sum(stats[:, 5])
    avg_iou = np.sum(stats[:, 2]) / sum_tp if sum_tp > 0 else None
    avg_conf_tp = np.sum(stats[:, 3]) / sum_tp if sum_tp > 0 else None
    sum_fp = np.sum(stats[:, 6])
    avg_conf_fp = np.sum(stats[:, 4]) / sum_fp if sum_fp > 0 else None
    sum_fn = np.sum(stats[:, 7])
    return [avg_prec, avg_recall, avg_iou, avg_conf_tp, avg_conf_fp, sum_tp, sum_fp, sum_fn]


def save_stats(stats, path, model_name):
    header = ['Precision', 'Recall', 'AVG_IOU', 'AVG_CONF_TP', 'AVG_CONF_FP', 'Number of TP', 'Number of FP',
              'Number of FN']
    df = pd.DataFrame(stats, columns=header)
    df.to_csv(os.path.join(path, 'stats_{0}.csv'.format(model_name)))


def compute_distance(bb_a, bb_b):
    """
    :param bb_a:
    :param bb_b:
    :return: euclidean distance between two boxes and its x and y component
    """
    x_dist = np.power(bb_a[0] - bb_b[0], 2)
    y_dist = np.power(bb_a[1] - bb_b[1], 2)
    t_dist = np.sqrt(x_dist + y_dist)
    return int(np.sqrt(x_dist)), int(np.sqrt(y_dist)), int(t_dist)


def progress_bar(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '*' * filled_len + '-' * (bar_len - filled_len)
    print('\r[%s] %s' % (bar, percents), end='\r')
