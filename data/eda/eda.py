import data_utils as du
import bbox_utils as bbu
import stats_utils as su
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import pandas as pd
import os


def summary(cfg):
    sets = {'Test set': cfg['test_set'], 'Validation set': cfg['validation_set'], 'Training set': cfg['train_set']}
    for set_name, dataset in sets.items():
        print('STATS FOR {}'.format(set_name))
        labels = du.list_dirs(dataset['labels'], '.txt')
        boxes_per_image(labels)
        print()


def boxes_per_image(labels):
    bb_per_img = []
    empty_images = []
    for l in labels:
        bboxes = du.get_boxes(l)
        if bboxes is None:
            bb_per_img.append(0)
            empty_images.append(l)
            continue
        bb_per_img.append(bboxes.shape[0])

    bb_per_img = np.array(bb_per_img)
    empty_images = np.array(empty_images)
    avg = np.average(bb_per_img)
    max_bxs = np.max(bb_per_img)
    total = np.sum(bb_per_img)
    print('Total number of images: {}'.format(bb_per_img.shape[0]))
    print('Check: %d = %d' % (len(labels), bb_per_img.shape[0]))
    print('Number of images that do not contain objects: %d' % empty_images.shape[0])
    print('Total number of boxes: %d' % total)
    print('Average number of boxes per image: %.2f' % avg)
    print('Max number of boxes per image: %d' % max_bxs)
    return bb_per_img


def bb_per_image_to_csv(labels, save_to):
    bb_per_img = boxes_per_image(labels)
    # bins = np.bincount(bb_per_img)
    data_dict = {'num of boxes': bb_per_img}
    df = pd.DataFrame(data=data_dict)
    df.to_csv(os.path.join(save_to, 'boxes_per_image.csv'))


def distance(imgs, labels, new_h, new_w):
    x_dist = []
    y_dist = []
    t_dist = []
    for img, label in zip(imgs, labels):
        bboxes = du.get_boxes(label)
        if bboxes is None:
            continue
        if bboxes.shape[0] < 2:
            continue
        w, h = Image.open(img).size
        h_new, w_new = su.compute_new_size(h, w, new_h, new_w)
        bboxes = bbu.resize_boxes(bboxes, (h, w), (h_new, w_new))
        distances = compute_distances(bboxes)
        for item in distances:
            x_dist.append(item[0])
            y_dist.append(item[1])
            t_dist.append(item[2])
    print('Sanity check. Items in x: %d, items in y: %d, items in t: %d' % (len(x_dist), len(y_dist), len(t_dist)))
    print('min x: %d, min y: %d, min total: %d' % (np.amin(x_dist), np.amin(y_dist), np.amin(t_dist)))
    print('max x: %d, max y: %d, max total: %d' % (np.max(x_dist), np.max(y_dist), np.max(t_dist)))
    print('avg x: %.2f, avg y: %.2f, avg total: %.2f' % (np.average(x_dist), np.average(y_dist), np.average(t_dist)))
    return x_dist, y_dist, t_dist


def save_distances_to_csv(imgs, labels, new_h, new_w, save_to):
    x, y, t = distance(imgs, labels, new_h, new_w)
    data_dict = {'x dist': x, 'y dist': y, 'total dist': t}
    df = pd.DataFrame(data=data_dict)
    df.to_csv(os.path.join(save_to, 'distances.csv'))


def compute_distances(bboxes):
    no_boxes = bboxes.shape[0]
    distances = []
    for i in range(no_boxes):
        for j in range(i + 1, no_boxes):
            x, y, t = su.compute_distance(bboxes[i], bboxes[j])
            distances.append([x, y, t])
    return distances


def boxes_dimensions(labels, imgs, new_h, new_w):
    boxes_w = []
    boxes_h = []
    for img, label in zip(imgs, labels):
        bboxes = du.get_boxes(label)
        if bboxes is None:
            continue
        if bboxes.shape[0] < 2:
            continue
        w, h = Image.open(img).size
        h_new, w_new = su.compute_new_size(h, w, new_h, new_w)
        bboxes = bbu.resize_boxes(bboxes, (h, w), (h_new, w_new))
        for item in bboxes:
            boxes_w.append(item[2])
            boxes_h.append(item[3])
    return boxes_w, boxes_h


def save_boxes_dims_to_csv(labels, imgs, new_h, new_w, save_to):
    w, h = boxes_dimensions(labels, imgs, new_h, new_w)
    data_dict = {'width': w, 'height': h}
    df = pd.DataFrame(data=data_dict)
    df.to_csv(os.path.join(save_to, 'boxes_dims.csv'))


def plot_dimensions(labels_path, img_path, new_h, new_w):
    boxes_w, boxes_h = boxes_dimensions(labels_path, img_path, new_h, new_w)

    hist, xedges, yedges = np.histogram2d(boxes_w, boxes_h, bins=50, normed=False)
    hist = hist.T
    hist = hist / len(boxes_w)
    mesh_x, mesh_y = np.meshgrid(xedges, yedges)
    fig_cmap, ax_cmap = plt.subplots(nrows=1, ncols=1)
    cmap = plt.get_cmap('Greys')

    img = ax_cmap.pcolormesh(mesh_x, mesh_y, hist, cmap=cmap)
    fig_cmap.colorbar(img, ax=ax_cmap)
    ax_cmap.set_xlim(xmin=0, xmax=80)
    ax_cmap.set_ylim(ymin=0, ymax=80)
    ax_cmap.xaxis.set_ticks(np.arange(0, ax_cmap.get_xlim()[1] + 5, 5))
    ax_cmap.yaxis.set_ticks(np.arange(0, ax_cmap.get_ylim()[1] + 5, 5))
    ax_cmap.set_title('Distribution of boxes by width and height', fontsize=10)
    ax_cmap.set_xlabel('Width [px]', fontsize=10)
    ax_cmap.set_ylabel('Height [px]', fontsize=10)
    fig_cmap.tight_layout()

    fig_dist, ax_dim = plt.subplots(nrows=2, ncols=1)

    ax_dim[0].plot(np.bincount(boxes_w) / np.sum(np.bincount(boxes_w)))
    ax_dim[0].grid()
    ax_dim[0].set_title('Distribution of boxes width', fontsize=10)
    ax_dim[0].set_xlabel('Width [px]', fontsize=10)
    ax_dim[0].set_ylabel('Frequency', fontsize=10)

    ax_dim[1].plot(np.bincount(boxes_h) / np.sum(np.bincount(boxes_h)))
    ax_dim[1].grid()
    ax_dim[1].set_title('Distribution of boxes height', fontsize=10)
    ax_dim[1].set_xlabel('Height [px]', fontsize=10)
    ax_dim[1].set_ylabel('Frequency', fontsize=10)

    for ax in ax_dim:
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
    fig_dist.tight_layout()


def plot_bxs_per_img(labels_path):
    # statistics on number of boxes per image
    bb_per_img = boxes_per_image(labels_path)
    bins = np.bincount(bb_per_img) / np.sum(np.bincount(bb_per_img))
    # bins = np.bincount(bb_per_img)
    print("Check: %f" % np.sum(bins))
    fig_no_boxes = plt.figure(figsize=(7, 3))
    fig_no_boxes.suptitle('Number of boxes per image')
    ax_1 = fig_no_boxes.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_1.set_xlabel('Number of boxes')
    ax_1.set_ylabel('Frequency')
    ax_1.grid()
    ax_1.plot(bins, lw=2, marker='x')
    ax_1.set_ylim(ymin=0.0)
    ax_1.set_xlim(xmin=0.0)
    ax_1.xaxis.set_ticks(np.arange(0, ax_1.get_xlim()[1], 1))


def plot_distances(labels_path, img_path, new_h, new_w):
    # statistics on distances between centres
    x, y, t = distance(img_path, labels_path, new_h, new_w)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(140, 90), normed=False)
    hist = hist.T
    hist = hist / len(x)
    mesh_x, mesh_y = np.meshgrid(xedges, yedges)

    fig_dist, ax_dist = plt.subplots(nrows=3, ncols=1)

    ax_dist[0].plot(np.bincount(t) / np.sum(np.bincount(t)), lw=2)
    ax_dist[0].set_title('Distribution of distances between centres of objects', fontsize=10)
    ax_dist[0].set_xlabel('Distance [px]', fontsize=10)
    ax_dist[0].set_ylabel('Frequency', fontsize=10)
    ax_dist[0].grid()

    ax_dist[1].plot(np.bincount(x) / np.sum(np.bincount(x)), lw=2)
    ax_dist[1].grid()
    ax_dist[1].set_title('Distribution of x component of distance', fontsize=10)
    ax_dist[1].set_xlabel('x component of distance [px]', fontsize=10)
    ax_dist[1].set_ylabel('Frequency', fontsize=10)

    ax_dist[2].plot(np.bincount(y) / np.sum(np.bincount(y)), lw=2)
    ax_dist[2].grid()
    ax_dist[2].set_title('Distribution of y component of distance', fontsize=10)
    ax_dist[2].set_xlabel('y component of distance [px]', fontsize=10)
    ax_dist[2].set_ylabel('Frequency', fontsize=10)

    for ax in ax_dist:
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)

    fig_dist.tight_layout()

    fig_joint_dist, ax_joint = plt.subplots(nrows=2, ncols=1)
    cmap = plt.get_cmap('YlGnBu')
    # cmap = plt.get_cmap('Greys')

    img = ax_joint[0].pcolormesh(mesh_x, mesh_y, hist, cmap=cmap)
    # ax_joint[0].set_xlim(xmin=0, xmax=150)
    # ax_joint[0].set_ylim(ymin=0, ymax=150)
    ax_joint[0].set_title('Distribution of distances by x and y components', fontsize=10)
    ax_joint[0].set_xlabel('x component [px]', fontsize=10)
    ax_joint[0].set_ylabel('y component [px]', fontsize=10)

    ax_joint[1].pcolormesh(mesh_x, mesh_y, hist, cmap=cmap)
    ax_joint[1].set_xlim(xmin=0, xmax=50)
    ax_joint[1].set_ylim(ymin=0, ymax=50)
    ax_joint[1].xaxis.set_ticks(np.arange(0, ax_joint[1].get_xlim()[1] + 5, 5))
    ax_joint[1].set_title('Distribution of distances by x and y components', fontsize=10)
    ax_joint[1].set_xlabel('x component [px]', fontsize=10)
    ax_joint[1].set_ylabel('y component [px]', fontsize=10)
    fig_joint_dist.colorbar(img, ax=ax_joint[0])
    fig_joint_dist.colorbar(img, ax=ax_joint[1])
    fig_joint_dist.tight_layout()
    # fig_distances.suptitle('Distances between centres of objects')


def plot(cfg):
    sets = (cfg['test_set'], cfg['validation_set'], cfg['train_set'])
    new_h = cfg['new_h']
    new_w = cfg['new_w']
    for dataset in sets:
        labels = du.list_dirs(dataset['labels'], '.txt')
        imgs = du.list_dirs(dataset['images'], '.jpg')
        imgs, labels = du.match_imgs_with_labels(imgs, labels)
        plot_bxs_per_img(labels)
        plot_dimensions(labels, imgs, new_h, new_w)
        plot_distances(labels, imgs, new_h, new_w)
    plt.show()


def save_csv(cfg):
    sets = {'Test set': cfg['test_set'], 'Validation set': cfg['validation_set'], 'Training set': cfg['train_set']}
    new_h = cfg['new_h']
    new_w = cfg['new_w']
    for set_name, dataset in sets.items():
        labels = du.list_dirs(dataset['labels'], '.txt')
        imgs = du.list_dirs(dataset['images'], '.jpg')
        imgs, labels = du.match_imgs_with_labels(imgs, labels)
        save_to = os.path.join(cfg['save_to'], set_name)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        bb_per_image_to_csv(labels, save_to)
        save_distances_to_csv(imgs, labels, new_h, new_w, save_to)
        save_boxes_dims_to_csv(labels, imgs, new_h, new_w, save_to)


def main():
    cfg_path = 'eda_cfg.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    mode = cfg['mode']
    if mode == 'plot':
        plot(cfg)
    elif mode == 'save_csv':
        save_csv(cfg)
    elif mode == 'summary':
        summary(cfg)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
