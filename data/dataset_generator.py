import os
import random
import cv2
import numpy as np


class DatasetGenerator:

    def __init__(self, img_dir, labels_dir, img_size, grid_size, no_boxes, shuffle=True, sqrt=True):
        self.img_dir = img_dir
        self.shuffle = shuffle
        self.labels_dir = labels_dir

        self.data_labels = self.get_labels()
        self.data_imgs = self.get_imgs()
        self.image_w = img_size[0]
        self.image_h = img_size[1]
        self.grid_n = grid_size[0]
        self.grid_m = grid_size[1]
        self.no_boxes = no_boxes
        self.grid_w = self.image_w / self.grid_n
        self.grid_h = self.image_h / self.grid_m
        self.sqrt = sqrt

    def get_imgs(self):
        imgs = []
        for label in self.data_labels:
            imgs.append(label.replace('.txt', '.jpg'))
        return imgs

    def get_labels(self):
        labels = []
        for file in os.listdir(self.labels_dir):
            if file.endswith('.txt') and not file.startswith('.'):
                labels.append(file)
        if self.shuffle:
            random.shuffle(labels)
        return labels

    def reshuffle(self):
        ziped = list(zip(self.data_imgs, self.data_labels))
        random.shuffle(ziped)
        self.data_imgs, self.data_labels = zip(*ziped)

    def get_boxes(self, txt_name):
        with open(txt_name, 'r') as file:
            coords = file.read().splitlines()
        if coords[0] == 'None':
            return None

        for i, line in enumerate(coords):
            tmp = line.split()
            coords[i] = [int(x) for x in tmp]
        return np.array(coords)

    def resize_img(self, img):
        height, width = img.shape[:2]
        if height > self.image_h or width > self.image_w:
            img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
        return img

    def resize_and_adjust_labels(self, orgn_size, boxes):
        label = np.zeros(5 * self.no_boxes * self.grid_m * self.grid_n)
        if boxes is None:
            return label

        w_ratio = self.image_w / orgn_size[1]
        h_ratio = self.image_h / orgn_size[0]

        if w_ratio != 1 and h_ratio != 1:
            boxes[:, 0] = np.round(boxes[:, 0] * w_ratio)
            boxes[:, 1] = np.round(boxes[:, 1] * h_ratio)
            boxes[:, 2] = np.round(boxes[:, 2] * w_ratio)
            boxes[:, 3] = np.round(boxes[:, 3] * h_ratio)

        for ind, box in enumerate(boxes):
            # calculate coordinates of cell
            if box[0] >= self.image_w:
                box[0] = self.image_w - 1
            if box[1] >= self.image_h:
                box[1] = self.image_h - 1
            i = np.floor(box[0] / self.grid_w)
            j = np.floor(box[1] / self.grid_h)
            # calculate index of cell in label vector
            k = int(5 * self.no_boxes * ((j + 1) * self.grid_n - (self.grid_n - i)))
            for box_n in range(self.no_boxes):
                try:
                    label[k + 5 * box_n] = box[0] / self.grid_w - i
                    label[k + 1 + 5 * box_n] = box[1] / self.grid_h - j
                    if self.sqrt:
                        label[k + 2 + 5 * box_n] = np.sqrt(box[2]) / self.image_w
                        label[k + 3 + 5 * box_n] = np.sqrt(box[3]) / self.image_h
                    else:
                        label[k + 2 + 5 * box_n] = box[2] / self.image_w
                        label[k + 3 + 5 * box_n] = box[3] / self.image_h
                    label[k + 4 + 5 * box_n] = 1
                except IndexError:
                    print('k: %d, i: %.1f, j: %.1f, box_w: %f, box_h: %f, labels_size %s' % (
                        k, i, j, box[0], box[1], str(np.shape(label))))
                    raise IndexError
        return label

    def get_number_of_batches(self, batch_size):
        return int(np.floor(len(self.data_labels) / batch_size))

    def get_minibatch(self, batch_size):
        self.reshuffle()
        images = []
        labels = []
        empty = False
        counter = 0
        batch_counter = 0
        while True:
            # OpenCV returns image as (height,width,channels), where channels are BGR.
            img = cv2.imread(os.path.join(self.img_dir, self.data_imgs[counter]), cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            boxes = self.get_boxes(os.path.join(self.labels_dir, self.data_labels[counter]))
            # resize img
            img = self.resize_img(img)
            # resize and adujst labels
            try:
                boxes = self.resize_and_adjust_labels((height, width), boxes)
            except IndexError:
                print('In %s' % (self.data_imgs[counter]))
            images.append(img)
            labels.append(boxes)
            counter += 1

            if len(self.data_labels) == counter:
                # print('No more items in data set')
                empty = True

            if counter % batch_size == 0:
                batch_counter += 1
                # print('Counter is %d' % counter)
                # print('Batch number %d has been loaded' % batch_counter)
                # yield images, labels
                # yield np.array(images, dtype=np.int), np.array(labels, dtype=np.float32)
                yield images, np.array(labels, dtype=np.float32)
                del images
                del labels
                images = []
                labels = []

                if empty:
                    break
