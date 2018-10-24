import os
import random
import cv2
import numpy as np
import json


class DatasetGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.img_dir = None
        self.labels_dir = None
        self.image_w = None
        self.image_h = None
        self.grid_w = None
        self.grid_h = None
        self.no_boxes = None
        self.sqrt = False
        self.shuffle = False
        self.subset_length = -1
        self.data_imgs = []
        self.data_labels = []
        self.read_config()
        self.__get_data()
        self.grid_pw = self.image_w / self.grid_w
        self.grid_ph = self.image_h / self.grid_h

    def read_config(self):
        cfg = None
        if os.path.isfile(self.cfg):
            with open(self.cfg, 'r') as json_file:
                cfg = json.load(json_file)
        else:
            try:
                cfg = json.loads(self.cfg)
            except ValueError as e:
                print('Inappropriate format of json file ' + str(e) + '. In dataset generator.')
                exit(1)

        self.img_dir = cfg['images']
        self.labels_dir = cfg['annotations']
        config = cfg['configuration']
        self.image_w = config['img_size']['width']
        self.image_h = config['img_size']['height']
        self.grid_w = config['grid_size']['width']
        self.grid_h = config['grid_size']['height']
        self.no_boxes = config['no_boxes']
        self.shuffle = config['shuffle']
        self.sqrt = config['sqrt']
        if "subset_length" in config:
            self.subset_length = config['subset_length']

    def __get_data(self):
        for img_d, lbl_d in zip(self.img_dir, self.labels_dir):
            self.data_imgs += [os.path.join(img_d, f) for f in os.listdir(img_d) if
                               f.endswith('.jpg') and not f.startswith('.')]
            self.data_labels += [os.path.join(lbl_d, f) for f in os.listdir(lbl_d) if
                                 f.endswith('.txt') and not f.startswith('.')]

        self.data_labels.sort()
        self.data_imgs.sort()
        # shuffle dataset
        self.reshuffle()
        # check if img correspond to label
        for a, b in zip(self.data_imgs, self.data_labels):
            if os.path.basename(a).replace('.jpg', '') != os.path.basename(b).replace('.txt', ''):
                print('Image has wrong annotation. Annotation file should have same name as image file.')
                print(a + ' ' + b)
                exit(1)
        if self.subset_length > 0:
            tmp = list(zip(self.data_imgs, self.data_labels))
            self.data_imgs, self.data_labels = zip(*random.sample(tmp, self.subset_length))

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

    def __resize_and_adjust_labels(self, orgn_size, boxes, resize_only=False):
        label = np.zeros(5 * self.no_boxes * self.grid_h * self.grid_w)
        if boxes is None:
            return label

        w_ratio = self.image_w / orgn_size[1]
        h_ratio = self.image_h / orgn_size[0]

        if w_ratio != 1 and h_ratio != 1:
            boxes[:, 0] = np.round(boxes[:, 0] * w_ratio)
            boxes[:, 1] = np.round(boxes[:, 1] * h_ratio)
            boxes[:, 2] = np.round(boxes[:, 2] * w_ratio)
            boxes[:, 3] = np.round(boxes[:, 3] * h_ratio)

        if resize_only:
            return boxes

        for ind, box in enumerate(boxes):
            # calculate coordinates of cell
            if box[0] >= self.image_w:
                box[0] = self.image_w - 1
            if box[1] >= self.image_h:
                box[1] = self.image_h - 1
            i = np.floor(box[0] / self.grid_pw)
            j = np.floor(box[1] / self.grid_ph)
            # calculate index of cell in label vector
            k = int(5 * self.no_boxes * ((j + 1) * self.grid_w - (self.grid_w - i)))
            for box_n in range(self.no_boxes):
                label[k + 5 * box_n] = box[0] / self.grid_pw - i
                label[k + 1 + 5 * box_n] = box[1] / self.grid_ph - j
                if self.sqrt:
                    label[k + 2 + 5 * box_n] = np.sqrt(box[2] / self.image_w)
                    label[k + 3 + 5 * box_n] = np.sqrt(box[3] / self.image_h)
                else:
                    label[k + 2 + 5 * box_n] = box[2] / self.image_w
                    label[k + 3 + 5 * box_n] = box[3] / self.image_h
                label[k + 4 + 5 * box_n] = 1

        return label

    def get_number_of_batches(self, batch_size):
        return int(np.floor(len(self.data_labels) / batch_size))

    def get_minibatch(self, batch_size, resize_only=False):
        self.reshuffle()
        images = []
        labels = []
        empty = False
        counter = 0
        while True:
            # OpenCV returns image as (height,width,channels), where channels are BGR.
            img = cv2.imread(self.data_imgs[counter], cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            boxes = self.get_boxes(self.data_labels[counter])
            # resize img
            img = self.resize_img(img)
            # resize and adujst labels
            boxes = self.__resize_and_adjust_labels((height, width), boxes, resize_only)
            images.append(img)
            labels.append(boxes)
            counter += 1

            if len(self.data_labels) == counter:
                empty = True

            if counter % batch_size == 0:
                if resize_only:
                    yield images, labels
                else:
                    yield images, np.array(labels, dtype=np.float32)
                del images
                del labels
                images = []
                labels = []

                if empty:
                    break

    def get_dataset_size(self):
        return len(self.data_imgs)
