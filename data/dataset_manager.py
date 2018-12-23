import os
import json
import shutil
import random
import numpy as np
import cv2

import data_utils as du
import stats_utils as su
import bbox_utils as bbu
import image_utils as imu

from data.label_creator import LabelCreator
from data.data_aug.data_aug import RandomHSV, RandomHorizontalFlip, RandomScale, Sequence, RandomTranslate


class DSManager:
    LABELS_TYPE = ('my_ds', 'txt_virat', 'txt_okutama', 'xgtf_mdrone', 'xgtf_UFC1', 'xgtf_UFC2', 'txt_vdrone',
                   'json_coco')
    DATA_EXTENSIONS = ('.MOV', '.mov', '.MP4', '.mp4', '.mpg', '.jpg')
    LABELS_EXTENSIONS = ('.txt', '.xgtf', '.json')
    EXTENSION_MAP = {'my_ds': '.txt', 'txt_virat': '.txt', 'txt_okutama': '.txt', 'xgtf_mdrone': '.xgtf',
                     'xgtf_UFC1': '.xgtf', 'xgtf_UFC2': '.xgtf', 'txt_vdrone': '.txt', 'json_coco': '.json'}

    def __init__(self, cfg):
        self.data = dict()
        self.data_type = ''
        self.labels_type = ''
        self.mode = ''
        self.save_location = ''
        self.aug_seq = list()
        self.frame_w = -1
        self.frame_h = -1
        self.checkpoint_path = None
        self.__read_cfg(cfg)

    def __read_cfg(self, cfg_path):
        if not os.path.isfile(cfg_path):
            raise ValueError
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        self.frame_w = cfg['frame_w']
        self.frame_h = cfg['frame_h']
        self.data_type = cfg['data_type']
        self.mode = cfg['mode']
        self.labels_type = cfg['labels_type']
        if self.labels_type not in self.LABELS_TYPE:
            raise ValueError('"labels_type" should be one of: {}'.format(self.LABELS_TYPE))
        self.__get_data(cfg)
        self.save_location = cfg['save_location']
        if 'checkpoint_path' in cfg:
            self.checkpoint_path = cfg['checkpoint_path']

        if (self.save_location is not None) and (not os.path.exists(self.save_location)):
            os.makedirs(self.save_location)
        if 'aug_seq' in cfg:
            self.aug_seq = cfg['aug_seq']

    def manage_dataset(self):
        if self.mode == 'sample':
            self.__sample()
        elif self.mode == 'validate':
            self.__validate_images()
        elif self.mode == 'create_subset':
            pass
        elif self.mode == 'augment':
            self.__augment_dataset()
        else:
            raise ValueError('Cant recognize mode: {}. Mode should be sample or validate'.format(self.mode))

    def __sample(self):
        if self.data_type == 'images':
            self.__sample_from_images()
        elif self.data_type == 'videos':
            self.__sample_from_videos()
        else:
            raise ValueError('Cant recognize data type: {}. data_type should be "images" or '
                             '"videos"'.format(self.data_type))

    def __sample_from_images(self):
        if self.labels_type in ('json_coco', 'txt_vdrone'):
            lc = LabelCreator(self.labels_type)
            total = len(self.data)
            num_skipped = 0
            for i, (data_p, label_p) in enumerate(self.data.items()):
                bboxes = np.array(lc.get_boxes_for_image(label_p, os.path.basename(data_p)))
                if bboxes.size == 0:
                    num_skipped += 1
                    continue
                self.__save_sample(data_p, bboxes)
                su.progress_bar(i, total, 'sampling dataset')
            print('{} out of {} images were skipped, since they dont contain humans.'.format(num_skipped, total))

    def __sample_from_videos(self):
        raise NotImplementedError('Sampling from video not yet implemented.')

    def __validate_images(self):
        if self.labels_type == 'my_ds':
            img_path_dst = os.path.join(self.save_location, 'Images')
            label_path_dst = os.path.join(self.save_location, 'Annotations')
            if not os.path.exists(img_path_dst):
                os.makedirs(img_path_dst)
            if not os.path.exists(label_path_dst):
                os.makedirs(label_path_dst)
            imgs_to_remove = set()
            labels_to_remove = set()
            lc = LabelCreator(self.labels_type)
            if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'r') as f:
                    i = int(f.readline())
            else:
                i = 0
            data_list = list(self.data.items())
            while True:
                if len(self.data) == i:
                    i = 0
                img_path, labels_path = data_list[i]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                bboxes = lc.get_boxes_for_image(labels_path)
                if len(bboxes) > 0:
                    bboxes = bbu.convert_center_to_2points(bboxes)
                    imu.draw_bbox(bboxes, img, color=(0, 0, 255))
                img = imu.resize_img(img, self.frame_h, self.frame_w)
                img = imu.pad_img(img, self.frame_h, self.frame_w)
                cv2.imshow('VALIDATION', img)
                k = cv2.waitKey(0)
                if k == 27:
                    break
                elif k == ord(' '):
                    i += 1
                elif k == ord(','):
                    i -= 1
                    if i < 0:
                        i = 0
                elif k == ord('d'):
                    if img_path in imgs_to_remove:
                        imgs_to_remove.remove(img_path)
                        labels_to_remove.remove(labels_path)
                        print('Items {} and {} were removed from set'.format(os.path.basename(img_path),
                                                                             os.path.basename(labels_path)))
                    else:
                        imgs_to_remove.add(img_path)
                        labels_to_remove.add(labels_path)
                        print('Items {} and {} were added to set'.format(os.path.basename(img_path),
                                                                         os.path.basename(labels_path)))
                    i += 1
            if imgs_to_remove != set() and labels_to_remove != set():
                for img_src, label_src in zip(imgs_to_remove, labels_to_remove):
                    shutil.move(img_src, img_path_dst)
                    shutil.move(label_src, label_path_dst)

            if self.checkpoint_path is not None:
                with open(self.checkpoint_path, 'w') as f:
                    i -= len(imgs_to_remove)
                    f.write(str(i))

    def __get_data(self, cfg):
        path_types = cfg['path_types']
        if path_types['data'] == 'folders' and path_types['annotations'] == 'folders':
            for data_dir, labels_dir in zip(cfg['data_path'], cfg['labels_path']):
                if (not os.path.isdir(data_dir)) or (not os.path.isdir(labels_dir)):
                    raise ValueError('If path_type was specified as folders you should provide all paths in data_path '
                                     'and in labels_path as paths to directories')
                d_paths = du.list_dirs(data_dir, file_ext=self.DATA_EXTENSIONS)
                for d_path in d_paths:
                    base_name = os.path.splitext(os.path.basename(d_path))[
                        0]  # without extension
                    l_path = os.path.join(labels_dir, base_name + self.EXTENSION_MAP[self.labels_type])
                    if os.path.isfile(l_path):
                        self.data.update({d_path: l_path})
                    else:
                        print('Unable to find annotations for {}'.format(d_paths))

        if path_types['data'] == 'single_file' and path_types['annotations'] == 'single_file':
            d_path = cfg['data_path'][0]
            l_path = cfg['labels_path'][0]
            if os.path.splitext(os.path.basename(d_path))[0] == os.path.splitext(os.path.basename(l_path))[0]:
                if (os.path.isfile(d_path) and d_path.endswith(self.DATA_EXTENSIONS) and os.path.isfile(l_path)
                        and l_path.endswith(self.LABELS_EXTENSIONS)):
                    self.data.update({d_path: l_path})

        if path_types['data'] == 'multiple_files' and path_types['annotations'] == 'multiple_files':
            for d_path, l_path in zip(cfg['data_paths'], cfg['labels_path']):
                if os.path.splitext(os.path.basename(d_path))[0] == os.path.splitext(os.path.basename(l_path))[0]:
                    if (os.path.isfile(d_path) and d_path.endswith(self.DATA_EXTENSIONS)
                            and os.path.isfile(l_path) and l_path.endswith(self.LABELS_EXTENSIONS)):
                        self.data.update({d_path: l_path})
                    else:
                        print('Unable to combine {} with {}'.format(d_path, l_path))
                else:
                    raise ValueError('Path to data and path labels should point to files with same name')

        if path_types['data'] == 'folders' and path_types['annotations'] == 'single_file':
            # here we need to do at least simple input sanitazation
            data = du.list_dirs(cfg['data_path'], file_ext=self.DATA_EXTENSIONS)
            self.data = dict.fromkeys(data, cfg['labels_path'][0])
        if path_types['data'] == 'multiple_files' and path_types['annotations'] == 'single_file':
            self.data = dict.fromkeys(cfg['data_path'], cfg['labels_path'][0])

    def __save_sample(self, data, bboxes):
        if not os.path.exists(os.path.join(self.save_location, 'Images')):
            os.makedirs(os.path.join(self.save_location, 'Images'))
        if not os.path.exists(os.path.join(self.save_location, 'Annotations')):
            os.makedirs(os.path.join(self.save_location, 'Annotations'))
        img_base_name = os.path.basename(data)
        txt_base_name = os.path.splitext(img_base_name)[0] + '.txt'
        shutil.copy2(data, os.path.join(self.save_location, 'Images', img_base_name))
        txt_file = os.path.join(self.save_location, 'Annotations', txt_base_name)
        np.savetxt(txt_file, bboxes, fmt='%d')

    def create_subset(self, subset_size):
        new_subset = random.sample(list(self.data.items()), subset_size)
        img_dir = os.path.join(self.save_location, 'Images')
        labels_dir = os.path.join(self.save_location, 'Annotations')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        for data, label in new_subset:
            shutil.move(data, img_dir)
            shutil.move(label, labels_dir)

    def __augment_dataset(self):
        aug_seq = list()
        lc = LabelCreator(self.labels_type)
        for aug in self.aug_seq:
            if aug['name'] == 'HSV':
                hue = aug['hue']
                sat = aug['sat']
                brightness = aug['brightness']
                aug_seq.append(RandomHSV(hue=hue, saturation=sat, brightness=brightness))
            if aug['name'] == 'TRANSLATE':
                translate = aug['translate']
                diff = aug['diff']
                aug_seq.append(RandomTranslate(translate=translate, diff=diff))
            if aug['name'] == 'FLIP':
                prob = aug['prob']
                aug_seq.append(RandomHorizontalFlip(p=prob))
            if aug['name'] == 'SCALE':
                scale = aug['scale']
                diff = aug['diff']
                aug_seq.append(RandomScale(scale=scale, diff=diff))
        seq = Sequence(aug_seq, 1)

        for i, (img_path, labels_path) in enumerate(self.data.items()):
            img = cv2.imread(img_path)[:, :, ::-1]
            bboxes = lc.get_boxes_for_image(labels_path)
            if bboxes.size <= 0:
                continue
            bboxes = bbu.convert_center_to_2points(bboxes)
            # bboxes = bboxes.astype(float)
            img, bboxes = seq(img, bboxes)
            if bboxes.size <= 0:
                continue
            if np.any(bboxes <= 0):
                continue
            img = np.ascontiguousarray(img, dtype=np.uint8)
            img = img[..., ::-1]
            cv2.imshow('DataAug', img)
            name = os.path.basename(os.path.splitext(img_path)[0])
            bboxes = bbu.convert_2points_to_center(bboxes)
            self.__save_data(img, bboxes, name, prefix='aug')
            su.progress_bar(i, len(self.data), prefix='Augmenting dataset')

    def __save_data(self, data, bboxes, name, prefix=''):
        if not os.path.exists(os.path.join(self.save_location, 'Images')):
            os.makedirs(os.path.join(self.save_location, 'Images'))
        if not os.path.exists(os.path.join(self.save_location, 'Annotations')):
            os.makedirs(os.path.join(self.save_location, 'Annotations'))
        img_dir = os.path.join(self.save_location, 'Images')
        label_dir = os.path.join(self.save_location, 'Annotations')
        img_name = prefix + "_" + name + ".jpg"
        counter = 0
        while os.path.exists(os.path.join(img_dir, img_name)):
            counter += 1
            img_name = prefix + '_' + str(counter) + "_" + name + ".jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), data)
        label_name = prefix + "_" + name + '.txt'
        if counter > 0:
            label_name = prefix + '_' + str(counter) + "_" + name + ".txt"
        np.savetxt(os.path.join(label_dir, label_name), bboxes, fmt='%d')


if __name__ == '__main__':
    cfg = '/Users/mac/Documents/Study/IND/yolo_object_detection/data/cfg/dsmanager_conf.json'
    dsm = DSManager(cfg)
    dsm.manage_dataset()
