import os
import json
import data_utils as du
import numpy as np
import shutil
import stats_utils as su
from data.label_creator import LabelCreator


class DSManager:
    LABELS_TYPE = ('my_ds', 'txt_virat', 'txt_okutama', 'xgtf_mdrone', 'xgtf_UFC1', 'xgtf_UFC2', 'txt_vdrone')
    DATA_EXTENSIONS = ('.MOV', '.mov', '.MP4', '.mp4', '.mpg', '.jpg')
    LABELS_EXTENSIONS = ('.txt', '.xgtf')

    def __init__(self, cfg):
        self.data_type = ''
        self.path_type = ''
        self.labels_type = ''
        self.data_path = []
        self.labels_path = []
        self.mode = ''
        self.save_location = ''
        self.__read_cfg(cfg)

    def __read_cfg(self, cfg_path):
        if not os.path.isfile(cfg_path):
            raise ValueError
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        self.data_type = cfg['data_type']
        self.mode = cfg['mode']
        self.path_type = cfg['path_type']
        self.labels_type = cfg['labels_type']
        if self.labels_type not in self.LABELS_TYPE:
            raise ValueError('"labels_type" should be one of: {}'.format(self.LABELS_TYPE))
        self.__get_data(cfg['data_path'], cfg['labels_path'])
        self.save_location = cfg['save_location']
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

    def manage_dataset(self):
        if self.mode == 'sample':
            self.__sample()
        elif self.mode == 'validate':
            self.__validate_images()
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
        if self.labels_type == 'txt_vdrone':
            lc = LabelCreator(self.labels_type)
            total = len(self.data_path)
            num_skipped = 0
            for i, (data_p, label_p) in enumerate(zip(self.data_path, self.labels_path)):
                bboxes = np.array(lc.get_boxes_for_image(label_p))
                if bboxes.size == 0:
                    num_skipped += 1
                    continue
                self.__save_sample(data_p, bboxes)
                su.progress_bar(i, total, 'sampling dataset')

            print('{} out of {} images were skipped, since they dont contain humans.'.format(num_skipped, total))

    def __sample_from_videos(self):
        raise NotImplementedError('Sampling from video not yet implemented.')

    def __validate_images(self):
        pass

    def __get_data(self, data, labels):
        if self.path_type == 'single_file':
            if os.path.isfile(data[0]) and data[0].endswith(self.DATA_EXTENSIONS):
                self.data_path = data
            else:
                raise ValueError('Path provided in "data_path" should point to existing file and have one of the '
                                 'following extensions: {}'.format(self.DATA_EXTENSIONS))
            if os.path.isfile(labels[0]) and labels[0].endswith(self.LABELS_EXTENSIONS):
                self.labels_path = labels
            else:
                raise ValueError('Path provided in "labels_path" should point to existing file and have one of the '
                                 'following extensions: {}'.format(self.LABELS_EXTENSIONS))
        elif self.path_type == 'folders':
            for data_dir, labels_dir in zip(data, labels):
                if not os.path.isdir(data_dir) or not os.path.isdir(labels_dir):
                    raise ValueError('If path_type was specified as folders you should provide all paths in data_path '
                                     'and in labels_path as paths to directories')
                self.data_path += du.list_dir(data_dir, self.DATA_EXTENSIONS)
                self.labels_path += du.list_dir(labels_dir, self.LABELS_EXTENSIONS)
        elif self.path_type == 'multiple_files':
            raise NotImplementedError('Multiple files are not yet supported')
        else:
            raise ValueError('Cant recognize path type: {}. path_type should be "single_file" or "folders" or '
                             '"multiple_files"')
        self.__data_to_labels()

    def __data_to_labels(self):
        if len(self.data_path) < len(self.labels_path):
            no_data = []
            for l_item in self.labels_path:
                if os.path.splitext(os.path.basename(l_item))[0] not in [os.path.splitext(os.path.basename(x))[0] for x
                                                                         in self.data_path]:
                    no_data.append(l_item)
            print('Unable to find data for the following labels: {}. They will be ignored.'.format(no_data))
            for to_del in no_data:
                self.labels_path.remove(to_del)
        if len(self.labels_path) < len(self.data_path):
            no_data = []
            for d_item in self.data_path:
                if os.path.splitext(os.path.basename(d_item))[0] not in [os.path.splitext(os.path.basename(x))[0] for x
                                                                         in self.labels_path]:
                    no_data.append(d_item)
            print('Unable to find labels for the following data: {}. They will be ignored'.format(no_data))
            for to_del in no_data:
                self.data_path.remove(to_del)
        if len(self.data_path) == len(self.labels_path):
            self.data_path.sort()
            self.labels_path.sort()
            for d_item, l_item in zip(self.data_path, self.labels_path):
                if os.path.splitext(os.path.basename(d_item))[0] != os.path.splitext(os.path.basename(l_item))[0]:
                    raise ValueError('Data and label have different names: {}:{}'.format(d_item, l_item))

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


if __name__ == '__main__':
    cfg = '/Users/mac/Documents/Study/IND/yolo_object_detection/data/cfg/dsmanager_conf.json'
    dsm = DSManager(cfg)
    dsm.manage_dataset()
