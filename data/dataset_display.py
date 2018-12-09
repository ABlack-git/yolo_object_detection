"""
Purpose of this file is to verify that all videos have correct labels.
Loads videos and draw bboxes on them and displays them.
"""
import cv2
from data.label_creator import LabelCreator
import image_utils as imgu
import bbox_utils as bbu
import data_utils as du
import os
import json


class DatasetDisplay:
    LABELS_TYPE = ('my_ds', 'txt_virat', 'txt_okutama', 'xgtf_mdrone', 'xgtf_UFC1', 'xgtf_UFC2', 'txt_vdrone',
                   'json_coco')
    DATA_EXTENSIONS = ('.MOV', '.mov', '.MP4', '.mp4', '.mpg', '.jpg')
    LABELS_EXTENSIONS = ('.txt', '.xgtf', '.json')
    EXTENSION_MAP = {'my_ds': '.txt', 'txt_virat': '.txt', 'txt_okutama': '.txt', 'xgtf_mdrone': '.xgtf',
                     'xgtf_UFC1': '.xgtf', 'xgtf_UFC2': '.xgtf', 'txt_vdrone': '.txt', 'json_coco': '.json'}

    def __init__(self, cfg):
        self.data_type = ''
        self.data = {}
        self.l_type = ''
        self.frame_w = -1
        self.frame_h = -1
        self.read_cfg(cfg)

    def read_cfg(self, cfg_path):
        if not os.path.isfile(cfg_path):
            raise ValueError
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        self.frame_w = cfg['frame_w']
        self.frame_h = cfg['frame_h']
        if cfg['data_type'] != 'videos' and cfg['data_type'] != 'images':
            raise ValueError('"data_type" in config file should be either "videos" or "images".')
        self.data_type = cfg['data_type']
        if cfg['labels_type'] not in self.LABELS_TYPE:
            raise ValueError('"labels_type" should be either "my_ds", "txt_virat", "txt_okutama", "xgtf_mdrone", '
                             '"xgtf_UFC1", "xgtf_UFC2" or "txt_vdrone"')
        self.l_type = cfg['labels_type']
        path_types = cfg['path_types']
        if path_types['data'] == 'folders' and path_types['annotations'] == 'folders':
            for data_dir, labels_dir in zip(cfg['data_path'], cfg['labels_path']):
                if (not os.path.isdir(data_dir)) or (not os.path.isdir(labels_dir)):
                    raise ValueError('If path_type was specified as folders you should provide all paths in data_path '
                                     'and in labels_path as paths to directories')
                d_paths = du.list_dirs(data_dir, file_ext=self.DATA_EXTENSIONS)
                for d_path in d_paths:
                    base_name = os.path.splitext(os.path.basename(d_path))[0]  # without extension
                    l_path = os.path.join(labels_dir, base_name + self.EXTENSION_MAP[self.l_type])
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

    def display_dataset(self):
        if self.data_type == 'videos':
            self.__display_videos()
        if self.data_type == 'images':
            self.__display_images()

    def __display_videos(self):
        for v_path, l_path in self.data.items():
            cv2.destroyAllWindows()
            video = cv2.VideoCapture(v_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            ms = int((1 / fps) * 1000)
            v_name = os.path.basename(v_path)
            lc = LabelCreator(self.l_type)
            frame_no = 0
            cv2.namedWindow(v_name)
            while True:
                grabbed, frame = video.read()

                if not grabbed:
                    break
                bboxes = lc.get_boxes_for_frame(l_path, frame_no)
                imgu.draw_text('Frame number: ' + str(frame_no), frame)
                if len(bboxes) > 0:
                    bboxes = bbu.convert_center_to_2points(bboxes)
                    imgu.draw_bbox(bboxes, frame, color=(0, 0, 255), thickness=5)
                old_size = frame.shape[:2]
                frame = imgu.resize_img(frame, self.frame_h, self.frame_w)
                if len(bboxes) > 0:
                    # bboxes = bbu.convert_center_to_2points(bboxes)
                    bboxes = bbu.resize_boxes(bboxes, old_size, (self.frame_h, self.frame_w))
                    imgu.draw_bbox(bboxes, frame, thickness=1)
                cv2.imshow(v_name, frame)

                k = cv2.waitKey(ms)
                if k == 27:
                    break
                if k == ord('p'):
                    input('Press enter to continue.')
                frame_no += 1

    def __display_images(self):
        lc = LabelCreator(self.l_type)
        for data, label in self.data.items():
            img = cv2.imread(data, cv2.IMREAD_COLOR)
            old_size = img.shape[:2]
            bboxes = lc.get_boxes_for_image(label, os.path.basename(data))
            img = imgu.resize_img(img, self.frame_h, self.frame_w)
            if len(bboxes) > 0:
                bboxes = bbu.convert_center_to_2points(bboxes)
                bboxes = bbu.resize_boxes(bboxes, old_size, img.shape[:2])
                imgu.draw_bbox(bboxes, img, color=[0, 0, 255], thickness=1)
            else:
                continue
            img = imgu.pad_img(img, self.frame_h, self.frame_w)
            # else:
            #     continue
            cv2.imshow('Image', img)
            k = cv2.waitKey(1 * 500)
            if k == 27:
                break
            if k == ord('p'):
                input('Press enter to continue.')


def main():
    cfg = '/Users/mac/Documents/Study/IND/yolo_object_detection/data/cfg/dataset_display_conf.json'
    dd = DatasetDisplay(cfg)
    dd.display_dataset()


if __name__ == '__main__':
    main()
