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
    LABELS_TYPE = ('my_ds', 'txt_virat', 'txt_okutama', 'xgtf_mdrone', 'xgtf_UFC1', 'xgtf_UFC2', 'txt_vdrone')
    DATA_EXTENSIONS = ('.MOV', '.mov', '.MP4', '.mp4', '.mpg', '.jpg')
    LABELS_EXTENSIONS = ('.txt', '.xgtf')

    def __init__(self, cfg):
        self.data_type = ''
        self.d_paths = []
        self.l_paths = []
        self.l_type = ''
        self.no_labels = []
        self.no_vids = []
        self.read_cfg(cfg)
        self.__data_to_labels()
        # if self.data_type == 'videos':
        #     self.__videos_to_labels()
        # if self.data_type == 'images':
        #     self.__data_to_labels()

    def read_cfg(self, cfg_path):
        if not os.path.isfile(cfg_path):
            raise ValueError
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if cfg['data_type'] != 'videos' and cfg['data_type'] != 'images':
            raise ValueError('"data_type" in config file should be either "videos" or "images".')
        self.data_type = cfg['data_type']
        if (cfg['path_type'] != "folders" and cfg['path_type'] != 'single_file'
                and cfg['path_type'] != "multiple_files"):
            raise ValueError('"path_type" in config file should be "folders" or "single_file" '
                             'or "multiple_files"')
        if cfg['labels_type'] not in self.LABELS_TYPE:
            raise ValueError('"labels_type" should be either "my_ds", "txt_virat", "txt_okutama", "xgtf_mdrone", '
                             '"xgtf_UFC1", "xgtf_UFC2" or "txt_vdrone"')
        self.l_type = cfg['labels_type']

        if cfg['path_type'] == 'folders':
            for data_dir, labels_dir in zip(cfg['data_path'], cfg['labels_path']):
                if (not os.path.isdir(data_dir)) or (not os.path.isdir(labels_dir)):
                    raise ValueError('If path_type was specified as folders you should provide all paths in data_path '
                                     'and in labels_path as paths to directories')
                self.d_paths += du.list_dirs(data_dir, file_ext=self.DATA_EXTENSIONS)
                self.l_paths += du.list_dirs(labels_dir, file_ext=self.LABELS_EXTENSIONS)

        if cfg['path_type'] == 'single_file':
            if os.path.isfile(cfg['data_path'][0]) and cfg['data_path'][0].endswith(self.DATA_EXTENSIONS):
                self.d_paths = cfg['data_paths']
            else:
                raise ValueError('Path provided in "data_path" should point to existing file and have one of the '
                                 'following extentions: {}'.format(self.DATA_EXTENSIONS))
            if os.path.isfile(cfg['labels_path'][0]) and cfg['data_path'][0].endswith(self.LABELS_EXTENSIONS):
                self.l_paths = cfg['labels_paths']
            else:
                raise ValueError('Path provided in "labels_path" should point to existing file and have one of the '
                                 'following extentions: {}'.format(self.LABELS_EXTENSIONS))

        if cfg['path_type'] == 'multiple_files':
            for data_path, label_path in zip(cfg['data_paths'], cfg['labels_path']):
                if os.path.isfile(data_path) and data_path.endswith(self.DATA_EXTENSIONS):
                    self.d_paths.append(data_path)
                else:
                    raise ValueError('Paths provided in "data_path" should point to existing files and ALL have one of '
                                     'the following extentions: .MOV, .mov, .MP4, .mp4, .mpg, .jpg')
                if os.path.isfile(label_path) and label_path.endswith(self.LABELS_EXTENSIONS):
                    self.l_paths.append(label_path)
                else:
                    raise ValueError('Paths provided in "labels_path" should point to existing files and have one of '
                                     'the following extentions: .txt or .xgtf')

    def display_dataset(self):
        if self.data_type == 'videos':
            self.__display_videos()
        if self.data_type == 'images':
            self.__display_images()

    def __videos_to_labels(self):
        print('Length of v_path list: ', len(self.d_paths))
        print('Length of l_path list: ', len(self.l_paths))
        l_tmp = [os.path.splitext(os.path.basename(l_path))[0] for l_path in self.l_paths]
        v_tmp = []
        for v_path in self.d_paths:
            if os.path.splitext(os.path.basename(v_path))[0] in l_tmp:
                if self.l_type != 'xgtf_UFC2':
                    v_tmp.append(v_path)
            else:
                self.no_labels.append(v_path)
        self.d_paths = v_tmp

        l_tmp = []
        v_tmp = [os.path.splitext(os.path.basename(v_path))[0] for v_path in self.d_paths]
        for path in self.l_paths:
            if os.path.splitext(os.path.basename(path))[0] in v_tmp:
                if self.l_type != 'xgtf_UFC2':
                    l_tmp.append(path)
            else:
                self.no_vids.append(path)
        self.l_paths = l_tmp
        self.l_paths.sort()
        self.d_paths.sort()
        if self.no_labels:
            print('No labels list:')
            for item in self.no_labels:
                print(os.path.basename(item), sep=', ')
            print()
        else:
            print('No labels list is empty')

        if self.no_vids:
            print('No videos list:')
            for item in self.no_vids:
                print(os.path.basename(item), sep=', ')
            print()
        else:
            print('No videos list is empty')
        print('Length of v_path list: ', len(self.d_paths))
        print('Length of l_path list: ', len(self.l_paths))

    def __data_to_labels(self):
        if len(self.d_paths) < len(self.l_paths):
            no_data = []
            for l_item in self.l_paths:
                if os.path.splitext(os.path.basename(l_item))[0] not in [os.path.splitext(os.path.basename(x))[0] for x
                                                                         in self.d_paths]:
                    no_data.append(l_item)
            print('Unable to find data for the following labels: {}. They will be ignored.'.format(no_data))
            for to_del in no_data:
                self.l_paths.remove(to_del)
        if len(self.l_paths) < len(self.d_paths):
            no_data = []
            for d_item in self.d_paths:
                if os.path.splitext(os.path.basename(d_item))[0] not in [os.path.splitext(os.path.basename(x))[0] for x
                                                                         in self.l_paths]:
                    no_data.append(d_item)
            print('Unable to find labels for the following data: {}. They will be ignored'.format(no_data))
            for to_del in no_data:
                self.d_paths.remove(to_del)
        if len(self.d_paths) == len(self.l_paths):
            self.d_paths.sort()
            self.l_paths.sort()
            for d_item, l_item in zip(self.d_paths, self.l_paths):
                if os.path.splitext(os.path.basename(d_item))[0] != os.path.splitext(os.path.basename(l_item))[0]:
                    raise ValueError('Data and label have different names: {}:{}'.format(d_item, l_item))

    def __display_videos(self):
        for v_path, l_path in zip(self.d_paths, self.l_paths):
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
                frame = imgu.resize_img(frame, 720, 960)
                if len(bboxes) > 0:
                    # bboxes = bbu.convert_center_to_2points(bboxes)
                    bboxes = bbu.resize_boxes(bboxes, old_size, (720, 960))
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
        for data, label in zip(self.d_paths, self.l_paths):
            img = cv2.imread(data, cv2.IMREAD_COLOR)
            old_size = img.shape[:2]
            bboxes = lc.get_boxes_for_image(label)
            img = imgu.resize_img(img, 500, 500)
            if len(bboxes) > 0:
                bboxes = bbu.convert_center_to_2points(bboxes)
                bboxes = bbu.resize_boxes(bboxes, old_size, img.shape[:2])
                imgu.draw_bbox(bboxes, img, color=[0, 0, 255], thickness=1)
            img = imgu.pad_img(img, 500, 500)
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
