import bbox_utils as bbu
import numpy as np
import xml.etree.ElementTree as et
import json

"""
This class is designed to process big data files that cover entire video. 
"""


class LabelCreator:
    def __init__(self, labels_type):
        # "my_ds/txt_virat/txt_okutama/xgtf_mdrone/xgtf_UFC1/xgtf_UFC2/txt_vdrone"
        self.l_type = labels_type
        self.data_dict = {}

    def get_boxes_for_frame(self, labels_path, frame_no, old_shape=None, new_shape=None):
        bboxes = []
        if self.l_type == 'txt_okutama':
            data = self.read_txt(labels_path)
            bboxes = self.get_okutama_bbox(frame_no, data)
        if self.l_type == 'txt_virat':
            pass
        if self.l_type == 'xgtf_mdrone':
            data = self.read_xml(labels_path)
            bboxes = self.get_minidrone_bbox(frame_no, data)
        if self.l_type == 'xgtf_UFC1':
            data = self.read_xml(labels_path)
            bboxes = self.get_ufc1_bboxes(frame_no, data)
        if self.l_type == 'xgtf_UFC2':
            data = self.read_xml(labels_path)
            bboxes = self.get_ufc2_bboxes(frame_no, data)
        if (new_shape is not None) and (old_shape is not None):
            if bboxes:
                bbu.resize_boxes(bboxes, old_shape, new_shape)
        return bboxes

    def get_boxes_for_image(self, labels_path, img_name=None):
        bboxes = []
        if self.l_type == 'txt_vdrone':
            data = self.read_txt(labels_path)
            bboxes = self.get_visdrone_bbox(data)
        elif self.l_type == 'my_ds':
            data = self.read_txt(labels_path)
            bboxes = self.get_my_ds_bboxes(data)
        elif self.l_type == 'json_coco':
            if img_name is None:
                raise ValueError('You should provide image name if coco dataset is chosen')
            data = self.read_json(labels_path)
            bboxes = self.get_coco_bboxes(data, img_name)
        return bboxes

    def read_json(self, labels_path):
        if labels_path in self.data_dict:
            return self.data_dict[labels_path]
        print('Reading json...')
        with open(labels_path) as f:
            data = json.load(f)
        print('Loading complete.')
        self.data_dict.update({labels_path: data})
        return data

    def read_xml(self, labels_path):
        if labels_path in self.data_dict:
            return self.data_dict[labels_path]
        else:
            self.data_dict.update({labels_path: et.parse(labels_path)})

        return self.data_dict[labels_path]

    def get_coco_bboxes(self, data, img_name):
        img_id = -1
        for img_info in data['images']:
            if img_info['file_name'] == img_name:
                img_id = img_info['id']
        if img_id == -1:
            raise ValueError('image id is not found')
        bboxes = []
        for obj in data['annotations']:
            if obj['image_id'] == img_id and obj['category_id'] == 1:
                bboxes.append([int(x) for x in obj['bbox']])
        return bbu.convert_topleft_to_centre(bboxes)

    def read_txt(self, labels_path):
        """
        Reads txt, formats it and saves content to global_data.
        """
        data = []
        if labels_path in self.data_dict:
            return self.data_dict[labels_path]
        else:
            with open(labels_path) as file:
                tmp = file.read().splitlines()
                for item in tmp:
                    if self.l_type == 'txt_okutama':
                        splited = item.split(sep=' ')
                        splited = [int(x) if i < 9 else x.replace('"', '') for i, x in enumerate(splited)]
                        data.append(splited)
                    if self.l_type == 'txt_virat':
                        splited = item.split(sep=' ')
                        splited = [int(x) for x in splited]
                        data.append(splited)
                    if self.l_type == 'txt_vdrone':
                        splited = item.split(sep=',')
                        splited = [int(x) for x in splited if x != '']
                        data.append(splited)
                    if self.l_type == 'my_ds':
                        splited = item.split(sep=' ')
                        if splited[0] == 'None':
                            splited = []
                        else:
                            splited = [int(x) for i, x in enumerate(splited) if i < 4]
                        data.append(splited)
            self.data_dict.update({labels_path: data})

        return self.data_dict[labels_path]

    def get_okutama_bbox(self, frame_no, data):
        """
        This function extracts bounding boxes for given frame for okutama dataset.
        :param frame_no: number of frame.
        :param data:
        :return: bounding boxes.
        """
        bboxes = []
        for item in data:
            if item[5] == frame_no and item[6] != 1 and item[7] != 1:
                tmp = [x for i, x in enumerate(item) if 0 < i < 5]
                if bboxes:
                    # skip boxes that greatly overlap
                    tmp = np.tile(tmp, (len(bboxes), 1))
                    ious = bbu.iou(np.array(bboxes), tmp)
                    iou = (ious > 0.4).astype(int).sum()
                    if iou >= 1:
                        continue
                bboxes.append([x for i, x in enumerate(item) if 0 < i < 5])
        bboxes = bbu.convert_2points_to_center(bboxes)
        return bboxes

    def get_minidrone_bbox(self, frame_no, data):
        root = data.getroot()
        xmlns = "{http://lamp.cfar.umd.edu/viper#}"
        xmlns_data = "{http://lamp.cfar.umd.edu/viperdata#}"
        bboxes = []
        for elem in root.findall('.//' + xmlns + 'object[@name="Person"]'):
            vis_attr = elem.find('.//' + xmlns + 'attribute[@name="Visibility"]')
            occluded = False
            for vis_interval in vis_attr:
                lim = vis_interval.attrib.get('framespan').split(sep=':')
                if int(lim[0]) <= frame_no <= int(lim[1]):
                    if vis_interval.attrib.get('value') == 'strong_occlusion':
                        occluded = True
            if occluded:
                continue
            for child in elem.iter(tag=xmlns_data + 'bbox'):
                occluded = False
                # check for occlusion
                lim = child.attrib.get('framespan').split(sep=':')
                if (int(lim[0]) <= frame_no <= int(lim[1])) and (not occluded):
                    x = child.attrib.get('x')
                    y = child.attrib.get('y')
                    w = child.attrib.get('width')
                    h = child.attrib.get('height')
                    bboxes.append([max(int(x), 0), max(int(y), 0), max(int(w), 0), max(int(h), 0)])
        if bboxes:
            bboxes = bbu.convert_topleft_to_centre(bboxes)
        return bboxes

    def get_visdrone_bbox(self, data):
        bboxes = np.array([b for b in data if (int(b[5]) == 1 or int(b[5]) == 2) and int(b[7]) != 2])
        if len(bboxes) > 0:
            bboxes = bboxes[:, :5]
            bboxes = bbu.convert_topleft_to_centre(bboxes)
        return bboxes

    def get_ufc1_bboxes(self, frame_no, data):
        xmlns = "{http://lamp.cfar.umd.edu/viper#}"
        xmlns_data = "{http://lamp.cfar.umd.edu/viperdata#}"
        bboxes = []
        for person in data.findall('.//' + xmlns + 'object[@name="PERSON"]'):
            occluded = False
            attr_occlusion = person.find('.//' + xmlns + 'attribute[@name="Occlusion"]')
            for occ_element in attr_occlusion:
                fs = occ_element.attrib.get('framespan').split(':')
                if int(fs[0]) <= frame_no <= int(fs[1]):
                    if occ_element.attrib.get('value') == 1:
                        occluded = True
            if occluded:
                continue
            data_bbox = person.findall('.//' + xmlns_data + 'bbox')
            for bbox_element in data_bbox:
                fs = bbox_element.attrib.get('framespan').split(':')
                if int(fs[0]) <= frame_no <= int(fs[1]):
                    bboxes.append([int(bbox_element.attrib.get('x')),
                                   int(bbox_element.attrib.get('y')),
                                   int(bbox_element.attrib.get('width')),
                                   int(bbox_element.attrib.get('height')),
                                   ])
        if len(bboxes) > 0:
            bboxes = bbu.convert_topleft_to_centre(bboxes)

        return bboxes

    def get_ufc2_bboxes(self, frame_no, data):
        xmlns = "{http://lamp.cfar.umd.edu/viper#}"
        xmlns_data = "{http://lamp.cfar.umd.edu/viperdata#}"
        bboxes = []
        for obj in data.findall('.//' + xmlns + 'object'):
            if obj.find('.//' + xmlns_data + 'svalue').get('value') == 'man':
                for bbox_element in obj.findall('.//' + xmlns_data + 'bbox'):
                    fs = bbox_element.attrib.get('framespan').split(':')
                    if int(fs[0]) <= frame_no <= int(fs[1]):
                        bboxes.append([int(bbox_element.attrib.get('x')),
                                       int(bbox_element.attrib.get('y')),
                                       int(bbox_element.attrib.get('width')),
                                       int(bbox_element.attrib.get('height'))])
        if len(bboxes) > 0:
            bboxes = bbu.convert_topleft_to_centre(bboxes)
        return bboxes

    def get_my_ds_bboxes(self, data):
        return np.array(data)
