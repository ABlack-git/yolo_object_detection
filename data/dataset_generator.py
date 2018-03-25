class DatasetGenerator:

    def __init__(self, dataset_dir, img_size):
        self.dtst_dir = dataset_dir
        self.image_w = img_size[0]
        self.image_h = img_size[1]

    def resize_imgs_and_labels(self):
        raise NotImplementedError

    def adjust_labels(self):
        raise NotImplementedError

    def get_minibatch(self):
        raise NotImplementedError