import os
import random
import shutil

import cv2
import numpy as np

from bankcard_rec.aug import DataAugmentation


random.seed(58)
CHARS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_')
IMG_SHAPE = (46, 120, 3)


def train_test_spilt(inputs_dir, output_dir, test_train_ratio=0.2):
    """In this spilt, train and test dataset
    is better when train takes 80% and test takes 20%.
    The number of this ratio can be changed also.
    But still recommend 20% of test takes.
    """

    if not os.path.exists(inputs_dir) or not os.path.exists(output_dir):
        raise FileExistsError("Inputs/Outputs dir is not exist.")

    inputs_dir_li = [n for n in os.listdir(inputs_dir)]
    inputs_dir_len = len(inputs_dir_li)
    test_len = int(inputs_dir_len * test_train_ratio)

    random.shuffle(inputs_dir_li)
    test_part = inputs_dir_li[: test_len]
    train_part = inputs_dir_li[test_len:]
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for test_name in test_part:
        test_file_path = os.path.join(inputs_dir, test_name)
        shutil.copy(test_file_path, test_dir)
    for train_name in train_part:
        train_file_path = os.path.join(inputs_dir, train_name)
        shutil.copy(train_file_path, train_dir)


class ImageTextGenerator:
    """# The shape of image with following orders:
          (img_height, img_width, img_channel).
       # The backend of image is default by:
          channel_last.
    """

    def __init__(self,
                 dirs,
                 width_shift_range: int = 0,
                 height_shift_range: int = 0,
                 zoom_range: int = 0,
                 shear_range: int = 0,
                 rotation_range: int = 0,
                 blur_factor: int = 0,
                 noise_factor: float = 0.,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False):

        self.img_h, self.img_w, self.img_c = IMG_SHAPE
        self.chars_index = dict(zip(CHARS, range(len(CHARS))))
        self.dirs = dirs

        self.samples = []
        for filename in os.listdir(dirs):
            name, ext = os.path.splitext(filename)
            img_filepath = os.path.join(dirs, filename)
            description = name[:4]
            self.samples.append([img_filepath, description])
        self.cur_index = 0
        self.n = len(self.samples)

        # the abbreviation 'code:params' of augment operation
        self.param_dict = {
            'wsr': width_shift_range,
            'hsr': height_shift_range,
            'zor': zoom_range,
            'shr': shear_range,
            'ror': rotation_range,
            'blr': blur_factor,
            'nof': noise_factor,
            'hfl': horizontal_flip,
            'vfl': vertical_flip,
        }
        # the sign of whether taking any augmentation
        self.aug_switch = any(self.param_dict.values())

    def _process_img(self, img):
        # Extract this part from fetch function.
        # Process the each image with this procedure.
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32)
        img /= 255
        return img

    def _fetch_with_aug(self, batch_size):
        # Every time fetch one, soon will run out the samples.
        # So when reach the deadline, pick one index randomly.
        if self.cur_index >= self.n:
            self.cur_index = random.randint(0, self.n-1)
        # In batch size, the first one image is original one,
        # and other batch_size-1 is the augment image.
        img_filepath, description = self.samples[self.cur_index]
        self.cur_index += 1
        # Initialize x and y data, also img and str data
        x_data = np.zeros((batch_size, self.img_h, self.img_w, self.img_c), dtype=np.float32)
        y_data = [description] * batch_size

        img = cv2.imread(img_filepath)
        # Prepare the image augment.
        # Because in this time, img still keep dtype as uint8
        # and the DataAugmentation only accept this.
        aug = DataAugmentation(img, self.param_dict)
        # This add into the first original image.
        x_data[0] = self._process_img(img)
        # Generate the augmentation with size(batch-1).
        for i, aug_img in enumerate(aug.feed(batch_size-1), start=1):
            x_data[i] = self._process_img(aug_img)
        # Merge x_data(image) together and duplicate the label.

        return x_data, y_data

    def _build_data_without_aug(self):
        # These class variables will be activated when this methods has been called.
        # Because of the mini-batch of validation dataset,
        # these data should be arranged as soon as possible.
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, self.img_c), dtype=np.float32)
        self.texts = []
        for i, (img_filepath, description) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            self.imgs[i] = self._process_img(img)
            self.texts.append(description)

    def _fetch_without_aug(self, batch_size):
        self.cur_index += 1
        if self.cur_index + batch_size >= self.n:
            self.cur_index = 0
        return (self.imgs[self.cur_index: self.cur_index + batch_size],
                self.texts[self.cur_index: self.cur_index + batch_size])

    def flow(self, batch_size):
        """
        This kind of flow is suit for training generator.
        In this flow, batch_size must assign.
        Do that because the original dataset will soon be run out.

        :param batch_size: contains (one) sample (and rest are augment).
        :return: one epochs of data (with augmentation).
        """
        assert batch_size > 0
        # Build dataset in silence.
        if not self.aug_switch:
            self._build_data_without_aug()
        while True:
            # Verdict of which fetch method should be taken
            if self.aug_switch:
                x_data, y_data = self._fetch_with_aug(batch_size)
            else:
                x_data, y_data = self._fetch_without_aug(batch_size)
            # Mapping the chars with index of it.
            y_tmp = np.array(list(map(lambda x: [self.chars_index[a] for a in list(x)], y_data)), dtype=np.uint8)
            ys = np.zeros([y_tmp.shape[1], batch_size, len(CHARS)])
            for batch in range(batch_size):
                for i, row in enumerate(y_tmp[batch]):
                    ys[i, batch, row] = 1

            yield x_data, [y for y in ys]
