import os
import random
import shutil

import cv2
import numpy as np


char2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '_': 10}
num2char_dict = {value: key for key, value in char2num_dict.items()}


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

    print("[*]Training and validation dataset split successfully.")


class ImageTextGenerator:

    def __init__(self,
                 width_shift_range: int = 0,
                 height_shift_range: int = 0,
                 zoom_range: int = 0,
                 shear_range: int = 0,
                 rotation_range: int = 0,
                 blur_factor: int = 0,
                 noise_factor: float = 0.,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False):

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


class DataGenerator:

    def __init__(self, img_dir, img_wh_shape, batch_size, down_sample_factor, max_label_length):
        self.img_dir = img_dir
        self.img_w, self.img_h = img_wh_shape
        self.batch_size = batch_size
        self.per_pred_label_length = int(self.img_w // down_sample_factor)
        self.max_label_length = max_label_length

        self.img_samples = []
        self.label_samples = []
        for filename in os.listdir(self.img_dir):
            name, ext = os.path.splitext(filename)
            img_path = os.path.join(self.img_dir, filename)
            label = name[:4]
            self.img_samples.append(img_path)
            self.label_samples.append(label)
        self.img_samples = np.array(self.img_samples)
        self.label_samples = np.array(self.label_samples)

        self.img_nbr = len(self.img_samples)
        index = np.random.permutation(self.img_nbr)
        # np.random.permutation: same effects as shuffle but with return(a copy)
        self.img_samples = self.img_samples[index]
        self.label_samples = self.label_samples[index]

    def get_data(self):
        labels_length = np.zeros((self.batch_size, 1))
        pred_labels_length = np.full((self.batch_size, 1), self.per_pred_label_length, np.float64)

        while True:
            data, labels = [], []
            to_network_idx = np.random.choice(self.img_nbr, self.batch_size, replace=False)
            img_to_network = self.img_samples[to_network_idx]
            match_labels = self.label_samples[to_network_idx]

            for i, img_file in enumerate(img_to_network):
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.img_w, self.img_h))
                data.append(img)

                label = match_labels[i]
                labels_length[i][0] = len(label)
                nbr_label = [char2num_dict[ch] for ch in label]
                for n in range(self.max_label_length - len(label)):
                    nbr_label.append(char2num_dict['_'])
                labels.append(nbr_label)

            data = np.array(data, dtype=np.float64) / 255.0 * 2 - 1
            data = np.expand_dims(data, axis=-1)
            labels = np.array(labels, dtype=np.float64)

            inputs = {
                "y_true": labels,
                "img_inputs": data,
                "y_pred_length": pred_labels_length,
                "y_true_length": labels_length
            }
            outputs = {"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}

            yield (inputs, outputs)
