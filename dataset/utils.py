import os
import random

import cv2
import numpy as np

from aug import DataAugmentation


def train_val_split(src_dir, val_split_ratio=0.2):
    # Prepare a whole list of images and labels.
    data_path, labels = [], []
    for file in os.listdir(src_dir):
        data_path.append(src_dir + file)
        name, ext = os.path.splitext(file)
        labels.append(name[:4])

    # Select val-set by index randomly.
    length = len(data_path)
    rand_index = list(range(length))
    random.shuffle(rand_index)
    val_index = rand_index[0: int(val_split_ratio * length)]

    # Collect them and return.
    train_set, val_set = [], []
    for i in range(length):
        if i in val_index:
            val_set.append((data_path[i], labels[i]))
        else:
            train_set.append((data_path[i], labels[i]))
    return train_set, val_set


def data_wrapper(src_list, img_shape, max_label_length,
                 max_aug_nbr=0, aug_param_dict=None, name=None):
    """
    When fetching data to the training generator,
    the DataGenerator will select one batch from Pool randomly.
    But doing data augmentation in Pool will take a lot of memories.
    (1085 images * 80 augmentation number or even more)
    For saving those unnecessary cause, the Pool will be saved after augmentation.
    So the DataGenerator will fetch data from local instead of memories.

    :param src_list: Selected train/val img path with form (path, label).
    :param img_shape: Image shape (width, height)
    :param max_aug_nbr: Max number of doing augmentation.
    :param max_label_length: Max number of label length.
    :param aug_param_dict: A dict for saving data-augmentation parameters.
                           With code:
                           'hsr': height_shift_range      'wsr': width_shift_range
                           'ror': rotate_range            'zor': zoom_range            'shr': shear_range
                           'hfl': horizontal_flip         'vfl': vertical_flip
                           'nof': noise_factor            'blr': blur_factor
    :param save_with_name: Whether save it local or not, then it will return.
    :return: 0 or data, labels, labels_length
    """
    assert max_aug_nbr >= 0
    data, labels, labels_length = [], [], []
    save_local = False

    def valid_img(image):
        # Do some common process to image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, img_shape)
        image = image.astype(np.float32)
        return image

    def valid_label(label_string):
        # Get rid of the empty placeholder '_'.
        # Even it is the head of label.
        res = []
        for ch in label_string:
            if ch == '_':
                continue
            else:
                res.append(int(ch))
        n = len(res)
        for i in range(max_label_length - n):
            res.append(10)  # represent '_'
        # Return res for labels, length for labels_length
        return res, n

    for path, label in src_list:
        img = cv2.imread(path)
        data.append(valid_img(img))
        v_lab, v_len = valid_label(label)
        labels.append(v_lab)
        labels_length.append(v_len)

        if max_aug_nbr != 0 and aug_param_dict is not None and any(aug_param_dict):
            # Once trigger the data augmentation, it will be saved in local.
            save_local = True
            aug = DataAugmentation(img, aug_param_dict)
            # max_aug_nbr = original_img(.also 1) + augment_img
            for aug_img in aug.feed(max_aug_nbr-1):
                data.append(valid_img(aug_img))
                # Different augmentation of images, but same labels and length.
                labels.append(v_lab)
                labels_length.append(v_len)
    data = np.array(data, dtype=np.float64) / 255.0 * 2 - 1
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels, dtype=np.float64)
    labels_length = np.array(labels_length).T

    """if save_local:
        np.savez(name + ".npz", data=data, labels=labels, labels_length=labels_length)
        return 0
    else:"""
    return data, labels, labels_length
