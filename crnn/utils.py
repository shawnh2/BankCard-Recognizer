import os
import random
import shutil

import cv2
import numpy as np


def train_test_spilt(inputs_dir, output_dir, test_train_ratio=0.2):
    """Train and test dataset is better to take 4:1 -> (0.8:0.2)
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





def fake_ctc_loss(y_true, y_pred):
    '''
    这个函数是为了符合keras comepile的要求入口参数只能有y_true和y_pred
    之后在结合我们的ctc_loss_layer一起工作
    '''
    return y_pred