import os
import random

import cv2
import tqdm
import numpy as np

from crnn.cfg import *


class Preprocess:
    """ Generate a massive amount of data, and pack it in .npz """

    def __init__(self):
        if not os.path.exists(AUG_OUT_DIR):
            os.mkdir(AUG_OUT_DIR)
        self.imgs_dir = os.listdir(SRC_IN_DIR)

        self.pack_idx = 1
        self.cur_pack_nbr = 0

        self.pack_x = []
        self.pack_y = []

    def run(self):
        for _ in tqdm.tqdm(self.imgs_dir, total=len(self.imgs_dir)):
            for n in range(AUG_NBR):
                nbr = np.random.randint(4, 7)
                samples = random.sample(self.imgs_dir, nbr)
                x, y = self.concat(samples)
                self.add_to_pack(x, y)
        self.split()

    def concat(self, samples: list):
        sample_imgs = []
        sample_labels = ""
        for sample in samples:
            sample_img = cv2.imread(os.path.join(SRC_IN_DIR, sample))
            sample_img = cv2.resize(sample_img, (120, 46))
            sample_imgs.append(sample_img)
            sample_labels += sample[:4]
        # for img data
        x = np.hstack(sample_imgs)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, IMG_SIZE)
        x = x.astype(np.float32)
        # for labels
        pattern = f'{{:_<{MAX_LABEL_LENGTH}}}'
        sample_labels = sample_labels.replace('_', '')
        sample_labels = pattern.format(sample_labels)
        y = [ENCODE_DICT[ch] for ch in sample_labels]

        return x, y

    def split(self):
        """ train and val split """
        total = os.listdir(AUG_OUT_DIR)
        val_nbr = int(len(total) * VALIDATION_RATIO)

        f_train = open(TRAIN_TXT_PATH, 'a')
        f_val = open(VAL_TXT_PATH, 'a')
        for i, name in enumerate(total):
            if i < val_nbr:
                f_val.write(name + '\n')
            else:
                f_train.write(name + '\n')
        f_train.close()
        f_val.close()

    def add_to_pack(self, x, y):
        self.pack_x.append(x)
        self.pack_y.append(y)
        self.cur_pack_nbr += 1
        if self.cur_pack_nbr >= PACK_NBR_MAX:
            self.save_local()

    def save_local(self):
        np.savez(os.path.join(AUG_OUT_DIR, f'aug-pack-{self.pack_idx}'),
                 x=np.array(self.pack_x), y=np.array(self.pack_y))
        self.cur_pack_nbr = 0
        self.pack_idx += 1
        self.pack_x.clear()
        self.pack_y.clear()


if __name__ == '__main__':
    pre = Preprocess()
    pre.run()
