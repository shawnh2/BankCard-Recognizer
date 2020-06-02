import os
import random

import cv2
import tqdm
import numpy as np

from crnn.cfg import *


AUG_LIST = []

if not os.path.exists(AUG_OUT_DIR):
    os.mkdir(AUG_OUT_DIR)


def do_sample(k: int = 1):
    f_train = open(TRAIN_TXT_PATH, 'a')
    f_val = open(VAL_TXT_PATH, 'a')

    val = random.sample(AUG_LIST, k)
    for item in AUG_LIST:
        line = "{path} {label}\n".format(path=item[0], label=item[1])
        if item in val:
            f_val.write(line)
        else:
            f_train.write(line)
    # Clean it
    AUG_LIST.clear()
    f_train.close()
    f_val.close()


def do_concat(samples: list, name: str) -> (str, str):
    """ Return new path and label """
    sample_imgs = []
    sample_labels = ""
    for sample in samples:
        sample_img = cv2.imread(os.path.join(SRC_IN_DIR, sample))
        sample_img = cv2.resize(sample_img, (120, 46))
        sample_imgs.append(sample_img)
        sample_labels += sample[:4]
    stacked = np.hstack(sample_imgs)
    new_path = os.path.join(AUG_OUT_DIR, name + '.png')
    cv2.imwrite(new_path, stacked)
    return new_path, sample_labels


def main():
    imgs = os.listdir(SRC_IN_DIR)
    for i, _ in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
        for n in range(AUG_NBR):
            nbr = np.random.randint(4, 6)
            samples = random.sample(imgs, nbr)
            path, label = do_concat(samples, "{:05d}-{:03d}".format(i, n))
            AUG_LIST.append((path, label))
        # Every epoch choose k img from aug list as val img
        do_sample(k=max(4, int(AUG_NBR * 0.15)))


if __name__ == '__main__':
    main()
