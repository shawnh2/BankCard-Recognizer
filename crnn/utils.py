import cv2
import numpy as np

from crnn.cfg import *


def fake_ctc_loss(y_true, y_pred):
    return y_pred


class DataGenerator:

    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.img_w, self.img_h = IMG_SIZE
        self.batch_size = BATCH_SIZE
        self.pre_input_label_length = self.img_w // DOWNSAMPLE_FACTOR
        # Read from txt file
        with open(self.txt_path, 'r') as f:
            paths, labels = [], []
            for line in f.readlines():
                path, label = line.rstrip('\n').split(' ')
                paths.append(path)
                labels.append(label)
            self.data_list = np.array(paths)
            self.labels = np.array(labels)
        self.data_nbr = len(self.data_list)
        # Shuffle data list and labels
        index = np.random.permutation(self.data_nbr)
        self.data_list = self.data_list[index]
        self.labels = self.labels[index]

    def flow(self):
        # Feed inputs and outputs to training generator
        input_labels_length = np.full((self.batch_size, 1), self.pre_input_label_length, dtype=np.float64)
        working_labels_length = np.zeros((self.batch_size, 1))
        while True:
            working_data, working_labels = [], []
            index = np.random.choice(self.data_nbr, self.batch_size, replace=False)
            feed_data = self.data_list[index]
            feed_labels = self.labels[index]
            i = 0
            for path, str_label in zip(feed_data, feed_labels):
                # For image data
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
                working_data.append(img)
                # For label data
                label_length = len(str_label)
                working_labels_length[i][0] = label_length
                label = [ENCODE_DICT[c] for c in str_label]
                for _ in range(MAX_LABEL_LENGTH - label_length):
                    label.append(ENCODE_DICT['_'])
                working_labels.append(label)
                i += 1

            working_data = np.array(working_data, dtype=np.float64) / 255.0 * 2.0 - 1.0
            working_data = np.expand_dims(working_data, axis=-1)
            working_labels = np.array(working_labels, dtype=np.float64)
            inputs = {
                "labels": working_labels,
                "img_inputs": working_data,
                "input_length": input_labels_length,
                "label_length": working_labels_length
            }
            outputs = {"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}

            yield inputs, outputs
