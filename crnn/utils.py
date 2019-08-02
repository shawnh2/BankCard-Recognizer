import numpy as np
from dataset.utils import data_wrapper


class DataGenerator:

    def __init__(self,
                 data_list,
                 img_shape,
                 batch_size,
                 down_sample_factor=4,
                 max_label_length=26,
                 max_aug_nbr=0,
                 width_shift_range=0,    # ---------------------
                 height_shift_range=0,   # |
                 zoom_range=0,           # |
                 shear_range=0,          # |
                 rotation_range=0,       # |>>>>data_augmentation_options
                 blur_factor=None,       # |
                 add_noise_factor=0.,    # |
                 horizontal_flip=False,  # |
                 vertical_flip=False,    # ---------------------
                 has_wrapped_dataset=None
                 ):
        # The path of data.
        self.data_list = data_list
        self.img_w, self.img_h = img_shape
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.pre_pred_label_length = int(self.img_w // down_sample_factor)
        self.data_nbr = len(data_list)

        # the abbreviation of "code:params" in augment operations
        self.param_dict = {
            'wsr': width_shift_range,
            'hsr': height_shift_range,
            'zor': zoom_range,
            'shr': shear_range,
            'ror': rotation_range,
            'blr': blur_factor,
            'nof': add_noise_factor,
            'hfl': horizontal_flip,
            'vfl': vertical_flip
        }
        # the sign of whether taking any augmentation
        self.max_aug_nbr = max_aug_nbr
        if has_wrapped_dataset is not None:
            self.data, self.labels, self.labels_length = np.load(has_wrapped_dataset)
        else:
            sign = data_wrapper(data_list, img_shape, max_label_length, max_aug_nbr, self.param_dict, name="train")
            if isinstance(sign, str):
                self.data, self.labels, self.labels_length = np.load("%s.npz" % sign)
            else:
                self.data, self.labels, self.labels_length = sign

            # Shuffle the data by its index.
        index = np.random.permutation(self.data_nbr)
        self.data = self.data[index]
        self.labels = self.labels[index]
        self.labels_length = self.labels_length[index]

    def flow(self):
        # Feed inputs and outputs to training generator
        pred_labels_length = np.full((self.batch_size, 1), self.pre_pred_label_length, dtype=np.float64)

        while True:
            working_index = np.random.choice(self.data_nbr, self.batch_size, replace=False)
            working_data = self.data[working_index]
            working_labels = self.labels[working_index]
            working_labels_length = self.labels_length[working_index]
            inputs = {
                "y_true": working_labels,
                "img_inputs": working_data,
                "y_pred_length": pred_labels_length,
                "y_true_length": working_labels_length
            }
            outputs = {"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}

            yield (inputs, outputs)
