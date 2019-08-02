import numpy as np
from dataset.utils import data_wrapper


class DataGenerator:

    def __init__(self,
                 data_list,
                 img_shape,
                 batch_size,
                 down_sample_factor=4,
                 max_label_length=26,
                 max_aug_nbr=1,
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
        self.data_nbr = len(data_list) * max_aug_nbr
        self.local_dataset_path = None

        # Loading data file from .npz file
        self.load_data = None
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
        # Loading the data directly because it's been shuffled already.
        if has_wrapped_dataset is not None:
            print("[*] Using local wrapped dataset.")
            self.load_data = np.load(has_wrapped_dataset)
        else:
            sign = data_wrapper(data_list, img_shape, max_label_length, max_aug_nbr, self.param_dict, name="train")
            if isinstance(sign, str):
                self.load_data = np.load(sign)
                self.local_dataset_path = sign
            else:
                self.data, self.labels, self.labels_length = sign

    def flow(self):
        # Feed inputs and outputs to training generator
        pred_labels_length = np.full((self.batch_size, 1), self.pre_pred_label_length, dtype=np.float64)

        while True:
            # Selected working range randomly.
            working_start_index = np.random.randint(self.data_nbr - self.batch_size)
            working_end_index = working_start_index + self.batch_size
            if self.load_data is not None:
                # The reason why can't read all those data from .npz once for all,
                # is that it will cost a lots memories and slow down the training speed.
                # So clip and read it when it's necessary.
                working_data = self.load_data["data"][working_start_index: working_end_index]
                working_labels = self.load_data["labels"][working_start_index: working_end_index]
                working_labels_length = self.load_data["labels_length"][working_start_index: working_end_index]
            else:
                working_data = self.data[working_start_index: working_end_index]
                working_labels = self.labels[working_start_index: working_end_index]
                working_labels_length = self.labels_length[working_start_index: working_end_index]
            inputs = {
                "y_true": working_labels,
                "img_inputs": working_data,
                "y_pred_length": pred_labels_length,
                "y_true_length": working_labels_length
            }
            outputs = {"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}

            yield (inputs, outputs)
