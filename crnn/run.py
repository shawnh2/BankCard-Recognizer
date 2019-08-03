"""
Training Report:
(recording few different changes if you change the parameters here)

*img_size & aug_number*
It directly connect to the size of wrapped-data(.npz) if you do data augmentation.
For example, size(256, 32) with 80 aug is about 4GB, but size(120, 46) with 50 aug is about 1GB only.
So pick up these parameter carefully, you don't want your disk full of these giant.
But good news is there is a alert after training process, you can choose save them or not.

*epochs & batch_size & steps_per_epoch*
After data augmentation, the dataset become larger than before.
So every epoch will fetch N(batch_size*steps_per_epoch) data to train,
of course N is very huge sometimes, for saving the training resources,
we can decrease steps_per_epoch or batch_size and increase epochs to compromise.

[This model has been tested on GPU: NVIDIA GTX 1050]
If you have better GPU that this, you can change parameters above as you want.
It may bring much more efficient. Wish you luck :)
"""

import os
from crnn.cnn_blstm_ctc import CNN_BLSTM_CTC as CBC

# Ignore the log messages which level is under 2: TIPS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def main():
    model_save_path = "model/"
    img_size = (120, 46)
    num_classes = 11
    max_label_length = 26
    aug_number = 50
    epochs = 100

    model = CBC.build(img_size, num_classes, max_label_length)
    CBC.train(model,
              src_dir="../dataset/imgs/",
              save_dir=model_save_path,
              img_size=img_size,
              batch_size=16,
              aug_nbr=aug_number,
              max_label_length=max_label_length,
              down_sample_factor=4,
              epochs=epochs)


if __name__ == '__main__':
    main()
