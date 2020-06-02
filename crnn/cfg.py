# Preprocess
AUG_NBR = 80

SRC_IN_DIR = "../dataset/card_nbr/imgs"
AUG_OUT_DIR = "../dataset/card_nbr/aug_imgs"

TRAIN_TXT_PATH = "../dataset/card_nbr/train.txt"
VAL_TXT_PATH = "../dataset/card_nbr/val.txt"

# Training
NUM_CLASSES = 11  # 0-9 and _
MAX_LABEL_LENGTH = 26
DOWNSAMPLE_FACTOR = 4

MODEL_OUT_DIR = "model/"
IMG_SIZE = (256, 32)
BATCH_SIZE = 8  # depends on your GPU
EPOCH = 100

# CODE
DECODE_DICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '_'}
ENCODE_DICT = {'0': 0, '1': 1, '2':2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '_': 10}
