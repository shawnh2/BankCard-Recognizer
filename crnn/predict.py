import cv2
import numpy as np
from keras import backend as K

from crnn.cfg import *
from crnn.cnn_blstm_ctc import build_model


def single_recognition(img_array, img_shape: tuple, model_path):
    img_w, img_h = img_shape
    # Image processing.
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, img_shape)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0 * 2.0 - 1.0

    img_batch = np.zeros((1, img_h, img_w, 1))
    img_batch[0, :, :, :] = img_array

    # Model for predict.
    model = build_model(is_training=False)
    model.load_weights(model_path)
    y_pred = model.predict(img_batch)
    y_pred_tensor_list, _ = K.ctc_decode(y_pred, [img_w // DOWNSAMPLE_FACTOR])
    y_pred_tensor = y_pred_tensor_list[0]
    y_pred_labels = K.get_value(y_pred_tensor)
    y_pred_text = ""
    for num in y_pred_labels[0]:
        y_pred_text += DECODE_DICT[num]

    return y_pred_text
