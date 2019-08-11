import cv2
import numpy as np
from keras import backend as K
from crnn.cnn_blstm_ctc import CNN_BLSTM_CTC

num2char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '_'}


def single_recognition(img, img_shape, model_path,
                       num_classes,
                       max_label_length,
                       downsample_factor=4):
    img_w, img_h = img_shape
    # Image processing
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_shape)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0 * 2.0 - 1.0

    img_batch = np.zeros((1, img_h, img_w, 1))
    img_batch[0, :, :, :] = img

    # Model for predict
    model = CNN_BLSTM_CTC.build(img_shape, num_classes, max_label_length, is_training=False)
    model.load_weights(model_path)
    y_pred = model.predict(img_batch)
    y_pred_tensor_list, _ = K.ctc_decode(y_pred, [img_w // downsample_factor])
    y_pred_tensor = y_pred_tensor_list[0]
    y_pred_labels = K.get_value(y_pred_tensor)
    y_pred_text = ""
    for num in y_pred_labels[0]:
        y_pred_text += num2char[num]

    return y_pred_text
