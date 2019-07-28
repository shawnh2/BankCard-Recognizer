import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Activation, BatchNormalization,
                          Permute, TimeDistributed, Flatten, Bidirectional, LSTM, Dense, Lambda)


class CRNN:
    """Based on frame model: VGG_BLSTM_CTC"""

    @staticmethod
    def build(img_wh_shape=(120, 46), nbr_classes=11, max_label_length=26):

        initializer = keras.initializers.he_normal()
        img_w, img_h = img_wh_shape

        inputs = Input(shape=(img_h, img_w, 1), name="img_inputs")  # 46*120*1
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_1")(inputs)  # 46*120*64
        x = BatchNormalization(name="bn_1")(x)
        x = Activation("relu", name="relu_1")(x)
        x = MaxPooling2D(strides=2, name="maxpool_1")(x)  # 23*60*64

        x = Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_2")(x)  # 23*60*128
        x = BatchNormalization(name="bn_2")(x)
        x = Activation("relu", name="relu_2")(x)
        x = MaxPooling2D(strides=2, name="maxpool_2")(x)  # 12*30*128

        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_3")(x)  # 12*30*256
        x = BatchNormalization(name="bn_3")(x)
        x = Activation("relu", name="relu_3")(x)
        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_4")(x)  # 12*30*256
        x = BatchNormalization(name="bn_4")(x)
        x = Activation("relu", name="relu_4")(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="maxpool_3")(x)  # 6*30*256

        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_5")(x)  # 6*30*512
        x = BatchNormalization(name="bn_5")(x)
        x = Activation("relu", name="relu_5")(x)
        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_6")(x)  # 6*30*512
        x = BatchNormalization(name="bn_6")(x)
        x = Activation("relu", name="relu_6")(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="maxpool_4")(x)  # 3*30*512

        x = Conv2D(512, (2, 2), padding="same", activation="relu", kernel_regularizer=initializer, name="conv2d_7")(x)
        x = BatchNormalization(name="bn_7")(x)
        x = Activation("relu", name="relu_7")(x)
        conv_output = MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)  # 1*30*512
        x = Permute((2, 3, 1), name="permute")(conv_output)  # 30*512*1
        rnn_input = TimeDistributed(Flatten(), name="flatten")(x)  # 30*512

        y = Bidirectional(LSTM(256, kernel_regularizer=initializer, return_sequences=True),
                          merge_mode="sum", name="lstm_1")(rnn_input)  # 30*512
        y = BatchNormalization(name="bn_8")(y)
        y = Bidirectional(LSTM(256, kernel_regularizer=initializer, return_sequences=True), name="lstm_2")(y)  # 30*512

        y_pred = Dense(nbr_classes, activation="softmax", name="y_pred")(y)
        # base_model = Model(inputs=inputs, outputs=y_pred)
        # base_model.summary()
        y_true = Input(shape=[max_label_length], name="y_true")
        y_pred_length = Input(shape=[1], name="y_pred_length")
        y_true_length = Input(shape=[1], name="y_true_length")
        ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1, ),
                                 name="ctc_loss_output")([y_true, y_pred, y_pred_length, y_true_length])
        model = Model(inputs=[y_true, y_pred, y_pred_length, y_true_length], outputs=ctc_loss_output)
        model.summary()

        return model


def ctc_loss_layer(args):
    """
    :param args: (y_true, y_pred, pred_length, label_length)
    :return: ctc_batch_cost
    """
    y_true, y_pred, pred_length, label_length = args
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
    return batch_cost
