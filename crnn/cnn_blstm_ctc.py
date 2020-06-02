from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Activation, Conv2D, MaxPooling2D, Dropout,
                          Permute, Dense, LSTM, Lambda, TimeDistributed, Flatten, Bidirectional)

from crnn.cfg import *


def ctc_loss_layer(args):
    """
    y_true: True label.
    y_pred: Predict label.
    pred_length: Predict label length.
    label_length: True label length.

    :param args: (y_true, y_pred, pred_length, label_length).
    :return: batch_cost with shape (batch_size, 1).
    """

    y_true, y_pred, pred_length, label_length = args
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
    return batch_cost


def build_model(is_training: bool = True):
    initializer = initializers.he_normal()
    img_width, img_height = IMG_SIZE

    # block 1
    inputs = Input(shape=(img_height, img_width, 1), name='img_inputs')
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_1')(inputs)
    x = BatchNormalization(name="BN_1")(x)
    x = Activation("relu", name="RELU_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='Maxpool_1')(x)
    # block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_2')(x)
    x = BatchNormalization(name="BN_2")(x)
    x = Activation("relu", name="RELU_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='Maxpool_2')(x)
    # block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_3')(x)
    x = BatchNormalization(name="BN_3")(x)
    x = Activation("relu", name="RELU_3")(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_4')(x)
    x = BatchNormalization(name="BN_4")(x)
    x = Activation("relu", name="RELU_4")(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='Maxpool_3')(x)
    # block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_5')(x)
    x = BatchNormalization(name="BN_5")(x)
    x = Activation("relu", name="RELU_5")(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", use_bias=True,
               kernel_initializer=initializer, name='Conv2d_6')(x)
    x = BatchNormalization(name="BN_6")(x)
    x = Activation("relu", name="RELU_6")(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='Maxpool_4')(x)
    # block 5
    x = Conv2D(512, (2, 2), strides=(1, 1), padding='same', use_bias=True,
               activation='relu', kernel_initializer=initializer, name='Conv2d_7')(x)
    x = BatchNormalization(name="BN_7")(x)
    x = Activation("relu", name="RELU_7")(x)
    conv_out = MaxPooling2D(pool_size=(2, 1), name="Conv_output")(x)
    # flatten to seq
    x = Permute((2, 3, 1), name='Permute')(conv_out)
    rnn_input = TimeDistributed(Flatten(), name='Flatten_by_time')(x)
    # rnn
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True),
                      merge_mode='sum', name='LSTM_1')(rnn_input)
    y = BatchNormalization(name='BN_8')(y)
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)
    # out
    y_pred = Dense(NUM_CLASSES + 1, activation='softmax', name='y_pred')(y)  # num_classes and one for empty
    y_true = Input(shape=[MAX_LABEL_LENGTH], name='y_true')
    y_pred_length = Input(shape=[1], name='y_pred_length')
    y_true_length = Input(shape=[1], name='y_true_length')
    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [y_true, y_pred, y_pred_length, y_true_length]
    )

    base_model = Model(inputs=inputs, outputs=y_pred)
    # base_model.summary()
    model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
    # model.summary()

    if is_training:
        return model
    else:
        return base_model
