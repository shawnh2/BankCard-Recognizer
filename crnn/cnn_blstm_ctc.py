from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Activation, Conv2D, MaxPooling2D,
                          Permute, Dense, LSTM, Lambda, TimeDistributed, Flatten, Bidirectional)

from crnn.cfg import *


def ctc_loss_layer(args):
    labels, y_pred, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(is_training: bool = True):
    initializer = initializers.he_normal()
    img_width, img_height = IMG_SIZE

    # CNN(VGG)
    inputs = Input(shape=(img_height, img_width, 1), name='img_inputs')
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool1')(x)

    x = Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2')(x)

    x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv4')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool3')(x)

    x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='conv5')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='conv6')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool4')(x)

    x = Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=initializer, name='conv7')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    conv_out = MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)

    # CNN to RNN
    x = Permute((2, 3, 1))(conv_out)
    rnn_input = TimeDistributed(Flatten())(x)

    # RNN
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True),
                      merge_mode='sum', name='LSTM_1')(rnn_input)
    y = BatchNormalization()(y)
    y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)

    y_pred = Dense(NUM_CLASSES, activation='softmax', name='y_pred')(y)
    labels = Input(shape=[MAX_LABEL_LENGTH], name='labels')
    input_length = Input(shape=[1], name='input_length')
    label_length = Input(shape=[1], name='label_length')
    ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
        [labels, y_pred, input_length, label_length]
    )

    if is_training:
        model = Model(inputs=[labels, inputs, input_length, label_length], outputs=ctc_loss_output)
        # model.summary()
        return model
    else:
        base_model = Model(inputs=inputs, outputs=y_pred)
        # base_model.summary()
        return base_model
