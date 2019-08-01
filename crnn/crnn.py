from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.layers import (Lambda, Dense, Bidirectional, Flatten, TimeDistributed, Permute,
                          Activation, Input, LSTM, Conv2D, MaxPooling2D, BatchNormalization)


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


class CNN_BLSTM_CTC:

    @staticmethod
    def build(img_size=(256, 32), num_classes=11, max_label_length=26):
        initializer = initializers.he_normal()
        img_width, img_height = img_size

        def PatternUnits(inputs, index, activation="relu"):
            """
            A pattern unit with both BatchNormalization and Activation.

            :param inputs: The previous inputs.
            :param index: The index that used in name.
            :param activation: Activation method, default is relu.
            :return: The outputs.
            """
            inputs = BatchNormalization(name="BN_%d" % index)(inputs)
            inputs = Activation(activation, name="relu_%d" % index)(inputs)

            return inputs

        inputs = Input(shape=(img_height, img_width, 1), name='img_inputs')
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_1')(inputs)
        x = PatternUnits(x, 1)
        x = MaxPooling2D(strides=2, name='maxpl_1')(x)
        x = Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_2')(x)
        x = PatternUnits(x, 2)
        x = MaxPooling2D(strides=2, name='maxpl_2')(x)

        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_3')(x)
        x = PatternUnits(x, 3)
        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_4')(x)
        x = PatternUnits(x, 4)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpl_3')(x)

        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_5')(x)
        x = PatternUnits(x, 5)
        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='conv2d_6')(x)
        x = PatternUnits(x, 6)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpl_4')(x)

        x = Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=initializer, name='conv2d_7')(x)
        x = PatternUnits(x, 7)
        conv_output = MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)
        x = Permute((2, 3, 1), name='permute')(conv_output)

        rnn_input = TimeDistributed(Flatten(), name='for_flatten_by_time')(x)
        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True),
                          merge_mode='sum', name='LSTM_1')(rnn_input)
        y = BatchNormalization(name='BN_8')(y)
        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)

        y_pred = Dense(num_classes, activation='softmax', name='y_pred')(y)
        y_true = Input(shape=[max_label_length], name='y_true')
        y_pred_length = Input(shape=[1], name='y_pred_length')
        y_true_length = Input(shape=[1], name='y_true_length')
        ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(
            [y_true, y_pred, y_pred_length, y_true_length])

        model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
        model.summary()

        return model


if __name__ == '__main__':
    CNN_BLSTM_CTC.build()
