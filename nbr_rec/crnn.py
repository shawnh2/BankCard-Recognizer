import os

from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Activation, BatchNormalization,
                          Permute, TimeDistributed, Flatten, Bidirectional, LSTM, Dense, Lambda)

from nbr_rec.utils import DataGenerator


class CRNN:
    """Based on frame model: VGG_BLSTM_CTC"""

    @staticmethod
    def build(img_wh_shape, nbr_classes, max_label_length):
        # Annotations assume the img_wh_shape is (256, 32)

        initializer = initializers.he_normal()
        img_w, img_h = img_wh_shape

        inputs = Input(shape=(img_h, img_w, 1), name="img_inputs")  # 32*256*1
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_1")(inputs)  # 32*256*64
        x = BatchNormalization(name="bn_1")(x)
        x = Activation("relu", name="relu_1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid",name="maxpool_1")(x)  # 16*128*64

        x = Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_2")(x)  # 16*128*128
        x = BatchNormalization(name="bn_2")(x)
        x = Activation("relu", name="relu_2")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="maxpool_2")(x)  # 8*64*128

        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_3")(x)  # 8*64*256
        x = BatchNormalization(name="bn_3")(x)
        x = Activation("relu", name="relu_3")(x)
        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_4")(x)  # 8*64*256
        x = BatchNormalization(name="bn_4")(x)
        x = Activation("relu", name="relu_4")(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="maxpool_3")(x)  # 4*64*256

        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_5")(x)  # 4*64*512
        x = BatchNormalization(axis=-1, name="bn_5")(x)
        x = Activation("relu", name="relu_5")(x)
        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name="conv2d_6")(x)  # 4*64*512
        x = BatchNormalization(axis=-1, name="bn_6")(x)
        x = Activation("relu", name="relu_6")(x)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="maxpool_4")(x)  # 2*64*512

        x = Conv2D(512, (2, 2), padding="same", kernel_initializer=initializer, name="conv2d_7")(x)
        x = BatchNormalization(name="bn_7")(x)
        x = Activation("relu", name="relu_7")(x)
        conv_output = MaxPooling2D(pool_size=(2, 1), name="conv_output")(x)  # 1*64*512
        x = Permute((2, 3, 1), name="permute")(conv_output)  # 64*512*1
        rnn_input = TimeDistributed(Flatten(), name="flatten")(x)  # 64*512

        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True),
                          merge_mode="sum", name="lstm_1")(rnn_input)  # 64*512
        y = BatchNormalization(name="bn_8")(y)
        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name="lstm_2")(y)  # 64*512

        y_pred = Dense(nbr_classes, activation="softmax", name="y_pred")(y)
        # base_model = Model(inputs=inputs, outputs=y_pred)
        # base_model.summary()
        y_true = Input(shape=[max_label_length], name="y_true")
        y_pred_length = Input(shape=[1], name="y_pred_length")
        y_true_length = Input(shape=[1], name="y_true_length")
        ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1, ),
                                 name="ctc_loss_output")([y_true, y_pred, y_pred_length, y_true_length])
        model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
        model.summary()

        return model

    @staticmethod
    def train(model, train_dir, val_dir, weight_save_path,
              img_size, batch_size, max_label_length, down_sample_factor, epochs):
        print("[*]Training start...")
        model.compile(optimizer='adam', loss={'ctc_loss_output': fake_ctc_loss})
        train_gen = DataGenerator(train_dir, img_size, batch_size, down_sample_factor, max_label_length)
        val_gen = DataGenerator(val_dir, img_size, batch_size, down_sample_factor, max_label_length)
        print("[*]Finished initialize compile and generator...")
        print("[!]Starting training now!")
        model.fit_generator(generator=train_gen.get_data(),
                            steps_per_epoch=train_gen.img_nbr // batch_size,
                            validation_data=val_gen.get_data(),
                            validation_steps=val_gen.img_nbr // batch_size,
                            epochs=epochs)
        print("[*]Finished training...")
        model_name = os.path.join(weight_save_path, "model.h5")
        model.save(model_name)
        print("[*]Model weights has been saved in %s successfully!" % weight_save_path)


def ctc_loss_layer(args):
    """
    :param args: (y_true, y_pred, pred_length, label_length)
    :return: ctc_batch_cost
    """
    y_true, y_pred, pred_length, label_length = args
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
    return batch_cost


def fake_ctc_loss(y_true, y_pred):
    """
    This function is aimed at fitting with Keras compiler,
    the parameters should only be y_true nd y_pred.
    After, co-work with our ctc_loss_layer.
    """
    return y_pred
