from keras import callbacks
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
        # Annotations assume the img_wh_shape is (120, 46)

        initializer = initializers.he_normal()
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

        x = Conv2D(512, (2, 2), padding="same", kernel_regularizer=initializer, name="conv2d_7")(x)
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

    @staticmethod
    def train(model, train_dir, val_dir, weight_save_path,
              img_size, batch_size, max_label_length, down_sample_factor, epochs):
        print("[*]Training start...")

        checkpoint = callbacks.ModelCheckpoint(weight_save_path +
                                               "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                               monitor='val_loss',
                                               save_weights_only=True,
                                               save_best_only=True)
        reduce_lr_callback = callbacks.ReduceLROnPlateau(patience=3)
        logging = callbacks.TensorBoard(log_dir=weight_save_path)
        print("[*]Finished setting up with callbacks...")

        model.compile(optimizer='adam', loss={'ctc_loss_output': fake_ctc_loss})
        train_gen = DataGenerator(train_dir, img_size, batch_size, down_sample_factor, max_label_length)
        val_gen = DataGenerator(val_dir, img_size, batch_size, down_sample_factor, max_label_length)
        print("[*]Finished initialize compile and generator...")
        print("[!]Starting training now!")
        model.fit_generator(generator=train_gen.get_data(),
                            steps_per_epoch=train_gen.img_nbr // batch_size,
                            validation_data=val_gen.get_data(),
                            validation_steps=val_gen.img_nbr // batch_size,
                            callbacks=[checkpoint, reduce_lr_callback, logging],
                            epochs=epochs)
        print("[*]Finished training...")
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