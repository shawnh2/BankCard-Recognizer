# Fixed length for detecting number of a bankcard
# Default backend format of image is channel_last
import os
import time

import cv2
import numpy as np
from keras.models import Model
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam

from bankcard_rec.nbr_detc.utils import ImageTextGenerator


# Essential Constant
CHARS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_')
DIGIT = 4
IMG_SHAPE = (46, 120, 3)
IMG_H, IMG_W, IMG_C = IMG_SHAPE
# Part-in Dirs
INPUTS_DIR = r'X:\projects\bankcard_rec\img_src\imgs'
OUTPUT_DIR = r'X:\projects\bankcard_rec\nbr_detc'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
# Training Parameters
# After split train and test dataset,
# It all has 856 as Train, 214 as Val.
BATCH_SIZE = 80
EPOCHS = 10


class FixedRecCNN:

    @staticmethod
    def build():
        input_tensor = Input(IMG_SHAPE)
        x = input_tensor
        for i in range(3):
            x = Conv2D(16*2**i, (3, 3), activation='relu')(x)
            x = Conv2D(16*2**i, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)

        n_class = len(CHARS)
        x = [Dense(n_class, activation='softmax', name='d_%d' % (i+1))(x) for i in range(DIGIT)]
        model = Model(inputs=input_tensor, outputs=x)
        model.summary()
        return model

    @staticmethod
    def train():
        print("[Info]: Building and compiling model.")
        adam = Adam(lr=0.001)
        model = FixedRecCNN.build()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

        print("[Info]: Preparing the data generator.")
        train_gen = ImageTextGenerator(TRAIN_DIR,
                                       width_shift_range=15,
                                       height_shift_range=5,
                                       zoom_range=7,
                                       shear_range=15,
                                       rotation_range=12,
                                       blur_factor=2,
                                       noise_factor=0.005)
        val_gen = ImageTextGenerator(VAL_DIR)

        print("[Info]: Start training.")
        model.fit_generator(train_gen.flow(BATCH_SIZE),
                            steps_per_epoch=856,
                            validation_data=val_gen.flow(BATCH_SIZE),
                            validation_steps=214,
                            epochs=EPOCHS)

        print("[Info]: Saving model.")
        model_name = 'nbr_detc_{}.h5'.format(time.strftime("%m%d%H%M", time.localtime()))
        model.save(os.path.join(OUTPUT_DIR, model_name))
        print("[Info]: Done.")

    @staticmethod
    def predict(model_path, img_path):
        model = models.load_model(model_path)
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        img /= 255
        img = np.expand_dims(img, axis=0)
        titles = list(map(lambda x: "".join([CHARS[xx] for xx in x]),
                          np.argmax(np.array(model.predict(img)), 2).T))
        print(titles)


if __name__ == '__main__':
    FixedRecCNN.train()
    #model_path = r'X:\projects\nbr_detc\test1.h5'
    #test_img = r''
    #FixedRecCNN.predict(model_path, test_img)
