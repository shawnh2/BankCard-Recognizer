import os

from keras import callbacks

from crnn.cfg import *
from crnn.cnn_blstm_ctc import build_model
from crnn.utils import DataGenerator, fake_ctc_loss

# Ignore the log messages which level is under 2: TIPS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def run():
    model = build_model(is_training=True)
    ckpt = callbacks.ModelCheckpoint(MODEL_OUT_DIR + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                     save_weights_only=True, save_best_only=True)
    reduce_lr_cbk = callbacks.ReduceLROnPlateau(patience=3)
    model.compile(optimizer='adam', loss={'ctc_loss_output': fake_ctc_loss})

    train_gen = DataGenerator(TRAIN_TXT_PATH)
    val_gen = DataGenerator(VAL_TXT_PATH)
    print("[*] Training start!")
    model.fit_generator(generator=train_gen.flow(),
                        steps_per_epoch=train_gen.data_nbr // BATCH_SIZE,
                        validation_data=val_gen.flow(),
                        validation_steps=val_gen.data_nbr // BATCH_SIZE,
                        callbacks=[ckpt, reduce_lr_cbk],
                        epochs=EPOCH)
    print("[*] Training finished!")
    model.save(MODEL_OUT_DIR + "crnn_model.h5")
    print("[*] Model has been successfully saved in %s!" % MODEL_OUT_DIR)


if __name__ == '__main__':
    run()
