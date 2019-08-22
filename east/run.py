import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from east import cfg
from east.utils import data_generator
from east.eastnet import EAST, quad_loss

if __name__ == '__main__':
    east = EAST()
    east_network = east.east_network()
    east_network.summary()
    east_network.compile(loss=quad_loss, optimizer=Adam(lr=1e-3, decay=5e-4))
    if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
        east_network.load_weights(cfg.saved_model_weights_file_path)

    east_network.fit_generator(generator=data_generator(),
                               steps_per_epoch=cfg.steps_per_epoch,
                               epochs=cfg.epoch_nbr,
                               validation_data=data_generator(is_val=True),
                               validation_steps=cfg.validation_steps,
                               verbose=1,
                               initial_epoch=cfg.initial_epoch,
                               callbacks=[EarlyStopping(patience=cfg.patience, verbose=1),
                                          ModelCheckpoint(filepath=cfg.model_weights_path,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          verbose=1)])
    east_network.save(cfg.saved_model_file_path)
    east_network.save_weights(cfg.saved_model_weights_file_path)
