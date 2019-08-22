import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D, BatchNormalization

from east import cfg


class EAST:

    def __init__(self):
        self.input_img = Input(shape=(None, None, cfg.num_channels), dtype='float32', name='input_img')
        vgg16 = VGG16(input_tensor=self.input_img, weights='imagenet', include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block_conv1'), vgg16.get_layer('block_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        assert i + self.diff in cfg.feature_layers_range

        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D()(self.h(i))

    def h(self, i):
        assert i + self.diff in cfg.feature_layers_range

        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1, activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3, activation='relu', padding='same')(bn2)
            return conv_3

    def east_network(self):
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_output)
        east_detect = Concatenate(axis=-1, name='east_detect')([inside_score, side_v_code, side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


def quad_loss(y_true, y_pred):
    # loss for inside_score
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    # balance positive and negative samples in an image
    beta = 1 - tf.reduce_mean(labels)
    # first apply sigmoid activation
    predicts = tf.nn.sigmoid(logits)
    # log +epsilon for stable cal
    inside_score_loss = tf.reduce_mean(
        -1 * (beta * labels * tf.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * tf.log(1 - predicts + cfg.epsilon)))
    inside_score_loss *= cfg.lambda_inside_score_loss

    # loss for side_vertex_code
    vertex_logits = y_pred[:, :, :, 1:3]
    vertex_labels = y_true[:, :, :, 1:3]
    vertex_beta = 1 - (tf.reduce_mean(y_true[:, :, :, 1:2])
                       / (tf.reduce_mean(labels) + cfg.epsilon))
    vertex_predicts = tf.nn.sigmoid(vertex_logits)
    pos = -1 * vertex_beta * vertex_labels * tf.log(vertex_predicts +
                                                    cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * tf.log(
        1 - vertex_predicts + cfg.epsilon)
    positive_weights = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
    side_vertex_code_loss = \
        tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * positive_weights) / (
                tf.reduce_sum(positive_weights) + cfg.epsilon)
    side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, :, :, 3:]
    g_true = y_true[:, :, :, 3:]
    vertex_weights = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
    side_vertex_coord_loss = tf.reduce_sum(pixel_wise_smooth_l1norm) / (
            tf.reduce_sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss
    return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss


def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    pixel_wise_smooth_l1norm = (tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        axis=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    shape = tf.shape(g_true)
    delta_xy_matrix = tf.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(square, axis=-1))
    distance *= 4.0
    distance += cfg.epsilon
    return tf.reshape(distance, shape[:-1])
