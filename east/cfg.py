import os

# ------ CAN CHANGE ------

# dataset dir
data_dir = "../dataset/card/"
origin_img_dir_name = "image/"
origin_txt_dir_name = "txt/"
validation_split_ratio = 0.15

# suggest size to be in [256, 384, 512, 640, 736]
size = 384
# img size
max_train_img_size = int(size)
max_predict_img_size = int(size)
assert max_train_img_size in [256, 384, 512, 640, 736]
# batch size based on img size
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
# whether load weights
load_weights = False

# please assign your total img nbr
total_img = 644
# parameters for training
epoch_nbr = 24
initial_epoch = 0
patience = 5
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size


# ------ BETTER NOT TO CHANGE ------

# dataset
train_imgs_dir_name = "images_for_train/"
train_txts_dir_name = "labels_for_train/"
show_gt_img_dir_name = "show_gt_images/"
show_act_img_dir_name = "show_act_images/"
val_fname = 'val.txt'
train_fname = 'train.txt'

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

# east model
locked_layers = False
num_channels = 3
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]
# loss function
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

# model filepath
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model_weights'):
    os.mkdir('saved_model_weights')
model_weights_path = 'saved_model_weights/weights.{epoch:03d}-{val_loss:.3f}.h5'
saved_model_file_path = 'model/east_model.h5'
saved_model_weights_file_path = 'model/east_model_weights.h5'

# for nms
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
# for predict
pixel_threshold = 0.9
