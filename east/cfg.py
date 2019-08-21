# ------ CAN CHANGE ------

# dataset dir
data_dir = "../dataset/card/"
origin_img_dir_name = "image/"
origin_txt_dir_name = "txt/"
validation_split_ratio = 0.15

# suggest size to be in [256, 384, 512, 640, 736]
size = 384


# ------ BETTER NOT TO CHANGE ------

# dataset
train_imgs_dir_name = "images_for_train/"
train_txts_dir_name = "labels_for_train/"
show_gt_img_dir_name = "show_gt_images/"
show_act_img_dir_name = "show_act_images/"
val_fname = 'val.txt'
train_fname = 'train.txt'
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
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
