import os
import random

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from east import cfg
from east.utils import reorder_vertexes, shrink, point_inside_of_quad, point_inside_of_nth_quad


data_dir = cfg.data_dir
origin_img_dir = os.path.join(data_dir, cfg.origin_img_dir_name)
origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
train_imgs_dir = os.path.join(data_dir, cfg.train_imgs_dir_name)
train_txts_dir = os.path.join(data_dir, cfg.train_txts_dir_name)
show_gt_img_dir = os.path.join(data_dir, cfg.show_gt_img_dir_name)
show_act_img_dir = os.path.join(data_dir, cfg.show_act_img_dir_name)


def preprocess():
    # Missions:
    # 1. make all the dir that may use.
    # 2. generate train set.
    # 3. generate gt (ground truth) images.

    if not os.path.exists(train_imgs_dir):
        os.mkdir(train_imgs_dir)
    if not os.path.exists(train_txts_dir):
        os.mkdir(train_txts_dir)
    if not os.path.exists(show_gt_img_dir):
        os.mkdir(show_gt_img_dir)
    if not os.path.exists(show_act_img_dir):
        os.mkdir(show_act_img_dir)

    or_img_list = os.listdir(origin_img_dir)
    print("[*] Found %d origin images." % len(or_img_list))
    train_val_set = []
    for or_img_fnm, _ in zip(or_img_list, tqdm(range(len(or_img_list)))):
        or_img_nm, ext = os.path.splitext(or_img_fnm)
        img_path = os.path.join(origin_img_dir, or_img_fnm)
        with Image.open(img_path) as img:
            # resize img and cal scale ratio.
            d_width, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_width / img.width
            scale_ratio_h = d_height / img.height
            img = img.resize((d_width, d_height), Image.NEAREST).convert('RGB')
            show_gt_img = img.copy()
            # draw on the img.
            draw = ImageDraw.Draw(show_gt_img)
            txt_path = os.path.join(origin_txt_dir, or_img_nm + ".txt")
            with open(txt_path, 'r') as f:
                tag_list = f.readlines()
            # this array can save multi-tag labels.
            xy_list_array = np.zeros((len(tag_list), 4, 2))
            # paint gt img from the annotations.
            for anno, i in zip(tag_list, range(len(tag_list))):
                anno_col = anno.strip().split(',')
                anno_array = np.array(anno_col)
                # reshape it to 4*2 and scale it.
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                # make sure the xy_list is in right order.
                # if not, reorder it then store it.
                xy_list = reorder_vertexes(xy_list)
                xy_list_array[i] = xy_list
                _, shrink_xy_list, _ = shrink(xy_list)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                # draw gt image.
                # green line is label edge.
                # blue line is shrinking edge.
                draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                           tuple(xy_list[2]), tuple(xy_list[3]),
                           tuple(xy_list[0])],
                          width=2, fill="green")
                draw.line([tuple(shrink_xy_list[0]),
                           tuple(shrink_xy_list[1]),
                           tuple(shrink_xy_list[2]),
                           tuple(shrink_xy_list[3]),
                           tuple(shrink_xy_list[0])],
                          width=2, fill="blue")
                # yellow is head and foot line.
                vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                      [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                for q_th in range(2):
                    draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                               tuple(shrink_1[vs[long_edge][q_th][1]]),
                               tuple(shrink_1[vs[long_edge][q_th][2]]),
                               tuple(xy_list[vs[long_edge][q_th][3]]),
                               tuple(xy_list[vs[long_edge][q_th][4]])],
                              width=3, fill='yellow')
            # save train img and labels.
            img.save(os.path.join(train_imgs_dir, or_img_fnm))
            np.save(os.path.join(train_txts_dir, or_img_nm + '.npy'), xy_list_array)
            # save gt img.
            show_gt_img.save(os.path.join(show_gt_img_dir, or_img_fnm))
            train_val_set.append('{},{},{}\n'.format(or_img_fnm, d_width, d_height))

    train_img_list = os.listdir(train_imgs_dir)
    print('\nfound %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_txts_dir)
    print('found %d train labels.' % len(train_label_list))

    # split train and val set.
    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


def process_label():
    # Missions:
    # 1. generate act images.
    # 2. generate gt labels in train set.

    # Load stuff that generated from preprocess .
    with open(os.path.join(data_dir, cfg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
    with open(os.path.join(data_dir, cfg.train_fname), 'r') as f_train:
        f_list.extend(f_train.readlines())
    for line, _ in zip(f_list, tqdm(range(len(f_list)))):
        line_cols = str(line).strip().split(',')
        img_name_ext, width, height = line_cols[0].strip(), int(line_cols[1].strip()), int(line_cols[2].strip())
        img_name, _ = os.path.splitext(img_name_ext)
        gt = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        xy_list_array = np.load(os.path.join(train_txts_dir, img_name + '.npy'))
        with Image.open(os.path.join(train_imgs_dir, img_name_ext)) as im:
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3
                i_min = np.maximum(0, ji_min[1])
                i_max = np.minimum(height // cfg.pixel_size, ji_max[1])
                j_min = np.maximum(0, ji_min[0])
                j_max = np.minimum(width // cfg.pixel_size, ji_max[0])
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                            gt[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[i, j, 1] = 1
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]
                            draw.line([(px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size)],
                                      width=line_width, fill=line_color)
            # save act img.
            im.save(os.path.join(show_act_img_dir, img_name_ext))
        # save gt label in train label.
        np.save(os.path.join(train_txts_dir, img_name + '_gt.npy'), gt)


if __name__ == '__main__':
    print("[*] Start preprocess...")
    preprocess()
    process_label()
    print("\n[*] Done.")
