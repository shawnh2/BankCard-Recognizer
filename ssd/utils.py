import os
import random
import xml.etree.ElementTree as ET


def rename(src_dir, total=1084):
    """
    In order to label the image following by VOC2007 format,
    we need to rename it to 00xxxx.png.

    :param src_dir: Input dir that needs to rename.
    :param total: The total number of files in this dir, which is 1085 in this case.
    """
    for file in os.listdir(src_dir):
        _, ext = os.path.splitext(file)
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(src_dir, "%06d" % total + ext)
        os.rename(src_file, dst_file)
        total -= 1

    return 0


def write_xml(src_dir, dst_dir, pattern_path):
    """
    In this case, image only tag one label.
    So use pattern.xml to cover all the image,
    handle it only by changing tag <filename>, <path>.

    :param src_dir: Input image which needs to be tagged.
    :param dst_dir: Output dir where store Annotations(xml) files.
    :param pattern_path: Reference pattern XML file.
    """

    def fill_xml_value(filename, file_abs_path):
        tree = ET.parse(pattern_path)
        root = tree.getroot()
        for e_1 in root.iter("filename"):
            new_elem = filename
            e_1.text = new_elem
        for e_2 in root.iter("path"):
            new_elem = file_abs_path
            e_2.text = new_elem
        return tree

    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)
        name, ext = os.path.splitext(file)
        new_xml = fill_xml_value(name, file_path)
        dst_file_path = os.path.join(dst_dir, name + ".xml")
        new_xml.write(dst_file_path)

    return 0


def write_index(annotations_dir,
                dst_dir,
                trainval_percent=0.7,
                train_percent=0.8):
    """
    Writing image's index in ImageSets/Main.
    For all 4 files, test.txt, train.txt, trainval.txt, val.txt.
    """
    total_xml = os.listdir(annotations_dir)
    num = len(total_xml)

    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(range(num), tv)
    train = random.sample(trainval, tr)

    f_trainval = open(dst_dir + '/trainval.txt', 'w')
    f_test = open(dst_dir + '/test.txt', 'w')
    f_train = open(dst_dir + '/train.txt', 'w')
    f_val = open(dst_dir + '/val.txt', 'w')

    for i in range(num):
        name, ext = os.path.splitext(total_xml[i])
        name += "\n"
        if i in trainval:
            f_trainval.write(name)
            if i in train:
                f_train.write(name)
            else:
                f_val.write(name)
        else:
            f_test.write(name)

    f_trainval.close()
    f_train.close()
    f_val.close()
    f_test.close()
