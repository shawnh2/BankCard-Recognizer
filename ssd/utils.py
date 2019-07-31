import os


def rename(dir, total=1084):
    """
    In order to label the image following by VOC2007 format,
    we need to rename it to 00xxxx.png.

    :param dir: Input dir that needs to rename.
    :param total: The total number of files in this dir, which is 1085 in this case.
    """
    for file in os.listdir(dir):
        _, ext = os.path.splitext(file)
        src_file = os.path.join(dir, file)
        dst_file = os.path.join(dir, "%06d" % total + ext)
        os.rename(src_file, dst_file)
        total -= 1

if __name__ == '__main__':
    # rename(r"\ssd\VOC2007\JPEGImages")
    pass
