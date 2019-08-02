"""In data augmentation, this part may be the biggest consumption,
and it takes us a lot of time.
In order to optimize, we compile the code that most like python and numpy to C by cython.
and find it's becomes cheaper and faster than original python code.

By the way, we make a time compare table to show the biggest changes.
We repeat one function 10^N times per epoch, and get avg-time:
    [Function]      [N]     [Time-before]     [Time-after]     [Increasing]
     add_noise       3          0.2326           0.0159           x 14.63
     shear           3          6.9081           3.9274           x 1.75
     fill            3          38.0633          0.0312           x 1219.97
     resize          2          27.9830          0.0249           x 1123.81
     cv2-resize      2                           0.0029           / 85.86      [even worse]
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport tan
from libc.stdint cimport uint8_t
from aug.toolkits import crop


@cython.boundscheck(False)
@cython.wraparound(False)
def add_noise(np.ndarray[uint8_t, ndim=3] img_array,
              double p,
              unsigned int img_h,
              unsigned int img_w):
    """
    Salt & Pepper Noise:
    Add small white and black dot noise in image,
    so it's like salt and pepper.
    The p means the percentage of dots intensity.
    """
    cdef int noise
    # The recommend factor is 0.01-0.02
    # otherwise will effect the true pixel
    noise = int(p * img_w * img_h)

    cdef np.ndarray[np.int_t, ndim=1,
                    negative_indices=False,
                    mode='c'] rand_x = np.random.random_integers(0, img_h-1, noise)
    cdef np.ndarray[np.int_t, ndim=1,
                    negative_indices=False,
                    mode='c'] rand_y = np.random.random_integers(0, img_w-1, noise)
    cdef np.ndarray[np.int_t, ndim=1,
                    negative_indices=False,
                    mode='c'] flags = np.random.random_integers(0, 1, noise)

    for ch in range(3):
        for i in range(noise):
            if flags[i] == 0:
                # The reason why didn't set pure black(0) is
                # one fill mode called 'point'.
                # It fills every black pixel which is zero.
                img_array[rand_x[i], rand_y[i], ch] = 10
            else:
                img_array[rand_x[i], rand_y[i], ch] = 255

    return img_array


@cython.boundscheck(False)
@cython.wraparound(False)
def shear(np.ndarray[uint8_t, ndim=3] img_array,
          unsigned int angle,
          unsigned int img_h,
          unsigned int img_w):
    """
    Shear the img randomly.
    It has two direction: X and Y.
    So it will choose one when call it.
    """
    cdef int new_h, top_gap = 0, bottom_gap = 0
    cdef int new_w, left_gap = 0, right_gap = 0
    cdef int gaps, x, y, nbr
    cdef int i, j, ch
    cdef int flag
    cdef double a, pi = 3.1415926

    a = tan(angle * pi / 180.0)
    flag = np.random.randint(0, 2)
    if flag == 0:
        # shearing on x axis
        new_h = int(img_h + img_w * a)
        new_img = np.zeros((new_h, img_w, 3), dtype=np.uint8)
        for i in range(img_h):
            for j in range(img_w):
                x = int(i + j * a)
                new_img[x, j] = img_array[i, j]
        gaps = new_h - img_h
        top_gap = int(gaps / 2)
        bottom_gap = gaps - top_gap
    else:
        # shearing on y axis
        new_w = int(img_w + img_h * a)
        new_img = np.zeros((img_h, new_w, 3), dtype=np.uint8)
        for i in range(img_h):
            for j in range(img_w):
                y = int((img_h - i) * a + j)
                new_img[i, y] = img_array[i, j]
        gaps = new_w - img_w
        left_gap = int(gaps / 2)
        right_gap = gaps - left_gap

    return crop(new_img, (top_gap, bottom_gap, left_gap, right_gap))


@cython.boundscheck(False)
@cython.wraparound(False)
def fill(np.ndarray[uint8_t, ndim=3] img_array,
         unsigned int img_h,
         unsigned int img_w):
    """
    Fill the edges by nearest pixel.
    Basically by cutting the img from middle height,
    and for the head, go up and feed the pixel below,
    also for the foot part.
    """
    cdef unsigned int middle, p
    cdef uint8_t cur_pix

    middle = int(img_h / 2)

    # Head part
    for ch in range(3):
        for col in range(img_w):
            p = middle
            cur_pix = img_array[p, col, ch]
            p -= 1
            while p != 0:
                if img_array[p, col, ch] == 0:
                    img_array[p, col, ch] = cur_pix
                else:
                    cur_pix = img_array[p, col, ch]
                p -= 1

    # Foot part
    for ch in range(3):
        for col in range(img_w):
            p = middle
            cur_pix = img_array[p, col, ch]
            p += 1
            while p != img_h:
                if img_array[p, col, ch] == 0:
                    img_array[p, col, ch] = cur_pix
                else:
                    cur_pix = img_array[p, col, ch]
                p += 1

    return img_array


@cython.boundscheck(False)
@cython.wraparound(False)
def resize(np.ndarray[uint8_t, ndim=3] img_array,
           unsigned int img_h,
           unsigned int img_w,
           unsigned int dst_h,
           unsigned int dst_w):
    """Inter-Liner Algorithm"""
    cdef double scale_x, scale_y
    cdef double src_x, src_y
    cdef double value_0, value_1
    cdef int src_x_0, src_x_1, src_y_0, src_y_1
    cdef int k, dst_y, dst_x
    cdef np.ndarray[uint8_t, ndim=3] dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)

    scale_x = float(img_w) / float(dst_w)
    scale_y = float(img_h) / float(dst_h)

    for k in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # Original coords
                src_x = (float(dst_x) + 0.5) * scale_x - 0.5
                src_y = (float(dst_y) + 0.5) * scale_y - 0.5

                #INTER_LINEAR: 2*2 neighbors.
                src_x_0 = floor_int(src_x)
                src_y_0 = floor_int(src_y)
                src_x_1 = min(src_x_0 + 1, img_w - 1)
                src_y_1 = min(src_y_0 + 1, img_h - 1)

                value_0 = (src_x_1 - src_x) * img_array[src_y_0, src_x_0, k] + \
                          (src_x - src_x_0) * img_array[src_y_0, src_x_1, k]
                value_1 = (src_x_1 - src_x) * img_array[src_y_1, src_x_0, k] + \
                          (src_x - src_x_0) * img_array[src_y_1, src_x_1, k]
                dst_img[dst_y, dst_x, k] = int((src_y_1 - src_y) * value_0 + (src_y - src_y_0) * value_1)

    return dst_img


cdef int floor_int(double n):
    # decrease the extra functions of floor,
    # focus on double to int type.

    # eg. floor(300.16) ->  300
    #     floor(25.93)  ->  25
    #     floor(-23.11) ->  -24
    #     floor(-0.02)  ->  -1
    if n >= 0.:
        return int(n)
    else:
        return int(n-1)
