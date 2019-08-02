import cv2
import numpy as np
from scipy import ndimage


def width_shift_range(img_array, shift_l, shift_r):
    """
    Shifting the width of a image.
    Basically by cropping the width of an image,
    and then resizing it to original shape.
    """
    img_h, img_w, _ = img_array.shape
    crop_range = (0, 0, shift_l, shift_r)
    new_img = crop(img_array, crop_range)
    return cv2.resize(new_img, (img_w, img_h))


def height_shift_range(img_array, shift_t, shift_b):
    """
    Shifting the height of a image.
    Basically by cropping the height of an image,
    and then resizing it to original shape.
    """
    img_h, img_w, _ = img_array.shape
    crop_range = (shift_t, shift_b, 0, 0)
    new_img = crop(img_array, crop_range)
    return cv2.resize(new_img, (img_w, img_h))


def crop(img_array, x):
    """
    Crop image.
    the pixel in x with following orders:
      (from_top, from_bottom, from_left, from_right)
    The value is start from the img bounds.
    It could not beyond the shape of image.

    EG. An img with shape (46, 120, 3)
        x = (10, 20, 30, 40)
        return shape (16, 50, 3)
    """
    assert len(x) == 4
    i_h, i_w, i_c = img_array.shape
    top, bottom, left, right = x
    assert 0 <= top
    assert 0 <= left
    assert 0 <= bottom
    assert 0 <= right
    assert i_h > top + bottom
    assert i_w > left + right

    return img_array[top: i_h-bottom, left: i_w-right]


def zoom(img_array, zoom_ranges):
    """
    Zoom in the image.
    Basically by using crop to cut the edges,
    and fill the edges by nearest pixel.
    """
    img_h, img_w, _ = img_array.shape
    # giving four different zoom ranges
    temp_img = crop(img_array, zoom_ranges)
    return cv2.resize(temp_img, (img_w, img_h))


def rotate(img_array, angle):
    return ndimage.rotate(img_array, angle, reshape=False, mode='nearest')


def img_function(img_array, function):
    """
    Functions for image array.

    Example: function: lambda img: 255 - img (reverse image)
                       lambda img: (100.0/255) * img + 100 (zip pixel from 100-200)"""

    im = np.array(img_array)
    return function(im)


def blur(img_array, blur_factor):
    """
    Blur the image by Gaussian Blur,
    and it has effects on every channel.
    Recommend: int around 5.
    """
    for i in range(3):
        img_array[:, :, i] = ndimage.gaussian_filter(img_array[:, :, i], blur_factor)
    return np.uint8(img_array)


def horizontal_flip(img_array):
    """Flip image horizontally."""
    img_array = cv2.flip(img_array, 1)
    return img_array


def vertical_flip(img_array):
    """Flip image vertically."""
    img_array = cv2.flip(img_array, 0)
    return img_array
