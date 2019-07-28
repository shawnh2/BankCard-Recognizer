"""The easiest and most common method to reduce over-fitting on image data
is to artificially enlarge the dataset.
And in this case, we came up with a way to preprocess the image by using
some simple transformation in data augmentation.

The parameters that you shall give to the functions:
 width_shift_range: int              |  Input: img_array, factor     |  Output: img_array
height_shift_range: int              |  Input: img_array, factor     |  Output: img_array
              zoom: int              |  Input: img_array, factor     |  Output: img_array
             shear: int              |  Input: img_array, factor     |  Output: img_array *
            rotate: int              |  Input: img_array, factor     |  Output: img_array
              blur: int ~5           |  Input: img_array, factor     |  Output: img_array
         add_noise: float 0.01-0.03  |  Input: img_array, factor     |  Output: img_array *
   horizontal_flip: bool             |  Input: img_array             |  Output: img_array
     vertical_flip: bool             |  Input: img_array             |  Output: img_array
              fill: bool             |  Input: img_array             |  Output: img_array *
"""
import random
import datetime

from aug.toolkits import (width_shift_range, height_shift_range, zoom,
                          rotate, horizontal_flip, vertical_flip, blur)
from aug.ctoolkits import add_noise, shear, fill

random.seed(datetime.datetime.now())
# In these augmentation, not all of them should be involved in every transformation.
# So give them priority or orders to perform themselves.
# Using abbreviation code to represent.


class DataAugmentation:

    def __init__(self, img_array, param_dict, min_occur_ratio=0.05):
        self.img_array = img_array
        self.img_h, self.img_w, _ = img_array.shape
        self.ratio = min_occur_ratio

        # saving parameters locally with its code
        self.height_shift = param_dict['hsr']
        self.width_shift = param_dict['wsr']
        self.rotate = param_dict['ror']
        self.h_flip = param_dict['hfl']
        self.v_flip = param_dict['vfl']
        self.shear = param_dict['shr']
        self.noise = param_dict['nof']
        self.zoom = param_dict['zor']
        self.blur = param_dict['blr']

    def feed(self, batch=32):
        for _ in range(batch):
            img = self.img_array

            # -------------------------------------
            # These are transformation that should be
            # involved in every epochs.
            # Every random volume is been generated here.

            if self.width_shift != 0:
                shift_l = random.randint(0, self.width_shift)
                shift_r = random.randint(0, self.width_shift)
                img = width_shift_range(img, shift_l, shift_r)
            if self.height_shift != 0:
                shift_t = random.randint(0, self.height_shift)
                shift_b = random.randint(0, self.height_shift)
                img = height_shift_range(img, shift_t, shift_b)
            if self.zoom != 0:
                to = random.randint(0, self.zoom)
                bo = random.randint(0, self.zoom)
                le = random.randint(0, self.zoom)
                ri = random.randint(0, self.zoom)
                img = zoom(img, (to, bo, le, ri))
            if self.rotate != 0:
                angle = random.randint(-self.rotate, self.rotate)
                img = rotate(img, angle)
            if self.shear != 0:
                shear_factor = random.randint(0, self.shear)
                img = shear(img, shear_factor, self.img_h, self.img_w)
                img = fill(img, self.img_h, self.img_w)

            # -------------------------------------
            # These are transformation that may not be
            # involved in every epochs.
            # Cause it's happened randomly,
            # so give it a probability to happen.

            if self.noise != 0 and self.random_occur(self.ratio):
                img = add_noise(img, self.noise, self.img_h, self.img_w)
            if self.h_flip and self.random_occur():
                img = horizontal_flip(img)
            if self.v_flip and self.random_occur():
                img = vertical_flip(img)
            if self.blur and self.random_occur(self.ratio):
                blur_factor = random.randint(0, self.blur)
                img = blur(img, blur_factor)

            yield img

    def random_occur(self, ratio=0.2):
        assert 0 < ratio < 1
        m = int(1 / ratio)
        return True if random.randint(0, m) == m else False
