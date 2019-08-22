import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def array_to_pixmap(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    y, x, _ = img_array.shape
    bytes_per_line = 3 * x
    frame = QImage(img_array.data, x, y, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(frame)
    return pix


def selected_box(img_array, x0, y0, x1, y1):
    """Return the selected area on image."""
    x = abs(x0 - x1)
    y = abs(y0 - y1)
    x_start = min(x0, x1)
    y_start = min(y0, y1)
    return img_array[y_start: y_start + y, x_start: x_start + x, :]


def max_suitable_shape(x, y, limit_x, limit_y):
    """
    Scale the image to fit in canvas.
    If original img is bigger than canvas, follow maximum shrink factor.
    If original img is smaller than canvas, follow minimum grow factor.
    If original img is equal to canvas, then continue.
    """
    if x > limit_x or y > limit_y:
        alpha1 = x / limit_x
        alpha2 = y / limit_y
        factor = max(alpha1, alpha2)
        return x / factor, y / factor
    elif x < limit_x and y < limit_y:
        alpha1 = limit_x / x
        alpha2 = limit_y / y
        factor = min(alpha1, alpha2)
        return x * factor, y * factor
    else:
        return x, y


def rotate_bound(img_array, angle, scale=1.):
    w = img_array.shape[1]
    h = img_array.shape[0]
    # Angle in radians
    r_angle = np.deg2rad(angle)
    # Calculate new image width and height
    nw = (abs(np.sin(r_angle) * h) + abs(np.cos(r_angle) * w)) * scale
    nh = (abs(np.cos(r_angle) * h) + abs(np.sin(r_angle) * w)) * scale
    # Ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # Calculate the move from the old center to the new center combined with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # The move only affects the translation, so update the translation part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    new_img_array = cv2.warpAffine(img_array, rot_mat, (int(np.ceil(nw)), int(np.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    return new_img_array


def hard_coords(coords: list):
    """ Make coords to int and both width and height are equal. """
    coords_array = np.array(coords, dtype=int).reshape((4, 2))
    x0 = np.min(coords_array[:, 0], axis=0)
    y0 = np.min(coords_array[:, 1], axis=0)
    x1 = np.max(coords_array[:, 0], axis=0)
    y1 = np.max(coords_array[:, 1], axis=0)
    return x0, y0, x1, y1
