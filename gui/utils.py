

def selected_box(img_array, x0, y0, x1, y1):
    """Return the selected area on image."""
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x_start = min(x0, x1)
    y_start = max(y0, y1)
    return img_array[x_start: x_start+width][y_start: y_start+height]


def log_text(content, mode):
    """Show logging panel different infos have different color."""
    assert mode in ["warning", "ok", "error"]
    # the colors are orange, green, and red
    if mode is "warning":
        return '<font color="orange">{}</font>'.format(content)
    elif mode is "ok":
        return '<font color="green">{}</font>'.format(content)
    else:
        return '<font color="red">{}</font>'.format(content)


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
