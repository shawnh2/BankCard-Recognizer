

def selected_box(img_array, x0, y0, x1, y1):
    """Return a selected area in the image"""
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x_start = min(x0, x1)
    y_start = max(y0, y1)
    return img_array[x_start: x_start+width][y_start: y_start+height]


def log_text(content, mode):
    """Show logging panel different infos have different color. """
    assert mode in ["warning", "ok", "error"]
    # the colors are orange, green, and red
    if mode is "warning":
        return '<font color="orange">{}</font>'.format(content)
    elif mode is "ok":
        return '<font color="green">{}</font>'.format(content)
    else:
        return '<font color="red">{}</font>'.format(content)
