import re
import os.path as osp
import matplotlib.pyplot as plt

def get_image_regex_pattern(extension):
    '''
    Returns a regex pattern useful for grabbing filenames with image filename
    extensions
    '''
    if extension == ():
        return re.compile(r".*\.(tif|tiff|jpg|jpeg|png)")

    patter_str = r".*\.(" + extension[0]
    for ext in extension[1:]:
        pattern_str += "|" + ext
    patter_str += ")"

    return re.compile(patter_str)


def create_name_from_path(file_path, pre_ext, out_dir=False):
    '''
    Inserts custom tags seperated by dots between filename and extension
    '''
    if type(pre_ext) != list:
        pre_ext = [pre_ext]

    extensions = osp.basename(file_path).split(".")
    for i, ext in enumerate(pre_ext):
        if ext in extensions:
            print(
                "File: " + osp.basename(file_path)\
                + " already has extension: " \
                + ext
            )
            print("Ommitting from final filename")
            pre_ext.pop(i)

    out_path    = ".".join(file_path.split(".")[:-1]) \
                + "." + ".".join(pre_ext) + "." \
                + file_path.split(".")[-1]
    if out_dir:
        out_path = osp.join(
            out_dir,
            osp.basename(out_path)
        )

    return out_path


def show_image(image, cmap=None):
    '''
    Simply plots the image
    '''
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.show()
