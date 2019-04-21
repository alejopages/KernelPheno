import re

def get_image_regex_pattern(extension):

    if extension == ():
        return re.compile(r".*\.(tif|tiff|jpg|jpeg|png)")

    patter_str = r".*\.(" + extension[0]
    for ext in extension[1:]:
        pattern_str += "|" + ext
    patter_str += ")"

    return re.compile(patter_str)
