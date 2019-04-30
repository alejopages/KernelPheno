from main import KernelPheno
from utils import get_image_regex_pattern
from segmentation import *

import click
import os.path as osp
import subprocess as sp

'''
CONVERSION FROM MISC IMAGE FORMATS TO JPG OR OTHERWISE SPECIFIED FORMAT
'''

@click.command()
@click.argument(
    'dir'
)
@click.option(
    '-f',
    'format',
    help="Format to convert the images to",
    default='jpg',
    show_default=True
)
def convert(dir, format):

    if not osp.isdir(dir):
        print("This directory does not exist: " + dir)
        exit()

    mog_proc = sp.run(['mogrify', '-format', format, osp.join(dir, '*')])
    return


KernelPheno.add_command(convert)


'''
SEGMENTATION FUNCTION
'''

@click.command()
@click.argument(
    'img_path'
)
@click.option(
    '-t',
    '--type',
    help='The type of image to be segmented',
    type=click.Choice(['gray', 'color']),
    required=True
)
@click.option(
    '-m',
    '--method',
    help='The method used to do the segmentation',
    type=click.Choice(['thresh']),
    required=True
)
@click.option(
    '-e',
    '--extension',
    help='The image extension',
    type=click.Choice(['jpg', 'jpeg', 'png', 'tif', 'tiff']),
    multiple=True,
    default=None
)
@click.option(
    '-o',
    '--output',
    help='Output directory',
    default=None
)
def segmentation(img_path, type, method, extension, output):
    ''' Segment the kernels in the image or images '''
    PATTERN = get_image_regex_pattern(extension)
    images = []

    if PATTERN.match(img_path) and not osp.isdir(img_path):
        images.append(osp.abspath(img_path))
    else:
        for img in osp.listdir(img_path):
            if PATTERN.match(img):
                images.append(osp.abspath(img))

    if images == []:
        print("There were no images in the specified path")
        print("Check that the image(s) has a valid file extension")
        return

    if method == 'thresh':
        draw_bounding_boxes(images, output)


KernelPheno.add_command(segmentation)


def generate_dataset():
    '''
    Output dims for AlexNet = (227,227,3)
    '''


if __name__ == '__main__':
    segmentation()
