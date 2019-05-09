from .utils import get_image_regex_pattern, create_name_from_path
from .preprocess import (
    norm,
    segment_image,
    get_bg_avg,
    get_sorted_bboxes,
    plot_bbx
)
from .logger import get_logger

import click
import os.path as osp
import os
import subprocess as sp
import logging
import traceback as tb


from skimage.io import imread, imsave
import matplotlib.pyplot as plt


log = get_logger(level=logging.DEBUG)


@click.group()
def KernelPheno():
    ''' Kernel Vitreousness project management and phenotyping tools '''
    pass


''' COMMAND IMPORTS '''
from .zooexp import *


''' CONVERSION FROM MISC IMAGE FORMATS TO JPG OR OTHERWISE SPECIFIED FORMAT '''

@click.command()
@click.argument(
    'dir'
)
@click.option(
    '-f',
    'format',
    help='Format to convert the images to',
    default='jpg',
    show_default=True
)
@click.option(
    '--copyto',
    help='Copy images to this directory before converting',
    default=False
)
def convert(dir, format, copyto):
    ''' Convert images to specified format '''

    if not os.path.isdir(dir):
        log.error('The input dir does not exist')
        return

    if copyto:
        sp.run(['mkdir', '-p', copyto])
        sp.run(['cp', '-r', dir, copyto])

    sp.run(['mogrify', '-format', format, osp.join(dir, '*')])

    return

KernelPheno.add_command(convert)


''' SEGMENTATION FUNCTION '''

@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
# @click.option(
#     '-t',
#     '--type',
#     help='The type of image to be segmented',
#     type=click.Choice(['gray', 'rgb']),
#     required=True
# )
@click.option(
    '-e',
    '--extension',
    help='The image extension',
    type=click.Choice(['jpg', 'jpeg', 'png', 'tif', 'tiff']),
    multiple=True,
    default=None
)
def segment(indir, outdir, extension):
    '''
    Segment the kernels in the image or images

    params
    * indir:    directory with images to segment
    * outdir:   directory to output images (will create if doesn't exist)
    * type:     gray or rgb
    '''
    sp.run(['mkdir', '-p', outdir])
    PATTERN = get_image_regex_pattern()
    for image_path in os.listdir(indir):
        if not PATTERN.match(image_path): continue
        try:
            image = imread(osp.join(indir, image_path))
            seg_image = segment_image(image)
            out_fname = create_name_from_path(image_path, out_dir=outdir)
            imsave(out_fname, seg_image)
        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
            continue

KernelPheno.add_command(segment)


''' NORMALIZE IMAGES '''
@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
@click.argument(
    'type',
    type=click.Choice(['rgb', 'gray'])
)
@click.option(
    '--plot',
    help='Plot the comparison between normed and real'
)
def normalize(indir, outdir, type, plot):
    '''
    Perform mean scaling normalization method

    params
    * indir:    directory with images to normalize
    * outdir:   directory to output images (will create if doesn't exist)
    * type:     image type for normalization
    '''

    ###########################################################################
    # TO BE UPDATED WHEN RGB NORMALIZATION IMPLEMENTED
    if type == 'rgb':
        log.info('Normalization for color images has not yet been implemented')
        return
    ###########################################################################

    sp.run(['mkdir', '-p', outdir])

    PATTERN = get_image_regex_pattern()
    bg_avg = get_bg_avg(indir, PATTERN, type)

    for image_path in os.listdir(indir):
        if not PATTERN.match(image_path): continue
        log.info('Processing ' + image_path)

        try:
            log.info("Loading image")
            if type == 'gray':
                image = imread(osp.join(indir, image_path), as_gray=True)
            else:
                image = imread(osp.join(indir, image_path))
                if len(image.shape) == 2:
                    image = gray2rgb(image)

            normed = norm(image, bg_avg)

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
                fig.set_title('Normalized Image Comparison')
                if type == 'gray':
                    ax[0].imshow(image, cmap='gray')
                    ax[1].imshow(normed, cmap='gray')
                else:
                    ax[0].imshow(image)
                    ax[1].imshow(normed)
                figname = create_name_from_path(image_path, ('fig'), outdir)
                plt.savefig(figname)

            out_fname = create_name_from_path(image_path, out_dir=outdir)
            log.info("Saving file: " + out_fname)
            imsave(out_fname, normed)

        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            continue

    return

KernelPheno.add_command(normalize)


''' PLOT BOUNDING BOXES '''
@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
def plot_bbox(indir, outdir):
    '''
    Plot the bounding boxes for each image given in indir

    params
    * indir:    directory with images to normalize
    * outdir:   directory to output images (will create if doesn't exist)
    '''
    sp.run(['mkdir', '-p', outdir])

    PATTERN = get_image_regex_pattern()
    for image_name in os.listdir(indir):
        if not PATTERN.match(image_name): continue
        log.info("Processing " + osp.basename(image_name))

        try:
            image = imread(osp.join(indir, image_name))
        except Exception as e:
            log.error("Failed to process " + image_name)
            log.error(e)
            continue

        bboxes = get_sorted_bboxes(image)
        plot_bbx(image, bboxes)
        out_fname = osp.join(outdir, image_name)
        plt.savefig(out_fname)

KernelPheno.add_command(plot_bbox)

''' GENERATE DATASET '''

def generate_dataset():
    '''
    params
    '''
    # Output dims for AlexNet = (227,227,3)
    pass


if __name__ == '__main__':
    KernelPheno()
