from skimage.segmentation import slic
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.util import invert
from skimage import img_as_float, img_as_int

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os.path as osp
import os
import argparse
import click

from utils import create_name_from_path, show_image
from main import KernelPheno

###############################################################################
# Commands
###############################################################################

@click.command()
@click.argument(
    'indir'
)
@click.argument(
    outdir
)
@click.option(
    '-n',
    '--normalize',
    help='Normalize the images',
    default=False
)
@click.option(
    '-g',
    '--grayscale',
    help='Convert all images to grayscale',
    default=False
)

def generate_dataset(indir, outdir, normalize=False, grayscale=False):
    '''
    Generates dataset of thumbnail images for CNN training
    params
    * indir: directory with images to process
    * outdir: directory to put annotated thumbnails
    '''
    pass


def plot_bounding_boxes(indir, outdir):
    '''
    Plots the numbered bounding boxes for verification
    params
    * indir: directory with images to process
    * outdir: directory to plotted bounding boxes
    '''
    pass


def plot_normalization(indir, outdir):
    ''' show the comparison of normalized to raw images
    params
    * indir: directory with images to process
    * outdir: directory to plotted normalized images
    '''
    pass


###############################################################################
# Helper Functions
###############################################################################
