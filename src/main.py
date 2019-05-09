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
from .preprocess import (
    convert,
    segment,
    normalize,
    plot_bbox
)

if __name__ == '__main__':
    KernelPheno()
