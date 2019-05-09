from skimage.segmentation import slic, clear_border
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.util import invert
from skimage import img_as_float, img_as_int
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import os
import os.path as osp
import logging

from .logger import get_logger
from .main import KernelPheno
from utils import get_image_regex_pattern, show_image, is_gray

log = get_logger(level=logging.DEBUG)

''' GENERATE DATASET '''
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
@click.arguement(
    'type',
    type=click.Choice(['gray', 'rgb'])
)
@click.argument(
    'anno_file'
)

def generate_dataset(indir, outdir, type, anno_file):
    '''
    params
    '''
    # Output dims for AlexNet = (227,227,3)

    '''
    * create a folder in outdir for each annotation (1-5)
    * load annotations file
    * get bg_avg
    * loop through and load images from annotations
    *   Normalize image
    *   Get bounding boxes for image
    *   Compare number of bboxes to number of objects in annotation
    *   loop through bounding boxes and get vitr annotation
    *       extract bounding box thumbnail
    *       resize to AlexNet dims
    *       save in annotation's directory
    '''
    sp.run(['mkdir', '-p', outdir])
    for i in range(1, 6):
        sp.run(['mkdir', osp.join(outdir, str(i))])


    pass

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
            log.info('Loading image')
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
            log.info('Saving file: ' + out_fname)
            imsave(out_fname, normed)

        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
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
        log.info('Processing ' + osp.basename(image_name))

        try:
            image = imread(osp.join(indir, image_name))

            bboxes = get_sorted_bboxes(image)
            plot_bbx(image, bboxes)
            out_fname = osp.join(outdir, image_name)
            plt.savefig(out_fname)
        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
            continue

KernelPheno.add_command(plot_bbox)


def norm(image, bg_avg):

    log.debug('Normalizing image')

    gray = True if is_gray(image) else False

    image = img_as_float(image)

    filter = _get_background_filter(image)

    masked = image.copy()

    if gray:
        masked[filter] = 0
    else:
        masked[filter] = [0,0,0]

    diff = bg_avg - np.mean(masked, axis=(0,1))

    log.debug('Background diff: ' + str(diff))

    normed = image + diff

    if gray:
        normed[normed > 1.0] = 1.0
        normed[normed < -1.0] = -1.0
    else:
        # TODO: find out how to scale the color normalized to [0,255]
        pass

    return normed


def test_norm():
    image = imread('/home/apages/pysrc/KernelPheno/data/sample_images/DSC05389.jpeg', as_gray=True)
    PATTERN = get_image_regex_pattern()
    bg_avg = get_bg_avg('/home/apages/pysrc/KernelPheno/data/sample_images', PATTERN, type='gray')
    normed = normalize(image, bg_avg)
    plt.imshow(normed, cmap='gray')
    plt.show()
    return


def segment_image(image):
    log.debug('Segmenting image')
    filter = _get_background_filter(image)
    masked = image.copy()
    if is_gray(image):
        masked[invert(filter)] = 255
    else:
        masked[invert(filter)] = [255,255,255]
    return masked


def get_bg_avg(indir, PATTERN, type):
    ''' Get the background mean pixel values '''

    log.debug('Gettind background pixel average')

    if type == 'gray':
        sum = 0
    else:
        sum = np.array([0,0,0], dtype=float)

    img_count = 0
    for image_path in os.listdir(indir):
        log.debug('Processing ' + image_path)
        if not PATTERN.match(image_path): continue
        try:
            if type == 'gray':
                image = imread(osp.join(indir, image_path), as_gray=True)
            else:
                image = imread(osp.join(indir, image_path))
        except Exception as e:
            log.error('Failed to process ' + image_path)
            log.error(e)
            continue

        image = img_as_float(image)

        filter = _get_background_filter(image)
        masked = image.copy()

        if type == 'gray':
            masked[filter] = 0
        else:
            masked[filter] = [0,0,0]

        mean = np.mean(masked, axis=(0,1))
        sum += mean
        img_count += 1
    try:
        mean = sum / float(img_count)
        log.debug('All image background average: ' + str(mean))
    except ZeroDivisionError as zde:
        log.error('Zero division error, must not have had any images in indir')
        raise zde

    return mean


def plot_bbx(image, bboxes):

    # print(bboxes)

    gray = True if is_gray(image) else False

    fig, ax = plt.subplots(figsize=(6,6))

    if gray:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    for i, (minr, minc, maxr, maxc) in enumerate(bboxes):
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(minc - 20, minr - 20, str(i))

    return plt


def get_sorted_bboxes(image):
    ''' Generate the sorted bounding boxes '''
    log.debug('Getting sorted bounding boxes')
    filter = _get_background_filter(image)
    cleared = clear_border(filter)
    label_image = label(cleared)
    coords = []
    for region in regionprops(label_image, coordinates='rc'):
        if (region.area < 1000) \
            or (region.area > 100000) \
            or ((region.minor_axis_length / region.major_axis_length) < 0.2):
            continue

        coords.append(region.bbox) # minr, minc, maxr, maxc

    sorted_bbxs = _sort_bbxs(coords, image.shape[0])

    return sorted_bbxs


def _sort_bbxs(regions, num_rows):
    ''' Sort bboxes left to right, top to bottom '''
    def overlap(el1, el2):
        ''' determine in bounding boxes overlap along a row '''
        el1_minr, _, el1_maxr, _ = el1
        el2_minr, _, el2_maxr, _ = el2

        upper_max = max(el1_minr, el2_minr)
        lower_min = min(el1_maxr, el2_maxr)

        return (upper_max < lower_min)

    rows = []

    while(len(regions)):
        sorted_by_y = sorted(regions, key=lambda x: x[0])
        first_el = sorted_by_y[0]
        rows.append([first_el])
        regions.remove(first_el)
        sorted_by_y.pop(0)
        for el in sorted_by_y:
            if overlap(el, first_el) or overlap(el, rows[-1][-1]):
                rows[-1].append(el)
                regions.remove(el)

    sorted_bbxs = []
    for row in rows:
        sorted_bbxs += sorted(row, key=lambda x: x[1])
    return sorted_bbxs


def _get_background_filter(image):
    '''
    Get's the binary filter of the segmented image
    '''
    log.debug('Getting image background filter')
    if not is_gray(image):
        image = rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    return invert(bw)

#
# if __name__ == '__main__':
#     test_normalize()
