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

from logger import get_logger
from utils import get_image_regex_pattern, show_image, is_gray

log = get_logger(level=logging.DEBUG)

def norm(image, bg_avg):
    log.debug("Normalizing image")
    gray = True if is_gray(image) else False

    image = img_as_float(image)

    filter = _get_background_filter(image)

    masked = image.copy()

    if gray:
        masked[filter] = 0
    else:
        masked[filter] = [0,0,0]

    diff = bg_avg - np.mean(masked, axis=(0,1))

    log.debug("Background diff: " + str(diff))

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
    log.debug("Segmenting image")
    filter = _get_background_filter(image)
    masked = image.copy()
    if is_gray(image):
        masked[invert(filter)] = 255
    else:
        masked[invert(filter)] = [255,255,255]
    return masked


def get_bg_avg(indir, PATTERN, type):
    ''' Get the background mean pixel values '''

    log.debug("Gettind background pixel average")

    if type == 'gray':
        sum = 0
    else:
        sum = np.array([0,0,0], dtype=float)

    img_count = 0
    for image_path in os.listdir(indir):
        log.debug("Processing " + image_path)
        if not PATTERN.match(image_path): continue
        try:
            if type == 'gray':
                image = imread(osp.join(indir, image_path), as_gray=True)
            else:
                image = imread(osp.join(indir, image_path))
        except Exception as e:
            log.error("Failed to process " + image_path)
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
        log.debug("All image background average: " + str(mean))
    except ZeroDivisionError as zde:
        log.error("Zero division error, must not have had any images in indir")
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
    log.debug("Getting sorted bounding boxes")
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
