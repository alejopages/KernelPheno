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

from utils import create_name_from_path, show_image


def draw_bounding_boxes(img_paths, out_dir=False, gray=False):
    '''
    Draws bounding boxes and save figures
    '''
    print('Drawing Bounding Boxes')

    for img_path in img_paths:
        print('Processing ' + img_path)
        try:
            img = imread(img_path, as_gray=gray)

            if len(img.shape) == 2 and not gray:
                print('Error: expected color image, got grayscale')
                continue


            filter = _get_filter(img)

            label_image = label(invert(filter))

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

            if gray:
                ax.imshow(img, cmap='gray')
                img[filter] = 1.0
            else:
                ax.imshow(img)
                img[filter] = [255, 255, 255]

            for region in regionprops(label_image):
                # skip unreasonable bbox areas
                if region.area < 1000 or region.area > 100000:
                    continue

                # draw rectangle around segmented kernels
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)

                ax.add_patch(rect)

            exts = []
            if gray:
                exts += ['gray']
            exts += ['bbxs']

            out_name = create_name_from_path(img_path, exts, out_dir)
            print('Saving file: ' + out_name)
            plt.savefig(out_name)

        except FileNotFoundError as fnfe:
            print('image file does not exist')
            print(fnfe)
        except OSError as ose:
            print('output directory likely does not exist')
            print(ose)

    return


def draw_bounding_boxes(img, regions, save=False):
    '''
    Takes the minr minc maxr maxc coordinates of bounding boxes
    and plots on the img
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for i, (minr, minc, maxr, maxc) in enumerate(regions):
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(minc - 20, minr - 20, str(i))
    if save:
        plt.savefig(save)
    else:
        plt.show()

    return


def segment_images(img_paths, out_dir=False):
    for img_file in img_paths:
        img = imread(img_file)
        filter = _get_filter(img)
        filtered = img.copy()
        if len(img.shape) == 3:
            filtered[filter] = [0,0,0]
        else:
            filtered[invert(filter)] = 0
        out_name = create_name_from_path(img_file, 'seg', out_dir=out_dir)
        show_image(filtered)
        imsave(out_name, filtered)
    return




#############################################

def get_bboxes(img, plot=False, fig_location=False):
    filter = _get_filter(img)
    label_image = label(invert(filter))
    coords = []
    for region in regionprops(label_image):
        if region.area < 1000 or region.area > 100000:
            continue
        coords.append(region.bbox) # minr, minc, maxr, maxc

    sorted_bbxs = sort_bbxs(coords, img.shape[0])
    if plot:
        draw_bounding_boxes(img, sorted_bbxs, save=fig_location)
    return sorted_bbxs


def test_get_bboxes():
    img = imread("/home/apages/pysrc/KernelPheno/data/sample_images/DSC05389.jpeg")
    get_bboxes(img, plot=True)


def sort_bbxs(regions, num_rows):
    '''
    Sort bboxes left to right, top to bottom
    '''

    def overlap(el1, el2):
        el1_minr, _, el1_maxr, _ = el1
        el2_minr, _, el2_maxr, _ = el2

        inner_min = max(el1_minr, el2_minr)
        outer_min = min(el1_maxr, el2_maxr)

        return (inner_min < outer_min)

    rows = []

    while(len(regions)):
        sorted_by_y = sorted(regions, key=lambda x: x[0])
        rows.append([])
        first_el = sorted_by_y[0]
        for el in sorted_by_y:
            if overlap(el, first_el):
                rows[-1].append(el)
                regions.remove(el)

    sorted_bbxs = []
    for row in rows:
        sorted_bbxs += sorted(row, key=lambda x: x[1])
    return sorted_bbxs



def get_thumbnails(img, out_dir=False):
    '''
    Get's thumbnail images
    '''
    pass

###################################################
def _get_filter(image):
    '''
    Get's the binary filter of the segmented image
    '''
    print('Getting image background filter')
    if len(image.shape) == 3:
        image = rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    return bw


def _get_bg_avg(img_paths, gray=False):
    '''
    Get mean of background pixels for all images in img_paths
    '''
    print('Getting background average')
    if gray:
        sum = 0
    else:
        sum = np.array([0,0,0], dtype=float)

    img_count = 0

    for img_file in img_paths:
        try:
            img = img_as_float(imread(img_file, as_gray=gray))
            if ((len(img.shape) == 2) and not gray):
                print("Ignoring: " + img_file)
                print("> the gray argument is False but this image is grayscale")
                continue
        except FileNotFoundError as fnfe:
            print(fnfe)
            print('File: ' + img_file)

        filter = _get_filter(img)
        masked = img.copy()
        if gray:
            masked[invert(filter)] = 0
        else:
            masked[invert(filter)] = [0,0,0]

        mean = np.mean(masked, axis=(0,1))
        print(mean)
        sum += mean
        img_count += 1

    try:
        mean = sum / float(img_count)
    except ZeroDivisionError as zde:
        return 0

    print('Mean: ' + str(mean))
    return mean


def normalize_images(img_paths, out_dir=False, plot=False, gray=False):
    '''
    Normalize the images with respect to the background pixels for all provided
    images
    '''
    print('Normalizing images')
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2)

    bg_avg = _get_bg_avg(img_paths, gray=gray)

    if bg_avg is None and not gray:
        print("There were no rgb images in the directory")
        return

    for img_file in img_paths:
        print('Normalizing ' + img_file)
        try:
            img = img_as_float(imread(img_file, as_gray=gray))
            if len(img.shape) == 2 and not gray:
                print("Expected rgb image, got grayscale")
                print("Skipping ...")
                continue
        except FileNotFoundError as fnfe:
            print(fnfe)

        filter = _get_filter(img)
        masked = img.copy()

        if gray:
            masked[invert(filter)] = 0
        else:
            print("Image shape: " + str(img.shape))
            masked[invert(filter)] = [0,0,0]

        diff = bg_avg - np.mean(masked)

        print('Background diff: ' + str(diff))
        normed = img + diff

        if gray:
            normed[normed > 1.0] = 1.0
            normed[normed < -1.0] = -1.0

        if plot:
            print('Plotting')
            ax[0].set_title('Original')
            ax[1].set_title('Normalized')
            exts = ['fig']
            if gray:
                ax[0].imshow(img, cmap='gray')
                ax[1].imshow(normed, cmap='gray')
                exts.append('gray')
            else:
                ax[0].imshow(img)
                ax[1].imshow(normed)
            fig_name = create_name_from_path(img_file, exts, out_dir=out_dir)
            print('Saving figure: ' + fig_name)
            plt.savefig(fig_name)

        exts = []
        if gray:
            exts = ['gray']
        exts += ['norm']

        try:
            out_name = create_name_from_path(img_file, exts, out_dir=out_dir)
            print('Saving file: ' + out_name)
            imsave(out_name, normed)
        except OSError as ose:
            print(ose)
        except ValueError as ve:
            print(ve)
        finally:
            print(img_file)


def test_gray_normal():
    sample_img_dir = '/home/apages/pysrc/KernelPheno/data/sample_images'
    test_files = [osp.join(sample_img_dir, img_file) for img_file in os.listdir(sample_img_dir)]
    print("###################################################################")
    print("             GRAYSCALE")
    print("###################################################################")

    normalize_images(test_files, out_dir='/home/apages/pysrc/KernelPheno/data/tests', gray=True)
    print("###################################################################")
    print("             COLOR")
    print("###################################################################")
    normalize_images(test_files, out_dir='/home/apages/pysrc/KernelPheno/data/tests')


def __test_all():

    sample_img_dir = '/home/apages/pysrc/KernelPheno/data/sample_images'
    test_files = [osp.join(sample_img_dir, img_file) for img_file in os.listdir(sample_img_dir)]
    test_out = '/home/apages/pysrc/KernelPheno/data/tests'
    gray_normed = [osp.join(test_out, file) for file in os.listdir(test_out) if ('bbxs' not in file.split("."))]

    print('##################################################################')
    print('TESTING SEGMENTATION')
    print('##################################################################')
    print('\n\n')

    print('DRAWING BOUNDING BOXES')

    draw_bounding_boxes(test_files,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests')
    draw_bounding_boxes(test_files,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests',
                              gray=True)

    print('##################################################################')
    print('NORMALIZING IMAGES')
    print('##################################################################')
    print('COLOR:')
    normalize_images(test_files,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests',
                              plot=True)
    print('GRAY:')
    normalize_images(test_files,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests',
                              plot=True,
                              gray=True)


    print('##################################################################')
    print('\n\nDRAWING BBOXES AROUND NORMALIZED')
    print('##################################################################')

    draw_bounding_boxes(gray_normed,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests')
    draw_bounding_boxes(gray_normed,
                              out_dir='/home/apages/pysrc/KernelPheno/data/tests',
                              gray=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all functions on color and gray')
    parser.add_argument('--thumb', action='store_true', help='Test thumbnails script')
    parser.add_argument('--seg', action='store_true', help='Ouput Segmented images')
    parser.add_argument('--bbx', action='store_true', help='Test Get bboxes')
    args = parser.parse_args()
    if args.all:
        __test_all()
    elif args.thumb:
        create_thumbnails(['/home/apages/pysrc/KernelPheno/data/DSC05377.jpeg',
                           '/home/apages/pysrc/KernelPheno/data/DSC05389.jpeg',
                           '/home/apages/pysrc/KernelPheno/data/DSC05384.jpeg'],
                           out_dir='/home/apages/pysrc/KernelPheno/data/tests')
    elif args.seg:
        segment_images(['/home/apages/pysrc/KernelPheno/data/DSC05377.jpeg',
                        '/home/apages/pysrc/KernelPheno/data/DSC05389.jpeg',
                        '/home/apages/pysrc/KernelPheno/data/DSC05384.jpeg'],
                        out_dir='/home/apages/pysrc/KernelPheno/data/tests')
    elif args.bbx:
        test_get_bboxes()
