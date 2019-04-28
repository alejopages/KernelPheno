from skimage.segmentation import slic
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os.path as osp


def draw_bounding_boxes(img_paths, out_path=None):

    if not out_path:
        out_path = osp.dirname(img_paths[0])

    for img_path in img_paths:
        img_name = osp.basename(img_path)
        print("Processing " + img_path)

        try:
            col_image = imread(img_path)
            filter = get_filter(col_image)

            label_image = label(invert(filter))

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

            col_image[filter] = [255,255,255]
            # plt.imshow(col_image)
            # plt.show()
            # return
            ax.imshow(col_image)

            for region in regionprops(label_image):
                if region.area < 100 or region.area > 100000:
                    continue

                # draw rectangle around segmented kernels
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)

                ax.add_patch(rect)

            out_name = osp.join(out_path,
                            "".join(img_name.split(".")[:-1])
                                    + ".seg."
                                    + img_name.split(".")[-1]
                        )
            plt.savefig(out_name)
            # plt.show()


        except FileNotFoundError as fnfe:
            print(fnfe)
    return


def get_filter(image):
    if image.shape[2] == 3:
        image = rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    return bw


def get_bg_avg(img_paths):

    sum = np.array([0,0,0], dtype=float)
    img_count = 0

    for img_file in img_paths:
        col_img = imread(img_file)
        filter = get_filter(col_img)
        masked = col_img.copy()
        masked[invert(filter)] = [0,0,0]
        mean = np.mean(masked, axis=(0,1))
        sum += mean
        img_count += 1

    mean = sum / float(img_count)
    print("Mean: " + str(mean))
    return mean


def show_image(image, cmap=None):
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.show()


def normalize_images(img_paths):
    bg_avg = get_bg_avg(img_paths)
    bg_avg = -bg_avg + 255
    for img_file in img_paths:
        col_img = imread(img_file)
        filter = get_filter(col_img)
        masked = col_img.copy()
        masked[invert(filter)] = [0,0,0]
        diff = bg_avg - np.mean(masked)
        print("Diff: " + str(diff))
        normed = col_img + diff
        print(img_file)
        show_image(normed)


if __name__ == '__main__':
    normalize_images(["/home/apages/pysrc/KernelPheno/data/DSC05377.jpeg",
                      "/home/apages/pysrc/KernelPheno/data/DSC05389.jpeg",
                      "/home/apages/pysrc/KernelPheno/data/DSC05384.jpeg"])
