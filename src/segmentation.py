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
            gray_image = rgb2gray(col_image)
            thresh = threshold_otsu(gray_image)
            bw = closing(gray_image > thresh, square(3))
            
            inverted = invert(bw)
            label_image = label(inverted)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
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
                            "".join(img_name.split(".")[:-1]) + ".seg." + img_name.split(".")[-1]
                        )
            plt.savefig(out_name)
            # plt.show()


        except FileNotFoundError as fnfe:
            print(fnfe)
    return


if __name__ == '__main__':
    draw_bounding_boxes(["/home/apages/pysrc/KernelPheno/data/DSC05377.jpeg",
                         "/home/apages/pysrc/KernelPheno/data/DSC05389.jpeg",
                         "/home/apages/pysrc/KernelPheno/data/DSC05384.jpeg"])
