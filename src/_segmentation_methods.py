from skimage.segmentation import slic
from skimage.io import imread_collection, imsave

def kmeans_gray(img_path):

    try:
        img_array = imread_collection(
                        list(img_path),
                        as_gray=True,
                        conserve_memory=True
                    )
    except FileNotFoundError as fnfe:
        print("Could not read from directory: {}".format(image_path))
        print(fnfe.output)

    for image_name in img_path:
        print("Processing " + osp.basename(image_name))
        try:
            image = imread(image_name, as_gray=True)
            segged = slic(
                        image,
                        n_segments=2,
                        compactness=1,
                        max_iter=100,
                        sigma=0
                     )
            pre, base = osp.split(image_name)
            out_path = osp.join(
                            pre,
                            base.split(".")[:-1] + ".seg." + base.split(".")[-1]
                        )
            imsave(out_path, image)
        except ValueError as ve:
            print("Error segmenting " + image)
            print(ve.output)



def kmeans_color(image_path):
    pass
