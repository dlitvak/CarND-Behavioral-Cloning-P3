import csv
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg


def read_data(dir="data", csv_file="driving_log.csv", field_names=("center","left","right","steering")):
    """Read data from csv_file per field_names columns.  Obtain the image size."""
    data = []
    with open(os.path.join(dir, csv_file)) as f:
        csvReader = csv.DictReader(f)
        for row in csvReader:
            data.append(list(row[k] for k in field_names))

    X, y = [], []
    img_shape = None
    for row in data:
        centerImgUrl = row[0].strip()
        X.append(os.path.join(dir, centerImgUrl))
        steering = row[3]
        y.append(steering)

        # leftImgUrl = row[1].strip()
        # X.append(os.path.join(dir, leftImgUrl))
        # y.append(float(steering) - 0.2)
        #
        # rightImgUrl = row[2].strip()
        # X.append(os.path.join(dir, rightImgUrl))
        # y.append(float(steering) + 0.2)

        if img_shape is None:
            im = mpimg.imread(os.path.join(dir, centerImgUrl), format="RGB")
            img_shape = im.shape

    return np.array(X), np.array(y), img_shape


def resize_images_in_dir(dir="data", img_dir="IMG"):
    """Function used to resize all the images in dir once."""
    os.chdir(dir)
    orig_dir = img_dir + "_orig"
    os.rename(img_dir, orig_dir)
    os.mkdir(img_dir)

    imgs = os.listdir(orig_dir)
    for img_name in imgs:
        if not img_name.endswith("jpg"):
            continue

        img = Image.open(os.path.join(orig_dir, img_name))
        img.thumbnail((img.size[0] / 2, img.size[1] / 2), Image.ANTIALIAS)  # resizes image in-place
        resized_img = np.asarray(img, dtype=np.uint8)
        mpimg.imsave(os.path.join(img_dir, img_name), resized_img)

    os.chdir("..")

# resize_images_in_dir(dir="data")
# resize_images_in_dir(dir="col_data2")
# resize_images_in_dir(dir="col_data4_rev")
# resize_images_in_dir(dir="col_data3")

# resize_images_in_dir(dir="2nd_track1")