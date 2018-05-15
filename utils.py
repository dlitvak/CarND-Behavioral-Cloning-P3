import csv
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg

def readFile(dir="data/", csv_file="driving_log.csv", fieldNames=("center","left","right","steering")):
    """Read data in the fieldNames from the csv_file (in directory dir)."""

    data = []
    with open(dir + csv_file) as f:
        csvReader = csv.DictReader(f)
        for row in csvReader:
            data.append(list(row[k] for k in fieldNames))

    return data

def resize_images_in_dir(dir="data_/", img_dir="IMG"):
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