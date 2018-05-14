from utils import *
import numpy as np
import matplotlib.image as mpimg
# import matplotlib.pyplot as pplt

from steering_angle_predictor import SteeringAnglePredictor


DIR = "col_data2/"
CSV_FILE = "driving_log.csv"

BATCH_SIZE = 64
EPOCHS = 15

data = readFile(dir=DIR, csv_file=CSV_FILE, fieldNames=("center","steering"))
# print (data)  #fieldNames=("center","left","right","steering")
X, y =  [], []
for row in data:
    centerImgUrl = row[0]
    imread = mpimg.imread(DIR + centerImgUrl, format="RGB")
    X.append(imread)

    steering = row[1]
    y.append(steering)

X, y = np.array(X), np.array(y)
sap = SteeringAnglePredictor(img_shape=X.shape[1:], model_file="lenet2.h5", epochs=EPOCHS, batch_size=BATCH_SIZE, prev_model="lenet.h5")
sap.train(X, y, overwrite_model=True)

sap.test(X[:300], y[:300])


