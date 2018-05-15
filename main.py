from utils import *
import numpy as np
import matplotlib.image as mpimg

from utils import *
from steering_angle_predictor import SteeringAnglePredictor


BATCH_SIZE = 128
EPOCHS = 5

X2, y2, img_shape = read_data(dir="col_data2")
X3, y3, img_shape1 = read_data(dir="col_data4_rev")
X1, y1, img_shape2 = read_data(dir="data")

X = np.hstack((X2, X3))
y = np.hstack((y2, y3))
X_train = np.hstack((X1, X))
y_train = np.hstack((y1, y))

sap = SteeringAnglePredictor(img_shape=img_shape, model_file="lenet_no_lr_b128.h5", epochs=EPOCHS, batch_size=BATCH_SIZE, prev_model="lenet_cumul_more_drp_b40.h5")
sap.train(X, y, overwrite_model=True)

sap.test(X[:300], y[:300])


