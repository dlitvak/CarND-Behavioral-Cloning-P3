from utils import *
import numpy as np
import matplotlib.image as mpimg

from utils import *
from steering_angle_predictor import SteeringAnglePredictor


BATCH_SIZE = 64
EPOCHS = 10

# X1, y1, img_shape = read_data(dir="data")
X2, y2, img_shape = read_data(dir="col_data2")
X3, y3, img_shape1 = read_data(dir="col_data4_rev")

# X_train = np.vstack((X1, X2))
# y_train = np.hstack((y1, y2))
X = np.hstack((X2, X3))
y = np.hstack((y2, y3))


sap = SteeringAnglePredictor(img_shape=img_shape, model_file="lenet_cumul.h5", epochs=EPOCHS, batch_size=BATCH_SIZE)
sap.train(X, y, overwrite_model=True)
#
# sap.test(X[:300], y[:300])


