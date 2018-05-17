from utils import *
from steering_angle_predictor import SteeringAnglePredictor


BATCH_SIZE = 128
EPOCHS = 10

# load simulator data files (not included)
X2, y2, img_shape = read_data(dir="col_data2")
X3, y3, img_shape1 = read_data(dir="col_data4_rev")
X1, y1, img_shape2 = read_data(dir="data")
X4, y4, img_shape3 = read_data(dir="col_data3")
X5, y5, img_shape4 = read_data(dir="2nd_track1")

X = np.hstack((X2, X3))
y = np.hstack((y2, y3))
X = np.hstack((X1, X))
y = np.hstack((y1, y))
X = np.hstack((X4, X))
y = np.hstack((y4, y))
X = np.hstack((X5, X))
y = np.hstack((y5, y))

# Execute training.  Train saves the result in model_file
sap = SteeringAnglePredictor(img_shape=img_shape, model_file="model.h5", epochs=EPOCHS, batch_size=BATCH_SIZE, prev_model="2nd_track.h5")
sap.train(X, y, overwrite_model=True)

# sap.test(X[:500], y[:500])


