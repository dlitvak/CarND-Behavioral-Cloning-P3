# from sklearn.utils import shuffle
from utils import *
from steering_angle_predictor import SteeringAnglePredictor


BATCH_SIZE = 128
EPOCHS = 10

X2, y2, img_shape = read_data(dir="col_data2")
X3, y3, img_shape1 = read_data(dir="col_data4_rev")
X1, y1, img_shape2 = read_data(dir="data")
X4, y4, img_shape3 = read_data(dir="col_data3")

X = np.hstack((X2, X3))
y = np.hstack((y2, y3))
X = np.hstack((X1, X))
y = np.hstack((y1, y))
X = np.hstack((X4, X))
y = np.hstack((y4, y))

sap = SteeringAnglePredictor(img_shape=img_shape, model_file="lenet_recover.h5", epochs=EPOCHS, batch_size=BATCH_SIZE, prev_model="test_rev3")
sap.train(X, y, overwrite_model=True)

# X, y = shuffle(X, y)
sap.test(X[:500], y[:500])


