from nvidia_pipeline import NvidiaNet
from lenet import LeNet
from keras.models import load_model
from keras.utils import Sequence

from sklearn.utils import shuffle
import numpy as np
import math as m

class SteeringAnglePredictor:
    def __init__(self, img_shape=(160,320,3), model_file="nvidianet_model.h5", batch_size=128, epochs=5):
        # net = NvidiaNet()
        net = LeNet()
        self.nnModel = net.network(img_shape=img_shape)
        self.modelLoaded = False
        self.modelFile = model_file
        self.batchSize = batch_size
        self.epochs = epochs

    class DataSequence(Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array(batch_x), np.array(batch_y)

    def train(self, X, y, overwriteModel=True):
        X, y = shuffle(X, y)
        valid_len = m.ceil(0.2*len(X))
        X_valid, y_valid = X[0:valid_len], y[0:valid_len]
        X_train, y_train = X[valid_len:], y[valid_len:]

        self.nnModel.compile(optimizer='adam', loss='mean_squared_error')
        history = self.nnModel.fit_generator(generator=self.DataSequence(X_train, y_train, self.batchSize) ,
                                             epochs=self.epochs, validation_data=self.DataSequence(X_valid, y_valid, self.batchSize),
                                             shuffle=True, verbose=2)

        self.nnModel.save(filepath=self.modelFile, overwrite=overwriteModel)
        self.modelLoaded = True
        return history

    def test(self, X_test, y_test):
        if not self.modelLoaded:
            self.nnModel = load_model(self.modelFile)

        for i in range(len(X_test)):
            pred = self.nnModel.predict(np.array([X_test[i]]), batch_size=1)
            print("pred {}, real {}".format(pred, y_test[i]))

        # metrics = self.nnModel.evaluate(X_test, y_test)
        # for metric_i in range(len(self.nnModel.metrics_names)):
        #     metric_name = self.nnModel.metrics_names[metric_i]
        #     metric_value = metrics[metric_i]
        #     print('{}: {}'.format(metric_name, metric_value))
        # return metrics

    def quick_normalize_img_data(self, x):
        return np.ndarray.astype((x - 128.0) / 128.0, np.float32)

