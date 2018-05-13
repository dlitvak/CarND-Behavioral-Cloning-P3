from nvidia_pipeline import NvidiaNet
from lenet import LeNet
from keras.models import load_model

import numpy as np

class SteeringAnglePredictor:
    def __init__(self, img_shape=(160,320,3), model_file="nvidianet_model.h5", batch_size=128, epochs=5):
        # net = NvidiaNet()
        net = LeNet()
        self.nnModel = net.network(img_shape=img_shape)
        self.modelLoaded = False
        self.modelFile = model_file
        self.batchSize = batch_size
        self.epochs = epochs

    def train(self, X, y, overwriteModel=True):
        self.nnModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = self.nnModel.fit(X, y, epochs=self.epochs, validation_split=0.2, batch_size=self.batchSize, shuffle=True)

        self.nnModel.save(filepath=self.modelFile, overwrite=overwriteModel)
        self.modelLoaded = True
        return history

    def test(self, X_test, y_test):
        if not self.modelLoaded:
            self.nnModel = load_model(self.modelFile)

        # for i in range(len(X_test)):
        #     pred = self.nnModel.predict(np.array([X_test[i]]), batch_size=1)
        #     print("pred {}, real {}".format(pred, y_test[i]))

        metrics = self.nnModel.evaluate(X_test, y_test)
        for metric_i in range(len(self.nnModel.metrics_names)):
            metric_name = self.nnModel.metrics_names[metric_i]
            metric_value = metrics[metric_i]
            print('{}: {}'.format(metric_name, metric_value))
        return metrics

    def quick_normalize_img_data(self, x):
        return np.ndarray.astype((x - 128.0) / 128.0, np.float32)

