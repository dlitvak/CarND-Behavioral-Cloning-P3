from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

from keras import metrics

class NvidiaNet:
    def __init__(self):
        self.model = Sequential()

    def network(self, img_shape=(160, 320, 3)):
        # normalize
        self.model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_shape))

        # 160x320x3 to 78x158x24
        self.model.add(Convolution2D(24, kernel_size=5, strides=2))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(36, kernel_size=5, strides=2 ))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(48, kernel_size=5, strides=2 ))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(64, kernel_size=3, strides=1))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(64, kernel_size=3, strides=1))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())

        self.model.add(Dense(100))
        self.model.add(Activation('relu'))

        self.model.add(Dense(50))
        self.model.add(Activation('relu'))

        self.model.add(Dense(10))
        self.model.add(Activation('relu'))

        #Linear Regression Layer
        self.model.add(Dense(1, kernel_initializer="normal", activation="linear"))

        return self.model
