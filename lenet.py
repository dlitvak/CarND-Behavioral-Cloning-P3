from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Convolution2D


class LeNet:
    def __init__(self):
        self.model = Sequential()

    def network(self, img_shape=(160, 320, 3)):
        # normalize
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=img_shape))

        # 160x320x3 to 158x318x6
        self.model.add(Convolution2D(6, kernel_size=3, strides=1, padding="valid"))
        self.model.add(Activation('relu'))

        # to 154x314x16
        self.model.add(Convolution2D(16, kernel_size=5, strides=1, padding="valid"))
        self.model.add(Activation('relu'))

        # to 77x157x16
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        # to 76x156x26
        self.model.add(Convolution2D(26, kernel_size=2, strides=1,  padding="valid"))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(52, kernel_size=3, strides=1, padding="valid"))
        self.model.add(Activation('relu'))

        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        self.model.add(Flatten())

        self.model.add(Dense(400))
        self.model.add(Activation('relu'))

        self.model.add(Dense(120))
        self.model.add(Activation('relu'))

        self.model.add(Dense(60))
        self.model.add(Activation('relu'))

        # Linear Regression Layer
        self.model.add(Dense(1, kernel_initializer="normal", activation="linear"))

        return self.model
