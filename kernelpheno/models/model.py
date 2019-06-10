from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np

from math import ceil
import sys
import pickle
import subprocess as sp
import os.path as osp

np.random.seed(1000)

class AlexNet:

    def __init__(self):
        #Instantiate an empty self.model
        self.model = Sequential()

        # 1st Convolutional Layer> 4 and len(sys.argv) < 7
        self.model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
        self.model.add(Activation('relu'))
        # Max Pooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        self.model.add(Activation('relu'))
        # Max Pooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.model.add(Activation('relu'))

        # 4th Convolutional Layer
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.model.add(Activation('relu'))

        # 5th Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.model.add(Activation('relu'))
        # Max Pooling
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        self.model.add(Flatten())
        # 1st Fully Connected Layer
        self.model.add(Dense(4096, input_shape=(224*224*3,)))
        self.model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        self.model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        # Add Dropout
        self.model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        self.model.add(Dense(1000))
        self.model.add(Activation('relu'))
        # Add Dropout
        self.model.add(Dropout(0.4))

        # Output Layer
        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))

        self.model.summary()

        # Compile the self.model
        self.model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer='adadelta',
            metrics=['accuracy']
        )


    def train(self, model_name, data, outdir, epochs, batch_size,
              validation_split=0.2):

        epochs = int(epochs)
        batch_size = int(batch_size)

        base = osp.join(outdir, 'train')

        sp.run(['mkdir', '-p', base])
        sp.run(['mkdir', '-p', osp.join(base, 'models')])
        sp.run(['mkdir', '-p', osp.join(base, 'tb')])
        sp.run(['mkdir', '-p', osp.join(base, 'verif')])
        sp.run(['mkdir', '-p', osp.join(base, 'verif', 'train')])
        sp.run(['mkdir', '-p', osp.join(base, 'verif', 'valid')])

        tensorboard = TensorBoard(log_dir=osp.join(base, 'tb'))
        train_dg = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True
        )
        valid_dg = ImageDataGenerator(
            rescale=1./255
        )

        classes = ['0', '1', '2', '3', '4']

        train_generator = train_dg.flow_from_directory(
            batch_size=batch_size,
            directory=osp.join(data, 'train'),
            target_size=(224,224),
            class_mode='categorical',
            save_to_dir=osp.join(base, 'verif', 'train'),
#            subset='training',
            shuffle=False,
            color_mode='rgb',
            classes=classes
        )
        valid_generator = valid_dg.flow_from_directory(
            batch_size=batch_size,
            directory=osp.join(data, 'valid'),
            target_size=(224,224),
            class_mode='categorical',
            save_to_dir=osp.join(base, 'verif', 'valid'),
#            subset='validation',
            shuffle=False,
            color_mode='rgb',
            classes=classes
        )

        print("Training model: ")
        print("Devices: ")
        print(device_lib.list_local_devices())

        hist = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,
            validation_data=valid_generator,
            validation_steps=valid_generator.n // valid_generator.batch_size,
            callbacks=[tensorboard],
            epochs=epochs
        )

        hist = self.model.save(osp.join(base, 'models', model_name))
        print("Model saved")
        pickle.dump(hist.history, open(osp.join(base, 'models', 'hist.' + model_name), 'wb'))
        print("Model history saved")


if __name__ == '__main__':

    if len(sys.argv) == 6:
        model = AlexNet()
        model.train(*sys.argv[1:])
    else:
        print("USAGE: prog model_name datadir outdir epochs batch_size")
        print("ARGS PROVIDED: " + str(sys.argv))
