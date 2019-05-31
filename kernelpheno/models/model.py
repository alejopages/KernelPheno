from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard

import pandas as pd
import numpy as np

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
        self.model.add(Dense(17))
        self.model.add(Activation('softmax'))

        self.model.summary()

        # Compile the self.model
        self.model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer='adam',
            metrics=['accuracy']
        )


    def train(self, model_name, data, outdir, epochs, batch_size):

        epochs = int(epochs)
        batch_size = int(batch_size)

        base = osp.join(outdir, 'train')

        sp.run(['mkdir', '-p', base])
        sp.run(['mkdir', '-p', osp.join(base, 'models')])
        sp.run(['mkdir', '-p', osp.join(base, 'tb')])
        sp.run(['mkdir', '-p', osp.join(base, 'verif')])

        tensorboard = TensorBoard(log_dir=osp.join(base, 'tb'))
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2
        )
        train_dg.flow_from_directory(
            directory=data,
            target_size=(224,224,3),
            class_mode='other',
            save_to_dir=osp.join(base, 'verif'),
            subset='training',
            classes=range(1,6)
        )
        valid_dg.flow_from_directory(
            directory=data,
            target_size=(224,224,3),
            class_mode='other',
            save_to_dir=osp.join(base, 'verif'),
            subset='validation',
            classes=range(1,6)
        )

        hist = self.model.fit_generator(
            train_dg,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=valid_dg,
            validation_steps=valid_dg.samples // batch_size,
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
