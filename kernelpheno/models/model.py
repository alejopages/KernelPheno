import keras
from keras.self.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np

import pickle
import subprocess as sp

np.random.seed(1000)

class AlexNet:

    def __init__(self):
        #Instantiate an empty self.model
        self.self.model = Sequential()

        # 1st Convolutional Layer
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


    def train(self, model_name, data, outdir, epochs,
              label_fname='annotations.csv', batch_size=32):

        sp.run(['mkdir', '-p', osp.join(data, 'train')])
        sp.run(['mkdir', '-p', osp.join(data, 'train', 'tb')])
        sp.run(['mkdir', '-p', osp.join(data, 'train', 'verif')])

        tensorboard = Tensorboard(log_dir=osp.join(outdir, 'train', 'tb'))
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2
        )
        train_dg.flow_from_directory(
            directory=data,
            target_size=(224,224,3),
            class_mode='other',
            save_to_dir=osp.join(outdir, 'train', 'verif'),
            subset='training',
            classes=range(1,6)
        )
        valid_dg.flow_from_directory(
            directory=data,
            target_size=(224,224,3),
            class_mode='other',
            save_to_dir=osp.join(outdir, 'train', 'verif'),
            subset='validation',
            classes=range(1,6)
        )

        print("Training model")
        hist = self.model.fit_generator(
            train_dg,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=valid_dg,
            validation_steps=valid_dg.samples // batch_size,
            callbacks=[tensorboard],
            epochs=epochs
        )
        print("Model Trained")

        hist = self.model.save(osp.join(outdir, 'train', model_name))
        print("Model has been saved")
        pickle.dump(hist.history, open(osp.join(data, 'train', "hist." + model_name, 'wb')))
        print("Model history saved")
