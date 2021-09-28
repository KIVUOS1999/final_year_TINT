import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dense, Input, Flatten, Conv2D, MaxPool2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam,RMSprop

import os
import cv2

IMG_HIGHT = 64
IMG_WIDTH = 64

def load_file(n = 'numpy_saves/face64X64_filtered.npy'):
    X = np.load(n)
    return X

def discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(IMG_HIGHT,IMG_WIDTH,1)))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3,3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model

def generator(n):
    model = Sequential()
    n_nodes = 128 * 16 * 16

    model.add(Dense(n_nodes, input_dim=(n)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 128)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

    model.summary()
    return model

def gan(discriminator, generator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer = Adam(lr = 2e-4, beta_1 = 0.5), loss = "binary_crossentropy")
    return model

def preview(name, generator):
    fig, axis = plt.subplots(10,10, figsize = (20,20))
    for i in range(10):
        for j in range(10):
            axis[i,j].imshow(
                generator.predict(np.random.randn(100).reshape(1,100)).reshape(IMG_HIGHT,IMG_WIDTH), cmap= 'gray'
            )
    plt.savefig("temp/"+'Gray'+str(name)+".png")
    plt.close()
    