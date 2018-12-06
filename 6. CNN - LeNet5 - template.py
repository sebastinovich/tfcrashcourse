import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import random
import os

dataPath = "temp/"
if not os.path.exists(dataPath):
    os.makedirs(dataPath)
input = input_data.read_data_sets(dataPath, one_hot=True)

img_size = 28

trX, trY, teX, teY = input.train.images, \
                     input.train.labels, \
                     input.test.images,  \
                     input.test.labels

trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.Sequential([  
    #TODO - tf.keras.layers.Conv2D 
    #TODO - tf.keras.layers.MaxPooling2D 
    #TODO - tf.keras.layers.Conv2D 
    #TODO - tf.keras.layers.MaxPooling2D 
    #TODO - tf.keras.layers.Conv2D
    tf.keras.layers.Flatten(),
    #TODO - tf.keras.layers.Dense
    #TODO - tf.keras.layers.Dense
])

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

model.fit(trX, trY, epochs=10, batch_size=32)
model.evaluate(teX, teY)
