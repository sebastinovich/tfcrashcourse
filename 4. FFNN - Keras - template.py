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

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(200, activation=tf.nn.sigmoid),
  #TODO
  #TODO
  #TODO
  #TODO
])

model.compile(optimizer=#TODO,
              loss=#TODO,
              metrics=#TODO)

model.fit(#TODO, 
            epochs=10, 
            batch_size=100, 
            validation_data=#TODO)
model.evaluate(#TODO)