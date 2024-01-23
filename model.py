import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from tensorflow.keras.layers import Input, Flatten, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
from tensorflow.keras.utils import plot_model


def convt2(shape=(112, 112, 3)):
    net = EfficientNetV2B3(include_top=False, input_shape=(50, 200, 3), pooling='avg')
    net.traniable = False
    return net


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    # max_length = 9
    # img_width = 200
    # img_height = 50
    input_img = layers.Input(
        shape=(3, img_height, img_width, 3), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    convmodel = convt2(shape=(img_height, img_width, 3))
    x = TimeDistributed(convmodel)(input_img)

    print(x.shape)
    x = layers.Dense(1550, activation='relu')(x)
    new_shape = (50, 93)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(100, activation="relu", name="dense0")(x)
    x = layers.Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    # x = layers.Dropout(0.2)(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.legacy.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


characters = [' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I',
              'K', 'M', 'N', 'O', 'P', 'S', 'T', 'X', 'Y', '_']
max_length = 9
img_width = 200
img_height = 50
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
# model = build_model()
# model.load_weights('mod.h5')
# prediction_model = keras.models.Model(
#     model.get_layer(name="image").input, model.get_layer(name="dense2").output)
