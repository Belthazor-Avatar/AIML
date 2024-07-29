# Data Science Tools
import numpy as np
import pandas as pd

# Analysis tools
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras, config, data
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.metrics import Precision, Recall

# System API
import os
import pathlib
import random
import string
from psutil import virtual_memory
import cv2


batch_size = 32
epochs = 100
#img_size = (299, 299)
img_size = (224, 224)
input_shape = (img_size[0], img_size[1], 3)
data_dir = './Dataset/'
data_path = pathlib.Path(data_dir)
#early_stop = EarlyStopping(min_delta=0.01, patience=14, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', mode='max',
                           min_delta=0.01, patience=16, verbose=1)
learning_rate = 0.001
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_accuracy',
    factor = 0.5,
    #patience = 3,
    patience = 4,
    verbose = 1,
    min_lr = 0.00001
)


@register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


data_augmentation = Sequential(
  [
      # layers.RandomFlip("horizontal_and_vertical"),  # flip images
      layers.RandomFlip("horizontal", input_shape=input_shape),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
  ], name="data_augmentation"
)

optimizer_a = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model_a.compile(optimizer=optimizer_a,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy', f1_m])

history_a = model_a.fit(
	X_train,
        y_train,
        shuffle=True,
        validation_data=(X_val, y_val),
        epochs=epochs,
	callbacks=[lr_reduction, early_stop]
)

y_pred_a = None
if 'model_a' in run or 'run_all' in run:
    with tf.device('/GPU:0'):
        X_test_gpu = tf.constant(X_test)
        y_pred_a = model_a.predict(X_test_gpu)

labels_2 = {}
for k,v in labels.items():
    labels_2[v] = k

pred_a_label = []
pred_a_label_index = []
if 'model_a' in run or 'run_all' in run:
    for i in y_pred_a.argmax(axis=1):
        pred_a_label.append(labels_2[i])
        pred_a_label_index.append(i)


if 'model_a' in run or 'run_all' in run:
    model_a_test_loss, model_a_test_acc, model_a_test_f1_m = model_a.evaluate(X_val, y_val)

if 'model_a' in run or 'run_all' in run:
    print(
        f"\tLoss: {model_a_test_loss}\n",
        f"\tAccuracy: {model_a_test_acc}\n",
        f"\tPrecision: {model_a_test_f1_m}\n"
    )


if 'model_a' in run or 'run_all' in run:
    print(classification_report(y_test.argmax(axis=1), model_a.predict(X_test).argmax(axis=1), zero_division=0))
