# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:00:33 2026

@author: yangj
"""

from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
import tensorflow as tf

#DCM-DeepCropMapping
#Reference: Xu J, Zhu Y, Zhong R, Lin Z, Xu J, Jiang H, Huang J, Li H, Lin T. 2020, 
#DeepCropMapping: A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping.
#Remote Sensing of Environment, 247: 111946. DOI: 10.1016/j.rse.2020.111946.
#DeepCropMapping

def deepcropmapping(num_classes=7):
    inputs = Input(shape=(15, 2))
    hidden_1 = Bidirectional(LSTM(256, return_sequences=True))(inputs)
    dropout_1 = Dropout(0.5)(hidden_1)
    hidden_2 = Bidirectional(LSTM(256, return_sequences=True))(dropout_1)
    attention_weight = softmax(Dense(1, name="att_weight")(hidden_2),axis=1)
    attention_mul = tf.matmul(tf.transpose(hidden_2, [0, 2, 1]),attention_weight)
    flatten = Flatten()(attention_mul)
    predictions = Dense(num_classes,activation="softmax")(flatten)

    model = Model(inputs=inputs, outputs=predictions)

    return model