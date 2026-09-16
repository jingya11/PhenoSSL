# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:20:46 2026

@author: yangj
"""
import pandas as pd
from osgeo import gdal
import os
import h5py
from datetime import datetime
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Flatten,Bidirectional,Dropout,Conv1D,MaxPooling1D,GlobalAveragePooling1D,MultiHeadAttention,LayerNormalization,Embedding,Input,Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
import tensorflow_probability as tfp

def load_labeled_data(path, n_classes=7):
    data = np.array(pd.read_csv(path))
    
    # Label: columns: 27
    y = to_categorical(data[:, 27], n_classes)

    # Features: columns 28–72
    x = data[:, 28:73]

    # (samples, 15, 2)
    x = np.stack(
        (x[:, :15], x[:, 15:30]),
        axis=2
    ).astype('float64')

    return x, y

def th_type(th_input, s_count, num_temp):

    if num_temp < 2 or th_input <= 0.55:
        return 0.5
    
    prev_count = s_count[num_temp-1]
    prev_2_count = s_count[num_temp-2]
    
    if prev_2_count==0:
        return 0.5
    
    relative_change = np.absolute(prev_count - prev_2_count) / prev_2_count
    
    condition_cal =((prev_count > prev_2_count) or 
                     (relative_change < 0.2))
    

    threshold_rules = [
        ([0.55, 0.6], 0.55 if condition_cal else 0.5),
        ([0.6, 0.65], 0.6 if condition_cal else 0.55),
        ([0.65, 0.7], 0.65 if condition_cal else 0.6),
        ([0.7, 1], 0.7 if condition_cal else 0.65)
    ]
    

    for range_select, output in threshold_rules:
        if range_select[0] < th_input <= range_select[1]:
            return output
 
def generate_pseudo_labels(model, x_unlabel,pheno_unlabel,class_threshold, batch_size=1024):
    preds = []
    for i in range(0, len(x_unlabel), batch_size):
        batch = x_unlabel[i:i+batch_size]
        pred= model(batch, training=False)
        preds.append(pred)
    preds = tf.concat(preds, axis=0)
    pseudo_labels_prob = tf.reduce_max(preds, axis=1)
    pseudo_labels = tf.argmax(preds, axis=1)     
    
    threshold_tensors = [tf.convert_to_tensor(th, dtype=tf.float32) for th in class_threshold]
    
    class_masks = []
    for class_idx, threshold in enumerate(threshold_tensors):
        class_mask = tf.equal(pseudo_labels, class_idx)
        prob_mask = pseudo_labels_prob > threshold
        class_masks.append(class_mask & prob_mask)
      
    class_masks_tensor = tf.stack(class_masks, axis=0)
    final_mask =  tf.reduce_any(class_masks_tensor, axis=0)

    pheno_match_mask = tf.equal(pseudo_labels, tf.argmax(pheno_unlabel, axis=1))
    final_mask2 = final_mask & pheno_match_mask
    
    filtered_data = tf.boolean_mask(x_unlabel, final_mask2)
    filtered_labels = tf.boolean_mask(pseudo_labels, final_mask2)
    
    return filtered_data, filtered_labels

def calculate_pseudo_label_stats(pseudo_labels, prediction_probabilities, num_classes=7):

    counts = []
    class_probs_list = []
    
    pseudo_labels_size = tf.shape(pseudo_labels)[0]
    indices = tf.stack([tf.range(pseudo_labels_size,dtype=tf.int64), pseudo_labels], axis=1)
    label_corresponding_probs = tf.gather_nd(prediction_probabilities, indices)
    
    for class_idx in range(num_classes):

        class_mask = (pseudo_labels == class_idx)
        class_count = tf.reduce_sum(tf.cast(class_mask, tf.int32))
        counts.append(class_count.numpy())
        
        if class_count.numpy() > 0:
            class_probs = tf.boolean_mask(label_corresponding_probs, class_mask)
            class_probs_list.append(class_probs.numpy().tolist())
        else:
            class_probs_list.append(None)
    
    return counts, class_probs_list
        
def aggregate_probabilities_by_category(samples_prob, num_categories=7):
    category_probs = [[] for _ in range(num_categories)]
    
    for step_probs in samples_prob:
        for i, prob in enumerate(step_probs):
            if prob is not None:
                valid_probs = [p for p in (prob if isinstance(prob, list) else [prob]) if p is not None]
                category_probs[i].extend(valid_probs)    
    return category_probs

def batch_generator(x, y, batch_size, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), batch_size):
        batch_idx = idx[i:i+batch_size]
        batch_x = tf.gather(x, batch_idx)
        batch_y = tf.gather(np.array(y,dtype=float), batch_idx)
        yield batch_x,batch_y