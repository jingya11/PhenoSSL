# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:20:32 2026

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
from function_phenossl import load_labeled_data, th_type,generate_pseudo_labels, calculate_pseudo_label_stats, aggregate_probabilities_by_category, batch_generator
from model import deepcropmapping

#Load data
dataset_train_export_feature_reshape,labels_train_final=load_labeled_data(r'...\PhenoSSL_jianhan\train_random512.csv')
dataset_val_export_feature_reshape,labels_val_final=load_labeled_data(r'...\PhenoSSL_jianhan\val_random.csv')
dataset_test_export_feature_reshape,_=load_labeled_data(r'...\PhenoSSL_jianhan\test_random_1054.csv')
all_unlabel_reshape,all_unlabel_pheno_resahpe=load_labeled_data(r'...\PhenoSSL_jianhan\Ulabel_80000_new_merged.csv')

#DeepCropMapping
model = deepcropmapping(num_classes=7)
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
logs=[]

pseudo_x=[]
pseudo_y=[]
val_acc_history = {} 
  
batch_size=64
epochs=200
all_steps=512//batch_size 
u=160

epoch_all_th_merge=[];
epoch_all_sample_count_0=[];
epoch_all_sample_count_1=[];
epoch_all_sample_count_2=[];
epoch_all_sample_count_3=[];
epoch_all_sample_count_4=[];
epoch_all_sample_count_5=[];
epoch_all_sample_count_6=[];

samples_count_epoch=[]
samples_prob_min_10_epoch=[]

for epoch in range(epochs):
    if epoch <=1:
        th_0=0.5
        th_1=0.5
        th_2=0.5
        th_3=0.5
        th_4=0.5
        th_5=0.5
        th_6=0.5

    else:
        th_0=th_type(np.array(epoch_prob_0),epoch_all_sample_count_0,epoch);
        th_1=th_type(np.array(epoch_prob_1),epoch_all_sample_count_1,epoch);
        th_2=th_type(np.array(epoch_prob_2),epoch_all_sample_count_2,epoch);
        th_3=th_type(np.array(epoch_prob_3),epoch_all_sample_count_3,epoch);
        th_4=th_type(np.array(epoch_prob_4),epoch_all_sample_count_4,epoch);
        th_5=th_type(np.array(epoch_prob_5),epoch_all_sample_count_5,epoch);
        th_6=th_type(np.array(epoch_prob_6),epoch_all_sample_count_6,epoch);
        
    class_threshold_all=[th_0,th_1,th_2,th_3,th_4,th_5,th_6]    
    samples_count=[];
    samples_prob=[];
    loss_all=0
    loss_sl_all=0
    loss_ssl_all=0
    
    epoch_all_th_merge.append(class_threshold_all) 
    
    labeled_data_gen = batch_generator(dataset_train_export_feature_reshape, labels_train_final, batch_size)
    unlabeled_data_gen=batch_generator(all_unlabel_reshape, all_unlabel_pheno_resahpe, u*batch_size) 
    
    for step in range(all_steps):
        try:
            batch_labeled_x, batch_labeled_y = next(labeled_data_gen)
            batch_unlabeled_x, batch_unlabeled_pheno = next(unlabeled_data_gen)               
        except StopIteration:
            break
        with tf.GradientTape() as tape:

            if epoch==0 :                
                logits_label=model(batch_labeled_x, training=True)
                loss_value_sl=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(batch_labeled_y, axis=-1), logits_label))
                loss_value_ssl=tf.convert_to_tensor(0,dtype='float32')
                # samples_count.append([0,0,0,0,0,0,0]) 
                # samples_prob.append([None,999,999,999,999,999,999])
            else:
                pseudo_x, pseudo_y = generate_pseudo_labels(model, batch_unlabeled_x, batch_unlabeled_pheno,class_threshold_all, batch_size=1024)
                
                logits_label=model(batch_labeled_x, training=True)
                loss_value_sl=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(batch_labeled_y, axis=-1), logits_label))
                
                if len(pseudo_x) > 0:
                    logits_unlabel=model(pseudo_x, training=True)
                    loss_value_ssl=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(pseudo_y,logits_unlabel))
                    
                    counts, probs_list_class = calculate_pseudo_label_stats(pseudo_y, logits_unlabel)                    
                    samples_count.append(counts)
                    samples_prob.append(probs_list_class)    
                
                else:                    
                    loss_value_ssl=tf.convert_to_tensor(0,dtype='float32')
                    samples_count.append([0,0,0,0,0,0,0]) 
                    samples_prob.append([None,None,None,None,None,None,None])
                    
            loss_value=tf.math.add(loss_value_sl, loss_value_ssl) 
            
            loss_sl_all += loss_value_sl
            loss_ssl_all += loss_value_ssl
            loss_all+=loss_value
            
        grads_german = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads_german, model.trainable_variables))
    print(f"Epoch {epoch} finished. Epoch {epoch}/{epochs} - Loss: {loss_all/all_steps:.4f} - loss_ce_s: {loss_sl_all/all_steps:.4f}- loss_ce_t: {loss_ssl_all/all_steps:.4f}")
                   
    if epoch==0:
         
        epoch_prob_0=0;
        epoch_prob_1=0;
        epoch_prob_2=0;
        epoch_prob_3=0;
        epoch_prob_4=0;
        epoch_prob_5=0;
        epoch_prob_6=0 ; 
        
        samples_count_0=0;
        samples_count_1=0;
        samples_count_2=0;
        samples_count_3=0;
        samples_count_4=0;
        samples_count_5=0;
        samples_count_6=0;
    
    else:
        all_pseudo_probs=aggregate_probabilities_by_category(samples_prob)
        
        epoch_prob_0=tfp.stats.percentile(all_pseudo_probs[0], 10.0, interpolation='midpoint')
        epoch_prob_1=tfp.stats.percentile(all_pseudo_probs[1], 10.0, interpolation='midpoint')
        epoch_prob_2=tfp.stats.percentile(all_pseudo_probs[2], 10.0, interpolation='midpoint')
        epoch_prob_3=tfp.stats.percentile(all_pseudo_probs[3], 10.0, interpolation='midpoint')
        epoch_prob_4=tfp.stats.percentile(all_pseudo_probs[4], 10.0, interpolation='midpoint')
        epoch_prob_5=tfp.stats.percentile(all_pseudo_probs[5], 10.0, interpolation='midpoint')
        epoch_prob_6=tfp.stats.percentile(all_pseudo_probs[6], 10.0, interpolation='midpoint')
    
        samples_count_0=tf.reduce_sum(samples_count,axis=0)[0];
        samples_count_1=tf.reduce_sum(samples_count,axis=0)[1];
        samples_count_2=tf.reduce_sum(samples_count,axis=0)[2];
        samples_count_3=tf.reduce_sum(samples_count,axis=0)[3];
        samples_count_4=tf.reduce_sum(samples_count,axis=0)[4];
        samples_count_5=tf.reduce_sum(samples_count,axis=0)[5];
        samples_count_6=tf.reduce_sum(samples_count,axis=0)[6];
       
    epoch_all_sample_count_0.append(samples_count_0)
    epoch_all_sample_count_1.append(samples_count_1)
    epoch_all_sample_count_2.append(samples_count_2)
    epoch_all_sample_count_3.append(samples_count_3)
    epoch_all_sample_count_4.append(samples_count_4)
    epoch_all_sample_count_5.append(samples_count_5)
    epoch_all_sample_count_6.append(samples_count_6)

    samples_count_epoch.append([samples_count_0,samples_count_1,samples_count_2,samples_count_3,samples_count_4,samples_count_5,samples_count_6])
    samples_prob_min_10_epoch.append([epoch_prob_0,epoch_prob_1,epoch_prob_2,epoch_prob_3,epoch_prob_4,epoch_prob_5,epoch_prob_6])
    
    data_sample_count = pd.DataFrame(samples_count_epoch)
    data_sample_count.to_csv(os.path.join(r'...\PhenoSSL_result\label512\U160','count_'+str(epoch)+'.csv'))
    
    data_sample_prob = pd.DataFrame(samples_prob_min_10_epoch)
    data_sample_prob.to_csv(os.path.join(r'...\PhenoSSL_result\label512\U160','prob_'+str(epoch)+'.csv'))
        

    preds = model.predict(dataset_val_export_feature_reshape)
    #acc = np.mean(preds == source_labels)
    matrixes = sm.confusion_matrix(np.argmax(labels_val_final, axis=-1).tolist(), np.argmax(preds,axis=-1).tolist())
    # print(matrixes)
    correct_predictions = np.diag(matrixes).sum()
    total_samples = matrixes.sum()
    overall_accuracy = correct_predictions / total_samples
    print(f"Overall val Accuracy: {overall_accuracy:.4f}")
    
    val_loss_ce=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(labels_val_final, axis=-1), preds))
    val_loss_all=val_loss_ce
    print(f"val loss: {val_loss_all:.4f} ")
    
    #model.load_weights(r'')
    preds= model.predict(dataset_test_export_feature_reshape)
    matrixes = sm.confusion_matrix(dataset_test_export_feature_label[:,0:1].tolist(), np.argmax(preds,axis=-1).tolist())
    # print(matrixes)
    correct_predictions = np.diag(matrixes).sum()
    total_samples = matrixes.sum()
    overall_accuracy_2 = correct_predictions / total_samples
    print(f"Overall test Accuracy: {overall_accuracy_2:.4f} ")
        

    model.save_weights(r'...\PhenoSSL_result\label512\U160\Model_PhenoSSL_512u160_'+str(epoch)+'.h5')
    logs.append({
            "epoch": epoch,
            "train_loss": loss_all/all_steps,
            "train_loss_ce_s": loss_sl_all/all_steps,            
            "train_loss_ce_t": loss_ssl_all/all_steps,
            "val_acc": overall_accuracy,
            "test_acc": overall_accuracy_2,
            'val_loss':val_loss_all
        })
    val_acc_history[epoch] = overall_accuracy
pd.DataFrame(logs).to_csv(r'...\PhenoSSL_result\label512\U160\Model_PhenoSSL_512u160_acc.csv', index=False)       
th_sample_prob = pd.DataFrame(epoch_all_th_merge)    
th_sample_prob.to_csv(os.path.join(r'...\PhenoSSL_result\label512\U160','th_512u160.csv'))                      
                
  