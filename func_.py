import numpy as np
import pandas as pd
import dask.dataframe as dk
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from dask_ml.model_selection import train_test_split 
import copy
import sys
import random
import config
import threading
from warnings import simplefilter
from datetime import datetime
from sklearn import metrics

from agent_class import Agent
from message import Message
from utils.dp_mechanisms import laplace
import utils.diffie_hellman as dh


from sklearn.linear_model import SGDClassifier
# # load từng batch
def dask_to_tf_dataset(dask_df, batch_size=128, num_classes=10): 
    def generator():
        for batch in dask_df.to_delayed():
            batch=batch.compute()  
            if batch.empty:
                continue

            X = batch.drop(columns='label').values.astype(np.float32)
            y = batch['label'].values
            y_onehot = to_categorical(y, num_classes=num_classes)  

            num_splits = max(1, len(X) // batch_size)  # Đảm bảo không chia nhỏ quá mức
            X_batches = np.array_split(X, num_splits)
            y_batches = np.array_split(y_onehot, num_splits)

            for X_batch, y_batch in zip(X_batches, y_batches):
                yield X_batch, y_batch
                
    output_signature = ( 
        tf.TensorSpec(shape=(None, 46), dtype=tf.float32), 
        tf.TensorSpec(shape=(None, 10), dtype=tf.int32),
    )
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

def bar_graph(data, feature):
    data[feature].value_counts().plot(kind="bar")
    

# def laplace_gamma_dp(weights, shape_param):
    
#     # Bước 3: Sinh nhiễu theo phân phối Gamma
#     gamma_1 = np.random.gamma(shape_param, lambda_param, size=weights.shape)
#     gamma_2 = np.random.gamma(shape_param, lambda_param, size=weights.shape)
    
#     # Bước 4: Tạo nhiễu theo phân phối Laplace Gamma
#     laplace_gamma_noise = gamma_1 - gamma_2
    
#     # Bước 5: Thêm nhiễu vào trọng số ban đầu
#     encrypted_weights = weights + laplace_gamma_noise
    
#     return encrypted_weights

def laplace_gamma_lamda(pre_val, post_val):
    # Bước 1: Tính toán độ nhạy Δ
    delta = np.max(np.abs(pre_val - post_val))
    
    # Bước 2: Xác định tham số scale λ
    lambda_param = delta / np.mean(pre_val) if np.mean(pre_val) != 0 else 1.0
    
    return lambda_param


def homomorphic_encryption(offset_encrypted_weights):
    """
    Áp dụng lớp mã hóa thứ ba bằng phương pháp Homomorphic Encryption (HE).
    """
    operations = ['add', 'subtract', 'multiply', 'divide']
    operation = np.random.choice(operations)
    factor = np.random.randint(1, 10)
    
    if operation == 'divide' and np.any(offset_encrypted_weights == 0):
        operation = np.random.choice(['add', 'subtract', 'multiply'])
    
    he_encrypted_weights = np.copy(offset_encrypted_weights)
    
    for i in range(he_encrypted_weights.shape[0]):
        for j in range(he_encrypted_weights.shape[1]):
            value = he_encrypted_weights[i, j]
            if operation == 'add':
                he_encrypted_weights[i, j] = value + factor
            elif operation == 'subtract':
                he_encrypted_weights[i, j] = value - factor
            elif operation == 'multiply':
                he_encrypted_weights[i, j] = value * factor
            elif operation == 'divide':
                he_encrypted_weights[i, j] = value / factor if factor != 0 else value
    
    return he_encrypted_weights