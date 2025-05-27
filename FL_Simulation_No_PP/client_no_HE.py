import copy
import sys
import os
import random
import threading
from warnings import simplefilter
from datetime import datetime, timedelta
from sklearn import metrics

import numpy as np
import tensorflow as tf
import pandas as pd
import tenseal as ts  # Giữ lại để tương thích với mã khởi tạo

# TensorFlow và Keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
# Giả lập module dp_mechanisms (thay bằng module thực tế của bạn)
from dp_mechanisms import laplace

# Từ điển độ trễ
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Client:
    def __init__(self, client_name, data_train, data_val, data_test, steps_per_epoch, val_steps, test_steps, active_clients_list):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.agent_dict = {}
        self.temp_dir1 = "federated_learning_log/" + datetime.now().strftime("Month%m-Day%d-%Hh-%Mp")
        os.makedirs(self.temp_dir1, exist_ok=True)
        self.temp_dir = self.temp_dir1 + "/" + client_name + "_log"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Global
        self.global_weights = {}
        self.global_biases = {}
        self.global_accuracy = {}
        self.global_loss = {}

        # Local
        self.model = self.init_model()
        self.local_weights = {}
        self.local_biases = {}
        self.local_accuracy = {}
        self.local_loss = {}
        self.compute_times = {}  # Thời gian xử lý trọng số
        self.convergence = 0  
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = val_steps
        self.test_steps = test_steps

        # Khởi tạo LATENCY_DICT
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name] = {}
        if 'server_0' not in LATENCY_DICT.keys():
            LATENCY_DICT['server_0'] = {}
        LATENCY_DICT['server_0'] = {client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds=np.random.random())

    def get_clientID(self):
        return self.client_name

    def set_agentsDict(self, agents_dict):
        self.agent_dict = agents_dict

    def get_steps_per_epoch(self):
        print("Train steps: ", self.steps_per_epoch)

    def get_validation_steps(self):
        print("Val steps: ", self.validation_steps)

    def get_test_steps(self):
        print("Test steps: ", self.test_steps)
        
    def get_temp_dir(self):
        temp = self.temp_dir1
        print(temp)
        return temp

    # Khởi tạo mô hình
    def init_model(self):
        features, labels = next(iter(self.data_train))
        input_shape = (features.shape[1], 1)

        # Mô hình binary classification
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.05)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.05)),
            layers.Dropout(0.7),   
            layers.BatchNormalization(),
            layers.Dense(1, activation='sigmoid')
        ])
        adam_optimizer = optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer= adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Huấn luyện mô hình
    def model_fit(self, iteration):
        file_path = self.temp_dir + "/Iteration_" + str(iteration) + ".csv"
        file_path_model = self.temp_dir + "/model_" + str(iteration) + ".keras"
        csv_logger = CSVLogger(file_path, append=True)

        if iteration > 1:
            print(f"{iteration} {self.client_name} Model update params with global weights!")
            index = 0
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    layer.set_weights([self.global_weights[iteration-1][index], self.global_biases[iteration-1][index]])
                    index += 1

        self.model.fit(self.data_train, epochs=5, validation_data=self.data_val, validation_steps=self.validation_steps,
                       steps_per_epoch=self.steps_per_epoch, verbose=1, callbacks=[csv_logger])
    
        self.model.save(file_path_model)
        print("Come done model fit\n")

        weights = []
        biases = []
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                weights.append(layer.get_weights()[0])
                biases.append(layer.get_weights()[1])
        return weights, biases

    # Tạo và xử lý trọng số
    def proc_weights(self, message):
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']

        weights, biases = self.model_fit(iteration)

        # Cập nhật local weights
        self.local_weights[iteration] = weights
        self.local_biases[iteration] = biases
        if iteration > 1:
            del self.local_weights[iteration-1]  
            del self.local_biases[iteration-1]

        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time
            
        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']
        body = {
            'weights': weights,  
            'biases': biases,
            'iter': iteration,
            'compute_time': compute_time,
            'simulated_time': simulated_time
        }

        print(self.client_name + " End Produce Weights")
        msg = Message(sender_name=self.client_name, recipient_name=self.agent_dict['server']['server_0'], body=body)
        return msg

    # Nhận trọng số từ server
    def recv_weights(self, message):
        body = message.body
        iteration, simulated_time = body['iteration'], body['simulated_time']

        return_weights = body['weights']
        return_biases = body['biases']

        self.global_weights[iteration], self.global_biases[iteration] = return_weights, return_biases
        
        self.save_global_model(iteration)
        if iteration > 1:
            del self.global_weights[iteration-1]  
            del self.global_biases[iteration-1]

        # Đánh giá mô hình
        self.local_accuracy[iteration], self.local_loss[iteration] = self.evaluate_accuracy(self.local_weights[iteration], self.local_biases[iteration])
        self.global_accuracy[iteration], self.global_loss[iteration] = self.evaluate_accuracy(self.global_weights[iteration], self.global_biases[iteration])
        if iteration > 2:
            del self.local_accuracy[iteration-2]  
            del self.local_loss[iteration-2]
            del self.global_accuracy[iteration-2]
            del self.global_loss[iteration-2]
        # Lưu lịch sử
        history1 = {'global_acc': self.global_accuracy[iteration], 'global_loss': self.global_loss[iteration]}
        history2 = {'local_acc': self.local_accuracy[iteration], 'local_loss': self.local_loss[iteration]}
        history3 = {'simulation_time': simulated_time + LATENCY_DICT[self.client_name]['server_0']}
        file_his_local = self.temp_dir + "/local_val.csv"
        file_his_global = self.temp_dir + "/global_val.csv"
        file_simulated_time = self.temp_dir + "/simulation_time.csv"
        pd.DataFrame([history1]).to_csv(file_his_global, index=False, header= not os.path.exists(file_his_global), mode='a')
        pd.DataFrame([history2]).to_csv(file_his_local, index=False, header= not os.path.exists(file_his_local), mode='a')
        pd.DataFrame([history3]).to_csv(file_simulated_time, index=False, header= not os.path.exists(file_simulated_time), mode='a')
        
        # Kiểm tra hội tụ
        converged = self.check_convergence(iteration)

        args = [self.client_name, iteration, self.local_accuracy[iteration], self.local_loss[iteration],
                self.global_accuracy[iteration], self.global_loss[iteration], self.compute_times[iteration], simulated_time]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'local accuracy: {} \n' \
                           'local loss: {} \n' \
                           'global accuracy: {} \n' \
                           'global_loss: {} \n' \
                           'local compute time: {} \n' \
                           'Simulated time to receive global weights: {} \n \n'

        print("Arguments: ", iteration_report.format(*args))
        del self.compute_times[iteration]
        msg = Message(sender_name=self.client_name, recipient_name='server_0',
                      body={'converged': converged, 'simulated_time': simulated_time + LATENCY_DICT[self.client_name]['server_0']})
        return msg

    # Đánh giá độ chính xác
    def evaluate_accuracy(self, weights, biases):
        index = 0
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                layer.set_weights([weights[index], biases[index]])
                index += 1
        loss, accuracy = self.model.evaluate(self.data_test, steps=self.test_steps)
        return accuracy, loss

    # Lưu mô hình global
    def save_global_model(self, iteration):
        index = 0
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                layer.set_weights([self.global_weights[iteration][index], self.global_biases[iteration][index]])
                index += 1
        model_path = os.path.join(self.temp_dir, f"global_model_iter_{iteration}.keras")
        self.model.save(model_path)
        print(f"Đã lưu global model cho iteration {iteration} tại {model_path}")

    # Kiểm tra hội tụ
    def check_convergence(self, iteration):
        if iteration < 2:
            return False
        
        acc_diff = abs(self.global_accuracy[iteration] - self.global_accuracy[iteration-1])
        loss_diff = abs(self.global_loss[iteration] - self.global_loss[iteration-1])
        
        if acc_diff < 0.01 and loss_diff < 0.05 and self.global_accuracy[iteration] > 0.9 and self.global_loss[iteration] < 0.1:
            self.convergence += 1
        else: 
            self.convergence = 0
        if self.convergence > 2:
            return True
        return False

    # Xóa client
    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration = body['removing_clients'], body['simulated_time'], body['iteration']
        print(f'[{self.client_name}] Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        return None
    