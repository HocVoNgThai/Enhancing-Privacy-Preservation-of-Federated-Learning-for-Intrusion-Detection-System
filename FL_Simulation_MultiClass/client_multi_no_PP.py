import copy
import sys
import os
import random
import threading
from warnings import simplefilter
from datetime import datetime, timedelta
from sklearn import metrics
import gc

import numpy as np
import tensorflow as tf
import pandas as pd
import tenseal as ts

# TensorFlow và Keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers, callbacks

# Giả lập module dp_mechanisms (thay bằng module thực tế của bạn)
from dp_mechanisms import laplace

LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return f"Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Client:
    def __init__(self, client_name, data_train, data_val, data_test, steps_per_epoch, val_steps, test_steps, active_clients_list):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.agent_dict = {}
        self.temp_dir1 = os.path.join("multiclass_FL_log_no_PP", datetime.now().strftime("Month%m-Day%d-%Hh-%Mp"))
        os.makedirs(self.temp_dir1, exist_ok=True)
        self.temp_dir = os.path.join(self.temp_dir1, f"{client_name}_log")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.batch_size = 512

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
        self.compute_times = {}
        self.convergence = 0

        # DP parameters
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = val_steps
        self.test_steps = test_steps

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

    def set_steps_per_epoch(self, steps_per_epoch=50):
        self.steps_per_epoch = steps_per_epoch

    def get_steps_per_epoch(self):
        print("Train steps:", self.steps_per_epoch)

    def set_validation_steps(self, validation_steps):
        self.validation_steps = validation_steps

    def get_validation_steps(self):
        print("Val steps:", self.validation_steps)

    def set_test_steps(self, test_steps):
        self.test_steps = test_steps

    def get_test_steps(self):
        print("Test steps:", self.test_steps)

    def get_temp_dir(self):
        return self.temp_dir1
        
    def check_data_distribution(self, data, data_name):
        labels = []
        for batch in data.take(50):  # Giảm số batch kiểm tra để tiết kiệm RAM
            _, batch_labels = batch
            labels.extend(batch_labels.numpy())
            if len(labels) >= 5000:  # Giảm số lượng mẫu kiểm tra
                break
        labels = np.array(labels)
        print(f"[{self.client_name}] Phân bố label trong {data_name}: {np.bincount(labels.astype(int))}")

    def init_model(self):
        input_shape = (46, 1)  # Sử dụng giá trị cố định
        model = models.Sequential([
            layers.Input(shape=input_shape),
            #layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.05)),
            layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), 
            layers.Dropout(0.5), 
            layers.BatchNormalization(),
            layers.Dense(5, activation='softmax')
        ])
        adam_optimizer = optimizers.Adam(learning_rate=1e-4)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Giảm patience
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def model_fit(self, iteration):
        file_path = os.path.join(self.temp_dir, f"Iteration_{iteration}.csv")
        file_path_model = os.path.join(self.temp_dir, f"model_{iteration}.keras")
        csv_logger = CSVLogger(file_path, append=True)

        if iteration > 1:
            index = 0
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    layer.set_weights([self.global_weights[iteration-1][index], self.global_biases[iteration-1][index]])
                    index += 1

        self.model.fit(self.data_train, epochs= 10, validation_data=self.data_val, validation_steps=self.validation_steps,
                       steps_per_epoch=self.steps_per_epoch, verbose=1, callbacks=[csv_logger])

        self.model.save(file_path_model)
        # Giải phóng mô hình tạm sau khi lưu
        del self.model
        gc.collect()
        self.model = keras.models.load_model(file_path_model)  # Tải lại mô hình từ file
        print("Done model fit\n")

        weights = []
        biases = []
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                weights.append(layer.get_weights()[0])
                biases.append(layer.get_weights()[1])
        return weights, biases

    def proc_weights(self, message):
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body.get('lock', None), body['simulated_time']

        weights, biases = self.model_fit(iteration)

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

        print(f"{self.client_name} End Produce Weights")
        msg = Message(sender_name=self.client_name, recipient_name=self.agent_dict['server']['server_0'], body=body)
        return msg

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


        self.local_accuracy[iteration], self.local_loss[iteration] = self.evaluate_accuracy(self.local_weights[iteration], self.local_biases[iteration])
        self.global_accuracy[iteration], self.global_loss[iteration] = self.evaluate_accuracy(self.global_weights[iteration], self.global_biases[iteration])
        if iteration > 2:
            del self.local_accuracy[iteration-2]
            del self.local_loss[iteration-2]
            del self.global_accuracy[iteration-2]
            del self.global_loss[iteration-2]

        history1 = {'global_acc': self.global_accuracy[iteration], 'global_loss': self.global_loss[iteration]}
        history2 = {'local_acc': self.local_accuracy[iteration], 'local_loss': self.local_loss[iteration]}
        history3 = {'simulation_time': simulated_time}
        file_his_local = os.path.join(self.temp_dir, "local_val.csv")
        file_his_global = os.path.join(self.temp_dir, "global_val.csv")
        file_simulated_time = os.path.join(self.temp_dir, "simulation_time.csv")
        pd.DataFrame([history1]).to_csv(file_his_global, index=False, header=not os.path.exists(file_his_global), mode='a')
        pd.DataFrame([history2]).to_csv(file_his_local, index=False, header=not os.path.exists(file_his_local), mode='a')
        pd.DataFrame([history3]).to_csv(file_simulated_time, index=False, header=not os.path.exists(file_simulated_time), mode='a')

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

    def evaluate_accuracy(self, weights, biases):
        index = 0
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                layer.set_weights([weights[index], biases[index]])
                index += 1
        loss, accuracy = self.model.evaluate(self.data_test, steps=self.test_steps)
        return accuracy, loss

    def save_global_model(self, iteration):
        index = 0
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                layer.set_weights([self.global_weights[iteration][index], self.global_biases[iteration][index]])
                index += 1
        model_path = os.path.join(self.temp_dir, f"global_model_iter_{iteration}.keras")
        self.model.save(model_path)
        print(f"Đã lưu global model cho iteration {iteration} tại {model_path}")

    def check_convergence(self, iteration):
        if iteration < 2:
            return False
        
        acc_diff = self.global_accuracy[iteration] - self.global_accuracy[iteration-1]
        loss_diff = self.global_loss[iteration] - self.global_loss[iteration-1]
        
        if 0 < acc_diff < 0.01 and 0 < loss_diff < 0.01 and self.global_accuracy[iteration] > 0.95 and self.global_loss[iteration] < 0.1:
            self.convergence += 1
        else: 
            self.convergence = 0
    
        if self.convergence > 6:
            return True
        return False

    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration = body['removing_clients'], body['simulated_time'], body['iteration']
        print(f'[{self.client_name}] Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        gc.collect()  # Giải phóng bộ nhớ sau khi xóa client
        return None