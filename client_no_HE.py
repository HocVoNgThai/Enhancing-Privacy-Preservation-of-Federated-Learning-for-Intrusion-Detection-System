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
from tensorflow.keras import layers, models
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
    def __init__(self, client_name, data_train, data_val, data_test, steps_per_epoch, val_steps, test_steps, active_clients_list, he_context):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.agent_dict = {}
        self.temp_dir = "federated_learning_log/" + datetime.now().strftime("%Hh-%Mp-Month%m-Day%d")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_dir = self.temp_dir + "/" + client_name + "_log"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Global
        self.global_weights = {}
        self.global_biases = {}
        self.global_accuracy = {}
        self.global_loss = {}
        self.global_test_acc = {}
        self.global_test_loss = {}

        # Local
        self.model = self.init_model()
        self.local_weights = {}
        self.local_weights_shape = []
        self.local_biases_shape = []
        self.local_biases = {}
        self.local_accuracy = {}
        self.local_loss = {}
        self.compute_times = {}  # Thời gian xử lý trọng số
        self.he_context = he_context  # Giữ lại để tương thích
        self.convergence = 0  # Số lần hội tụ
        self.unconvergence = 0

        # DP parameters
        self.alpha = 1.0
        self.epsilon = 1.0  # Tăng lên 1.0 để giảm nhiễu
        self.mean = 0
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = val_steps
        self.test_steps = test_steps
        self.local_weights_noise = {}
        self.local_biases_noise = {}

        # Khởi tạo LATENCY_DICT
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name] = {}
        if 'server_0' not in LATENCY_DICT.keys():
            LATENCY_DICT['server_0'] = {}
        LATENCY_DICT['server_0'] = {client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds=np.random.random())

        # Kiểm tra phân bố dữ liệu
        #self.check_data_distribution(self.data_train, "data_train")
        #self.check_data_distribution(self.data_test, "data_test")

    def get_clientID(self):
        return self.client_name

    def set_agentsDict(self, agents_dict):
        self.agent_dict = agents_dict

    def set_steps_per_epoch(self, steps_per_epoch=50):
        self.steps_per_epoch = steps_per_epoch

    def get_steps_per_epoch(self):
        print("Train steps: ", self.steps_per_epoch)

    def set_validation_steps(self, validation_steps):
        self.validation_steps = validation_steps

    def get_validation_steps(self):
        print("Val steps: ", self.validation_steps)

    def set_test_steps(self, test_steps):
        self.test_steps = test_steps

    def get_test_steps(self):
        print("Test steps: ", self.test_steps)

    # Kiểm tra phân bố dữ liệu (đã sửa)
    def check_data_distribution(self, data, data_name):
        labels = []
        for batch in data:
            _, batch_labels = batch
            labels.extend(batch_labels.numpy())
            if len(labels) >= 10000:
                break
        labels = np.array(labels)  # Chuyển đổi thành mảng NumPy
        print(f"[{self.client_name}] Phân bố label trong {data_name}: {np.bincount(labels.astype(int))}")

    # Khởi tạo mô hình
    def init_model(self):
        features, labels = next(iter(self.data_train))
        input_shape = (features.shape[1], 1)

        # Mô hình binary classification
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            #layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu"),
            layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            #layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu"),
            layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.6),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Thêm nhiễu DP
    def add_gamma_noise(self, local_weights, local_biases, iteration):
        weights_dp_noise = []
        biases_dp_noise = []
        sensitivity = 2 / (len(self.active_clients_list) * self.steps_per_epoch * self.alpha)
        for weight in local_weights:
            if abs(weight) > 1e-15:
                noise = laplace(mean=self.mean, sensitivity=sensitivity, epsilon=self.epsilon)
                weights_dp_noise.append(noise)
            else:
                weights_dp_noise.append(0)
        for bias in local_biases:
            if abs(bias) > 1e-15:
                noise = laplace(mean=self.mean, sensitivity=sensitivity, epsilon=self.epsilon)
                biases_dp_noise.append(noise)
            else:
                biases_dp_noise.append(0)

        self.local_weights_noise[iteration] = weights_dp_noise
        self.local_biases_noise[iteration] = biases_dp_noise
        print(f"[{self.client_name}] Noise stats - Weights: min={np.min(weights_dp_noise)}, max={np.max(weights_dp_noise)}")
        print(f"[{self.client_name}] Noise stats - Biases: min={np.min(biases_dp_noise)}, max={np.max(biases_dp_noise)}")

        weights_with_noise = local_weights + weights_dp_noise
        biases_with_noise = local_biases + biases_dp_noise
        return np.array(weights_with_noise), np.array(biases_with_noise)

    # Huấn luyện mô hình
    def model_fit(self, iteration):
        file_path = self.temp_dir + "/Iteration_" + str(iteration) + ".csv"
        file_path_model = self.temp_dir + "/model_" + str(iteration) + ".keras"
        csv_logger = CSVLogger(file_path, append=True)

        if iteration > 1:
            print(f"{iteration} {self.client_name} Model update params!")
            index = 0
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    layer.set_weights([self.local_weights[iteration-1][index], self.local_biases[iteration-1][index]])
                    index += 1

        # Huấn luyện
        self.model.fit(self.data_train, epochs=4, validation_data=self.data_val, validation_steps=self.validation_steps,
                       steps_per_epoch=self.steps_per_epoch, verbose=1, callbacks=[csv_logger])

        # Đóng băng BatchNormalization sau huấn luyện
        for layer in self.model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

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
            del self.local_weights[iteration-1]  # Xóa iteration cũ (giữ 2 iteration gần nhất)
            del self.local_biases[iteration-1]
        # Flatten weights
        if iteration == 1:
            self.save_shape(weights, biases, iteration)
        flattened_weights, flattened_biases = self.flatten_weights(weights, biases)

        # Thêm nhiễu DP
        lock.acquire()
        flattened_weights, flattened_biases = self.add_gamma_noise(flattened_weights, flattened_biases, iteration)
        lock.release()

        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time
        if iteration > 1:
            del self.compute_times[iteration-1]
            
        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']
        body = {
            'weights': flattened_weights,  # Gửi trực tiếp, không mã hóa
            'biases': flattened_biases,
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

        # Nhận trọng số trực tiếp (không giải mã)
        return_weights = body['weights']
        return_biases = body['biases']

        # Kiểm tra giá trị trọng số
        print(f"[{self.client_name}] Received weights stats: min={np.min(return_weights)}, max={np.max(return_weights)}, mean={np.mean(return_weights)}")
        print(f"[{self.client_name}] Received biases stats: min={np.min(return_biases)}, max={np.max(return_biases)}, mean={np.mean(return_biases)}")

        # Loại bỏ nhiễu
        return_weights -= self.local_weights_noise[iteration]
        return_biases -= self.local_biases_noise[iteration]

        self.global_weights[iteration], self.global_biases[iteration] = self.de_flatten_weights(return_weights, return_biases)
        self.save_global_model(iteration)
        if iteration > 1:
            del self.global_weights[iteration-1]  # Xóa iteration cũ
            del self.global_biases[iteration-1]
            del self.local_weights_noise[iteration-1]  # Xóa nhiễu cũ
            del self.local_biases_noise[iteration-1]
            
        # Đánh giá mô hình
        self.local_accuracy[iteration], self.local_loss[iteration] = self.evaluate_accuracy(self.local_weights[iteration], self.local_biases[iteration])
        self.global_accuracy[iteration], self.global_loss[iteration] = self.evaluate_accuracy(self.global_weights[iteration], self.global_biases[iteration])
        if iteration > 2:
            del self.local_accuracy[iteration-2]  # Giữ 2 iteration gần nhất
            del self.local_loss[iteration-2]
            del self.global_accuracy[iteration-2]
            del self.global_loss[iteration-2]
        # Lưu lịch sử
        history = {"global_acc": [], "global_loss": []}
        history['global_acc'].append(self.global_accuracy[iteration])
        history['global_loss'].append(self.global_loss[iteration])
        file_his = self.temp_dir + "/global_val.csv"
        if iteration == 1:
            pd.DataFrame(history).to_csv(file_his, index=False, header=True, mode='a')
        else:
            pd.DataFrame(history).to_csv(file_his, index=False, header=False, mode='a')

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
        tolerance_left_edge = 0.05
        tolerance_right_edge = 2.0

        if iteration > 1:
            if self.global_loss[iteration] > self.global_loss[iteration-1]:
                self.unconvergence += 1
            else:
                self.unconvergence -= 1
                if self.unconvergence < 0:
                    self.unconvergence = 0
            if self.global_accuracy[iteration] <= self.global_accuracy[iteration-1]:
                self.unconvergence += 1
            else:
                self.unconvergence -= 1
                if self.unconvergence < 0:
                    self.unconvergence = 0

        if np.std(self.global_loss[iteration]) < 0.05:
            self.convergence += 1

        flattened_global_weights, flattened_global_bias = self.flatten_weights(self.global_weights[iteration], self.global_biases[iteration])
        flattened_local_weights, flattened_local_bias = self.flatten_weights(self.local_weights[iteration], self.local_biases[iteration])

        weights_differences = np.abs(flattened_global_weights - flattened_local_weights)
        biases_differences = np.abs(flattened_global_bias - flattened_local_bias)

        if (weights_differences < tolerance_left_edge).all() and (biases_differences < tolerance_left_edge).all():
            self.convergence += 1
        elif (weights_differences > tolerance_right_edge).all() and (biases_differences > tolerance_right_edge).all():
            self.convergence += 1
        else:
            self.convergence -= 1
            if self.convergence < 0:
                self.convergence = 0

        if self.convergence > 3 and self.unconvergence < 3:
            return True
        elif self.unconvergence > 3:
            return True
        return False

    # Xóa client
    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration = body['removing_clients'], body['simulated_time'], body['iteration']
        print(f'[{self.client_name}] Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        return None

    # Lưu shape của trọng số và bias
    def save_shape(self, weights, biases, iteration):
        if iteration <2:
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    self.local_weights_shape.append(layer.get_weights()[0].shape)
                
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    self.local_biases_shape.append(layer.get_weights()[1].shape)
        return None

    # Flatten trọng số
    def flatten_weights(self, weights, biases):
        arr_1 = [weight.flatten() for weight in weights]
        arr_2 = [bias.flatten() for bias in biases]
        return np.concatenate(arr_1).ravel(), np.concatenate(arr_2).ravel()

    # De-flatten trọng số
    def de_flatten_weights(self, flattened_weights, flattened_biases):
        weights = []
        right_pointer = 0
        for shape in self.local_weights_shape:
            delta = 1
            for i in shape:
                delta *= i
            weights.append(np.array(flattened_weights[right_pointer:right_pointer + delta].reshape(shape)))
            right_pointer += delta

        biases = []
        right_pointer = 0
        for shape in self.local_biases_shape:
            delta = shape[0]
            biases.append(np.array(flattened_biases[right_pointer:right_pointer + delta].reshape(shape)))
            right_pointer += delta
        return weights, biases