# client_multi.py
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
import tenseal as ts # Đảm bảo đã import

# TensorFlow và Keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers, callbacks
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
    # <<< CHỈNH SỬA: Thêm he_context vào __init__ >>>
    def __init__(self, client_name, data_train, data_val, data_test, steps_per_epoch, val_steps, test_steps, active_clients_list, he_context):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.agent_dict = {}
        self.temp_dir1 = os.path.join("multiclass_FL_log", datetime.now().strftime("Month%m-Day%d-%Hh-%Mp"))
        os.makedirs(self.temp_dir1, exist_ok=True)
        self.temp_dir = os.path.join(self.temp_dir1, f"{client_name}_log")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.batch_size = 512

        # HE Parameters
        self.he_context = he_context
        # Client giữ khóa bí mật để giải mã
        self.he_secret_key = he_context.secret_key()
        self.param_shapes = {} # Lưu hình dạng của các layer
        self.param_sizes = {} # Lưu kích thước của các layer

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
        self.alpha = 1.0
        self.epsilon = 1.0
        self.mean = 0
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = val_steps
        self.test_steps = test_steps
        self.local_weights_noise = {}
        self.local_biases_noise = {}

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
            layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(5, activation='softmax')
        ])
        adam_optimizer = optimizers.Adam(learning_rate=1e-4)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Giảm patience
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def add_gamma_noise(self, local_weights, local_biases, iteration):
        weights_dp_noise = []
        biases_dp_noise = []
        sensitivity = 2 / (len(self.active_clients_list) * self.steps_per_epoch * self.alpha)

        for weight_layer in local_weights:
            noise_layer = np.zeros_like(weight_layer, dtype=np.float16)  # Sử dụng float16
            for idx, weight in np.ndenumerate(weight_layer):
                if abs(weight) > 1e-15:
                    noise = laplace(mean=self.mean, sensitivity=sensitivity, epsilon=self.epsilon)
                    noise_layer[idx] = noise
            weights_dp_noise.append(noise_layer)

        for bias_layer in local_biases:
            noise_layer = np.zeros_like(bias_layer, dtype=np.float16)
            for idx, bias in np.ndenumerate(bias_layer):
                if abs(bias) > 1e-15:
                    noise = laplace(mean=self.mean, sensitivity=sensitivity, epsilon=self.epsilon)
                    noise_layer[idx] = noise
            biases_dp_noise.append(noise_layer)

        self.local_weights_noise[iteration] = weights_dp_noise
        self.local_biases_noise[iteration] = biases_dp_noise

        weights_with_noise = [w + n for w, n in zip(local_weights, weights_dp_noise)]
        biases_with_noise = [b + n for b, n in zip(local_biases, biases_dp_noise)]
        return weights_with_noise, biases_with_noise

    # <<< THÊM VÀO: Các hàm tiện ích cho HE >>>
    def _flatten_and_encrypt(self, params_list):
        """Làm phẳng, lưu shape và mã hóa danh sách các tham số (weights hoặc biases)."""
        flat_params = []
        shapes = []
        sizes = []
        for params in params_list:
            shapes.append(params.shape)
            size = params.size
            sizes.append(size)
            flat_params.extend(params.flatten().tolist())
        
        # Mã hóa vector đã làm phẳng
        encrypted_vector = ts.ckks_vector(self.he_context, flat_params)
        return encrypted_vector, shapes, sizes

    def _decrypt_and_reconstruct(self, encrypted_vector, shapes, sizes):
        """Giải mã và tái tạo lại cấu trúc tham số."""
        decrypted_flat = encrypted_vector.decrypt(self.he_secret_key)
        
        reconstructed_params = []
        current_pos = 0
        for shape, size in zip(shapes, sizes):
            # Cắt phần vector tương ứng và reshape
            param_slice = decrypted_flat[current_pos : current_pos + size]
            reconstructed_params.append(np.array(param_slice, dtype=np.float32).reshape(shape))
            current_pos += size
        return reconstructed_params
    # <<< KẾT THÚC THÊM VÀO >>>


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

        self.model.fit(self.data_train, epochs= 2, validation_data=self.data_val, validation_steps=self.validation_steps,
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
                w, b = layer.get_weights()
                weights.append(w)
                biases.append(b)
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

        if lock:
            lock.acquire()
        weights_dp, biases_dp = self.add_gamma_noise(weights, biases, iteration)
        if lock:
            lock.release()

        # <<< CHỈNH SỬA: Mã hóa weights và biases >>>
        print(f"[{self.client_name}] Bắt đầu mã hóa tham số...")
        encrypted_weights, w_shapes, w_sizes = self._flatten_and_encrypt(weights_dp)
        encrypted_biases, b_shapes, b_sizes = self._flatten_and_encrypt(biases_dp)
        
        # Lưu lại shape để giải mã sau
        self.param_shapes[iteration] = {'weights': w_shapes, 'biases': b_shapes}
        self.param_sizes[iteration] = {'weights': w_sizes, 'biases': b_sizes}
        print(f"[{self.client_name}] Mã hóa hoàn tất.")
        # <<< KẾT THÚC CHỈNH SỬA >>>

        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time

        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']
        
        # <<< CHỈNH SỬA: Gửi đi dữ liệu đã mã hóa >>>
        body = {
            'encrypted_weights': encrypted_weights.serialize(), # Serialize để gửi đi
            'encrypted_biases': encrypted_biases.serialize(),
            'iter': iteration,
            'compute_time': compute_time,
            'simulated_time': simulated_time
        }
        # <<< KẾT THÚC CHỈNH SỬA >>>

        print(f"{self.client_name} End Produce Weights")
        # Sử dụng agent_dict['server']['server_0'].server_name để đảm bảo đúng tên
        recipient_server_name = list(self.agent_dict['server'].keys())[0]
        msg = Message(sender_name=self.client_name, recipient_name=recipient_server_name, body=body)
        return msg

    def recv_weights(self, message):
        body = message.body
        iteration, simulated_time = body['iteration'], body['simulated_time']

        # <<< CHỈNH SỬA: Nhận và giải mã global model >>>
        print(f"[{self.client_name}] Nhận global model, bắt đầu giải mã...")
        encrypted_global_weights = ts.CKKSVector.load(self.he_context, body['encrypted_global_weights'])
        encrypted_global_biases = ts.CKKSVector.load(self.he_context, body['encrypted_global_biases'])

        # Lấy shape và size đã lưu từ bước proc_weights
        w_shapes = self.param_shapes[iteration]['weights']
        b_shapes = self.param_shapes[iteration]['biases']
        w_sizes = self.param_sizes[iteration]['weights']
        b_sizes = self.param_sizes[iteration]['biases']

        return_weights_encrypted = self._decrypt_and_reconstruct(encrypted_global_weights, w_shapes, w_sizes)
        return_biases_encrypted = self._decrypt_and_reconstruct(encrypted_global_biases, b_shapes, b_sizes)
        print(f"[{self.client_name}] Giải mã hoàn tất.")

        # Loại bỏ nhiễu DP khỏi global model đã được giải mã
        return_weights = [w - n for w, n in zip(return_weights_encrypted, self.local_weights_noise[iteration])]
        return_biases = [b - n for b, n in zip(return_biases_encrypted, self.local_biases_noise[iteration])]
        # <<< KẾT THÚC CHỈNH SỬA >>>

        self.global_weights[iteration], self.global_biases[iteration] = return_weights, return_biases
        self.save_global_model(iteration)

        if iteration > 1:
            del self.global_weights[iteration-1]
            del self.global_biases[iteration-1]
            del self.local_weights_noise[iteration-1]
            del self.local_biases_noise[iteration-1]
            del self.param_shapes[iteration-1] # Xóa shape cũ
            del self.param_sizes[iteration-1] # Xóa size cũ

        # ... (Phần còn lại của hàm `recv_weights` giữ nguyên, vì nó hoạt động trên dữ liệu đã giải mã) ...
        # ... (đánh giá, lưu log, kiểm tra hội tụ) ...
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
        recipient_server_name = list(self.agent_dict['server'].keys())[0]
        msg = Message(sender_name=self.client_name, recipient_name=recipient_server_name,
                      body={'converged': converged, 'simulated_time': simulated_time + LATENCY_DICT[self.client_name][recipient_server_name]})
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
        
        acc_diff = abs(self.global_accuracy[iteration] - self.global_accuracy[iteration-1])
        loss_diff = abs(self.global_loss[iteration] - self.global_loss[iteration-1])
        
        if acc_diff < 0.01 and loss_diff < 0.05 and self.global_accuracy[iteration] > 0.9 and self.global_loss[iteration] < 0.1:
            self.convergence += 1
        else: 
            self.convergence = 0
    
        if self.convergence > 4:
            return True
        return False

    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration = body['removing_clients'], body['simulated_time'], body['iteration']
        print(f'[{self.client_name}] Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        gc.collect()  # Giải phóng bộ nhớ sau khi xóa client
        return None