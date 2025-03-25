import copy
import numpy as np
import sys, os
import random
import threading
from warnings import simplefilter
from datetime import datetime, timedelta
from sklearn import metrics
from keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from tensorflow.keras.models import load_model
#####
from dp_mechanisms import laplace

##### CODE SECTION
LATENCY_DICT = {}

tolerance_left_edge = 0.5
tolerance_right_edge=2.0
class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Client():
    def __init__(self, client_name, data_train, data_test, \
        active_clients_list):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.agent_dict = {}
        self.temp_dir = client_name + "_log/" + datetime.now().strftime("%Hh%Mp__%d-%m-%Y")
        os.mkdir(self.temp_dir)
        
        ## global
        self.global_weights = {}
        self.global_biases = {}
        self.global_accuracy = {}
        
        ## local
        self.local_weights = {}
        self.local_biases = {}
        self.local_accuracy = {}
        self.compute_times = {} # proc weight
        
        # dp parameter
        self.alpha = 1.0
        self.epsilon = 1.0
        self.mean = 0
        self.len_per_iteration = 50
        self.local_weights_noise ={}
        self.local_biases_noise = {}
        
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name]={}
        if 'server_0' not in LATENCY_DICT.keys():
            LATENCY_DICT['server_0']={}
                    
        LATENCY_DICT['server_0']={client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds= np.random.random())
            
    def get_clientID(self):
        return self.clientID
    
    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
    
    def add_gamma_noise(self, local_weights, local_biases, iteration):
        weights_shape = local_weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        biases_shape = local_biases.shape
        biases_dp_noise = np.zeros(biases_shape)
        
        len_per_iteration = 50 # data_train / iteration
        
        sensitivity =  2 / (len(self.active_clients_list)
                          *self.len_per_iteration*self.alpha)
        
        for i in range(weights_shape[0]):  # weights_modified is 2-D
            for j in range(weights_shape[1]):
                dp_noise = laplace(mean=self.mean, 
                                sensitivity=sensitivity,
                                epsilon=self.epsilon)
                weights_dp_noise [i][j] = dp_noise
                
        
        for i in range(biases_shape[0]):
            dp_noise = laplace(mean=self.mean,
                               sensitivity=sensitivity,
                               epsilon=self.epsilon)
            biases_dp_noise [i] = dp_noise
        
        weights_with_noise = copy.deepcopy(local_weights)  # make a copy to not mutate weights
        biases_with_noise = copy.deepcopy(local_biases)

        self.local_weights_noise[iteration] = weights_dp_noise
        weights_with_noise += weights_dp_noise
        self.local_biases_noise[iteration] = biases_dp_noise
        biases_with_noise += biases_dp_noise
    
        return weights_with_noise, biases_with_noise
    
    def model_fit(self, iteration):
        file_path = self.temp_dir +"/Iteration_"+str(iteration)+".csv"
        file_path_model = self.temp_dir+"/model_"+str(iteration)+".h5"
        
        features, labels = next(iter(self.data_train))
        input_shape = (features.shape[1], 1)
        output_shape = labels.shape[1]
        print(f"Input Shape: {input_shape}")

        from tensorflow import keras

        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=4),
            layers.Conv1D(filters=64, kernel_size=3,  padding="same",activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(output_shape, activation='softmax')
        ])
        csv_logger = CSVLogger(file_path, append=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        if iteration > 1:
            print(f"{iteration} Come here!")
            global_weights = copy.deepcopy(self.global_weights[iteration - 1])
            global_biases = copy.deepcopy(self.global_biases[iteration - 1])
            model.layers[-1].set_weights([global_weights,  global_biases])
        else:
            global_weights = None
            global_biases = None
        
        csv_logger = CSVLogger(file_path, append=True)
        if self.client_name=='client_0':
            steps_per_epoch = 14000
        elif self.client_name =='client_1':
            steps_per_epoch = 28000
        elif self.client_name == 'client_2':
            steps_per_epoch = 42000
        
        model.fit(self.data_train, epochs=5, steps_per_epoch=steps_per_epoch, verbose = 1, callbacks=[csv_logger])
        model.save(file_path_model)
        
        weights, biases = model.layers[-1].get_weights()
        return weights, biases
    
    def proc_weights(self, message):
        
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']

        # if iteration - 1 > len(self.train_datasets):  # iteration is indexed starting from 1
        #     raise (ValueError(
        #         'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
        #             iteration)))
        
        weights, biases = self.model_fit(iteration)

        self.local_weights[iteration] = weights
        self.local_biases[iteration] = biases
        
        final_weights, final_biases = copy.deepcopy(weights), copy.deepcopy(biases)
        
        # add noise lock để đảm bảo không xung đột
        lock.acquire()  # for random seed
        final_weights, final_biases = \
            self.add_gamma_noise(local_weights=weights, local_biases=biases, iteration=iteration)
        lock.release()
        
        #end
        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time
        

        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']

        body = {'weights': final_weights, 'biases': final_biases, 'iter': iteration,
                'compute_time': compute_time, 'simulated_time': simulated_time}  # generate body

        print(self.client_name + "Come end!")
        msg = Message(sender_name=self.client_name, recipient_name=self.agents_dict['server']['server_0'], body=body)
        return msg

    def recv_weights(self, message):
        body = message.body
        iteration, return_weights, return_biases, simulated_time \
        = body['iteration'], body['return_weights'], body['return_biases'], body['simulated_time']
        
        return_weights = copy.deepcopy(return_weights)
        return_biases = copy.deepcopy(return_biases)
        
        ## remove dp
        return_weights -= self.local_weights_noise[iteration] / len(self.active_clients_list)
        return_biases -= self.local_biases_noise[iteration] / len(self.active_clients_list)
        # / len(self.active_clients_list)
        
        self.global_weights[iteration] = return_weights
        self.global_biases[iteration]  = return_biases
        
        local_weights = self.local_weights[iteration]
        local_biases = self.local_biases[iteration]

        # Tính độ hội tụ
        converged = self.check_convergence((local_weights, local_biases), (
            return_weights, return_biases))  # check whether weights have converged
        
        local_accuracy = self.evaluate_accuracy(local_weights, local_biases, iteration)
        global_accuracy = self.evaluate_accuracy(return_weights, return_biases, iteration)
        
        self.local_accuracy[iteration] = local_accuracy
        self.global_accuracy[iteration] = global_accuracy

        args = [self.client_name, iteration, local_accuracy, global_accuracy]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'local accuracy: {} \n' \
                           'global accuracy: {} \n' \
        
        #latency - độ trễ giữa các client
        args.append(self.compute_times[iteration])
        iteration_report += 'local compute time: {} \n'

        args.append(simulated_time)
        iteration_report += 'Simulated time to receive global weights: {} \n \n'
        
        print("Arguments: ",iteration_report.format(*args))

        msg = Message(sender_name=self.client_name, 
                      recipient_name='server_0',
                      body={'converged': converged,
                            'simulated_time': simulated_time + LATENCY_DICT[self.client_name]['server_0']})
        return msg
        
    def evaluate_accuracy(self, local_weights, local_biases, iteration):
        file_path_model = self.temp_dir+"/model_"+str(iteration)+".h5"
        model = load_model(file_path_model)
        model.layers[-1].set_weights([local_weights, local_biases])
        
        loss, accuracy =model.evaluate(self.data_test, steps = 100)
    
        return accuracy
    
    def check_convergence(self, local_params, global_params):
        local_weights, local_biases = local_params
        global_weights, global_biases = global_params

        weights_differences = np.abs(global_weights - local_weights)
        biases_differences = np.abs(global_biases - local_biases)
        
        if (weights_differences < tolerance_left_edge).all() and (biases_differences <tolerance_right_edge).all():
            return True
        elif (weights_differences > tolerance_right_edge).all() and (biases_differences > tolerance_right_edge).all():
           return True     
        
        return False

    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration \
        = body['removing_clients'], body['simulated_time'], body['iteration']
        
        print(f'[{self.client_name}] :Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        return None