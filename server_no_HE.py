import sys
sys.path.append('..')

import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.pool import ThreadPool

# Giả lập module dp_mechanisms (thay bằng module thực tế của bạn)
from dp_mechanisms import laplace

num_iterations = 4
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Server:
    def __init__(self, server_name, active_clients_list):
        self.server_name = server_name
        self.global_weights = {}
        self.global_biases = {}
        self.active_clients_list = active_clients_list
        self.agents_dict = {}
        self.client_data_sizes = {}
        
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name] = {}
        if self.server_name not in LATENCY_DICT.keys():
            LATENCY_DICT[self.server_name] = {}
        LATENCY_DICT['server_0'] = {client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds=np.random.random())

    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
        for client_name in self.active_clients_list:
            client = self.agents_dict['client'][client_name]
            self.client_data_sizes[client_name] = client.steps_per_epoch * 512  # Giả sử batch_size = 512

    def get_av(self):
        return self.active_clients_list

    def get_agentsDict(self):
        return self.agents_dict

    def initIterations(self):
        return None

    # Tính trung bình trọng số và bias
    def average_params(self, messages):
        if not messages:
            return None, None
        
        # Lấy trọng số và bias từ message đầu tiên làm mẫu
        sample_weights = messages[0].body['weights']
        sample_biases = messages[0].body['biases']
        
        # Khởi tạo mảng tổng cho mỗi lớp
        weights_sum = [np.zeros_like(w) for w in sample_weights]
        biases_sum = [np.zeros_like(b) for b in sample_biases]
        
        # Tính tổng dữ liệu
        total_data = sum(self.client_data_sizes[m.sender] for m in messages)

        # Tính tổng trọng số và bias từ tất cả client, có trọng số
        for message in messages:
            weights = message.body['weights']
            biases = message.body['biases']
            client_weight = self.client_data_sizes[message.sender] / total_data
            for i in range(len(weights_sum)):
                weights_sum[i] += weights[i] * client_weight
                biases_sum[i] += biases[i] * client_weight
        return weights_sum, biases_sum

    def InitLoop(self):
        converged_clients = {}  
        active_clients_list = self.active_clients_list

        for iteration in range(1, num_iterations + 1):
            print("====================================== Đang chạy Iteration " + str(iteration) + " ======================================")
            weights = {}
            biases = {}

            m = multiprocessing.Manager()
            lock = m.Lock()

            # Gọi client tạo trọng số
            with ThreadPool(len(active_clients_list)) as calling_init_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {'iteration': iteration, 'lock': lock, 'simulated_time': LATENCY_DICT[self.server_name][client_name]}
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                calling_returned_messages = calling_init_pool.map(client_compute_caller, arguments)

            start_call_time = datetime.now()
            simulated_time = find_slowest_time(calling_returned_messages)

            # Tính trung bình trọng số
            self.global_weights[iteration], self.global_biases[iteration] = self.average_params(calling_returned_messages)

            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time

            # Trả trọng số trung bình về client
            with ThreadPool(len(active_clients_list)) as returning_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {
                        'iteration': iteration,
                        'weights': self.global_weights[iteration],
                        'biases': self.global_biases[iteration],
                        'simulated_time': simulated_time
                    }
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                returned_messages = returning_pool.map(client_weights_returner, arguments)

            simulated_time += find_slowest_time(returned_messages)
            start_return_time = datetime.now()

            removing_clients = set()
            for message in returned_messages:
                if message.body['converged'] and message.sender not in converged_clients:
                    converged_clients[message.sender] = iteration
                    removing_clients.add(message.sender)

            end_return_time = datetime.now()
            server_logic_time = end_return_time - start_return_time
            simulated_time += server_logic_time

            active_clients_list = [active_client for active_client in active_clients_list if active_client not in removing_clients]
            if len(active_clients_list) < 2:
                self.get_convergences(converged_clients)
                return

            # Xóa client nếu cần
            with ThreadPool(len(active_clients_list)) as calling_removing_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {'iteration': iteration, 'removing_clients': removing_clients,
                            'simulated_time': simulated_time + LATENCY_DICT[self.server_name][client_name]}
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                __ = calling_removing_pool.map(client_drop_caller, arguments)

            if iteration > 1:
                del self.global_weights[iteration-1] 
                del self.global_biases[iteration-1]
            print("====================================== Kết thúc Iteration " + str(iteration) + " ======================================")

        print(converged_clients)
        return None

    def get_convergences(self, converged_clients):
        for client_name in self.active_clients_list:
            if client_name in converged_clients:
                print(f'Client {client_name} converged on iteration {converged_clients[client_name]}')
            else:
                print(f'Client {client_name} never converged')
        return None

    def final_statistics(self):
        client_accs = []
        fed_acc = []
        for client_name, clientObject in self.agents_dict['client'].items():
            fed_acc.append(list(clientObject.global_accuracy.values()))
            client_accs.append(list(clientObject.local_accuracy.values()))

        print('Client\'s Accuracies are {}'.format(dict(zip(self.agents_dict['client'], fed_acc))))
        return None

# Hàm phụ trợ
def client_compute_caller(input_tuple):
    clientObject, message = input_tuple
    return clientObject.proc_weights(message=message)

def client_weights_returner(input_tuple):
    clientObject, message = input_tuple
    return clientObject.recv_weights(message)

def client_drop_caller(input_tuple):
    clientObject, message = input_tuple
    return clientObject.remove_active_clients(message)

def find_slowest_time(messages):
    simulated_communication_times = {message.sender: message.body['simulated_time'] for message in messages}
    slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
    return simulated_communication_times[slowest_client]