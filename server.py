import tenseal as ts
import sys
sys.path.append('..')

import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.pool import ThreadPool



def client_compute_caller(input_tuple):
    clientObject, message = input_tuple
    return clientObject.proc_weights(message=message)

# def client_compute_caller(clientObject, message):
#     return clientObject.proc_weights(message=message)

def client_weights_returner(input_tuple):
    clientObject, message = input_tuple
    return clientObject.recv_weights(message)
    # return converged

def client_drop_caller(input_tuple):
    clientObject, message = input_tuple
    return clientObject.remove_active_clients(message)

    
def find_slowest_time(messages):
    simulated_communication_times = {message.sender: message.body['simulated_time'] for message in messages}
    slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
    simulated_time = simulated_communication_times[slowest_client]  # simulated time it would take for server to receive all values
    return simulated_time

num_iterations = 5
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body
        
    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"


class Server():
    def __init__(self, server_name, active_clients_list):
        # ... (giữ nguyên phần __init__ hiện có)
        self.server_name = server_name
        self.global_weights = {}
        self.global_biases = {}
        self.active_clients_list = active_clients_list
        self.agents_dict = {}
        # Lấy context FHE từ client đầu tiên (tất cả client dùng chung context)
        self.context = None
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name]={}
        if self.server_name not in LATENCY_DICT.keys():
            LATENCY_DICT[self.server_name]={}
                    
        LATENCY_DICT['server_0']={client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds= np.random.random())
    
    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
    
    def get_av(self):
        return self.active_clients_list
    
    def get_agentsDict(self):
        return self.agents_dict
    
    def initIterations():
        return None

    def _get_fhe_context_from_client(self, client):
        """Lấy context FHE từ client"""
        # Giả sử tất cả client dùng chung context
        return client.context
    
    def InitLoop(self):
        # Lấy context FHE từ client đầu tiên
        if len(self.active_clients_list) > 0:
            first_client_name = self.active_clients_list[0]
            self.context = self.agents_dict['client'][first_client_name].context
        else:
            raise ValueError("No active clients available")

        converged_clients = {}
        active_clients_list = self.active_clients_list
        
        # Lấy context FHE từ client đầu tiên
        if self.context is None and len(active_clients_list) > 0:
            first_client = self.agents_dict['client'][active_clients_list[0]]
            self.context = self._get_fhe_context_from_client(first_client)
        
        for iteration in range(1, num_iterations+1):
            print(f"============================= Đang chạy Iteration {iteration} =============================")
            weights = {}
            biases = {}
            
            m = multiprocessing.Manager()
            lock = m.Lock()
            
            with ThreadPool(len(active_clients_list)) as calling_init_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {
                        'iteration': iteration, 
                        'lock': lock, 
                        'simulated_time': LATENCY_DICT[self.server_name][client_name]
                    }
                    msg = Message(
                        sender_name=self.server_name, 
                        recipient_name=client_name, 
                        body=body
                    )
                    arguments.append((clientObject, msg))
                calling_returned_messages = calling_init_pool.map(client_compute_caller, arguments)
            
            start_call_time = datetime.now()
            simulated_time = find_slowest_time(calling_returned_messages)
            
            # Giải mã và tính toán trung bình trên dữ liệu mã hóa
            if len(calling_returned_messages) == 0:
                continue
                
            # Lấy shape gốc từ message đầu tiên
            original_shape = calling_returned_messages[0].body['original_shape']
            
            # Khởi tạo tổng mã hóa
            first_msg = calling_returned_messages[0]
            encrypted_weights_sum = ts.lazy_ckks_vector_from(first_msg.body['encrypted_weights'])
            encrypted_weights_sum.link_context(self.context)
            
            encrypted_biases_sum = ts.lazy_ckks_vector_from(first_msg.body['encrypted_biases'])
            encrypted_biases_sum.link_context(self.context)
            
            # Cộng dồn các thông số mã hóa từ các client
            for msg in calling_returned_messages[1:]:
                encrypted_weights = ts.lazy_ckks_vector_from(msg.body['encrypted_weights'])
                encrypted_weights.link_context(self.context)
                encrypted_weights_sum += encrypted_weights
                
                encrypted_biases = ts.lazy_ckks_vector_from(msg.body['encrypted_biases'])
                encrypted_biases.link_context(self.context)
                encrypted_biases_sum += encrypted_biases
            
            # Tính trung bình (chia cho số client)
            count = len(calling_returned_messages)
            encrypted_weights_avg = encrypted_weights_sum * (1/count)
            encrypted_biases_avg = encrypted_biases_sum * (1/count)
            
            # Lưu thông số mã hóa để gửi về client
            self.global_weights[iteration] = encrypted_weights_avg.serialize()
            self.global_biases[iteration] = encrypted_biases_avg.serialize()
            self.original_shapes[iteration] = original_shape
            
            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time
            
            # Gửi thông số mã hóa về các client
            with ThreadPool(len(active_clients_list)) as returning_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {
                        'iteration': iteration,
                        'encrypted_weights': self.global_weights[iteration],
                        'encrypted_biases': self.global_biases[iteration],
                        'original_shape': original_shape,
                        'simulated_time': simulated_time
                    }
                    msg = Message(
                        sender_name=self.server_name, 
                        recipient_name=client_name, 
                        body=body
                    )
                    arguments.append((clientObject, msg))
                returned_messages = returning_pool.map(client_weights_returner, arguments)
            
            # ... (phần còn lại giữ nguyên)
            print("============================= Kết thúc Iteration "+str(iteration)+"=============================")
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