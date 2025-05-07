import sys
sys.path.append('..')

import numpy as np
import tenseal as ts
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.pool import ThreadPool

###

###
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

num_iterations = 10
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body
        
    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"
    
class Server():
    def __init__(self,server_name, active_clients_list):
        self.server_name = server_name
        self.global_weights = {}
        self.global_biases = {}
        self.global_weights_original_shape = {}
        self.active_clients_list = active_clients_list
        self.agents_dict = {}
        self.global_accuracy = {}
        self.global_loss = {}
        
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
    
####################################   HE CONTEXT  ###########################################
    # def init_he_context(self):
    #         """Thiết lập context mã hóa đồng hình"""
    #         context = ts.context(
    #             ts.SCHEME_TYPE.CKKS, # ckks cho số thực, bfv cho int
    #             poly_modulus_degree=8192,
    #             coeff_mod_bit_sizes=[60, 40, 40, 60]
    #         )
    #         context.generate_galois_keys()
    #         context.global_scale = 2**40
    #         return context

####################################   HE CONTEXT   ##########################################
    
    def average_encrypted_params(self, messages):
        # temp_sum_weights = sum(message.body['weights'] for message in calling_returned_messages)
        # temp_sum_biases = sum(message.body['biases'] for message in calling_returned_messages)
        
        # self.global_weights[iteration] = temp_sum_weights/len(self.active_clients_list)
        # self.global_biases[iteration] = temp_sum_biases/len(self.active_clients_list)
        
        message_0 = messages[0]
        encrypted_weights = ts.lazy_ckks_vector_from(message_0.body['encrypted_weights'])
        encrypted_biases = ts.lazy_ckks_vector_from(message_0.body['encrypted_biases'])
        context = ts.context_from(message_0.body['context'])
        
        encrypted_weights.link_context(context)
        encrypted_biases.link_context(context)
        encrypted_weights_sum = encrypted_weights
        encrypted_biases_sum = encrypted_biases
        
        for message in messages[1:]:
            encrypted_weights = ts.lazy_ckks_vector_from(message.body['encrypted_weights'])
            encrypted_biases = ts.lazy_ckks_vector_from(message.body['encrypted_biases'])
            context = ts.context_from(message.body['context'])
            encrypted_weights.link_context(context)
            encrypted_biases.link_context(context)
            encrypted_weights_sum += encrypted_weights
            encrypted_biases_sum += encrypted_biases
        
        avg_encrypted_weights = encrypted_weights_sum * (1/len(self.active_clients_list))
        avg_encrypted_biases = encrypted_biases_sum * (1/len(self.active_clients_list))
        
        return avg_encrypted_weights.serialize(), avg_encrypted_biases.serialize()

    def InitLoop(self):
        converged_clients = {} # client đã hội tụ (removed)
        active_clients_list = self.active_clients_list
        
        for iteration in range(1, num_iterations+1):
            print("====================================== Đang chạy Iteration "+str(iteration)+" ======================================")
            weights = {}
            biases = {}
            
            m = multiprocessing.Manager()
            lock = m.Lock() # thực hiện tránh xung đột tài nguyên nếu cần

######################################    CALL CLIENTS CREATE WEIGHTS&BIAS    ######################################################                 
            
            with ThreadPool(len(active_clients_list)) as calling_init_pool:
                arguments = []
                
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    
                    body = {'iteration': iteration, 'lock': lock, 'simulated_time': LATENCY_DICT[self.server_name][client_name]}
                    #message from server to client
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body = body)
                    
                    arguments.append((clientObject, msg))
                calling_returned_messages = calling_init_pool.map(client_compute_caller, arguments)
            
            
            start_call_time = datetime.now()
            simulated_time = find_slowest_time(calling_returned_messages)
            
            # truong hop nay cac weights shape đến từ client đều giống nhau
            self.global_weights_original_shape[iteration] = calling_returned_messages[0].body['weights_original_shape']
            self.global_weights[iteration], self.global_biases[iteration] = self.average_encrypted_params(calling_returned_messages)
            
            # add time server logic takes
            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time #Tổng thời gian cho đến bước này
######################################    CALL CLIENTS CREATE WEIGHTS&BIAS    ######################################################                 

            
######################################    RETURN NEW WEIGHTS    ######################################################            
            # Trả weights với bias trung bình mới về client 
            with ThreadPool(len(active_clients_list)) as returning_pool:
                arguments = []
                
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    
                    body = {'weights_original_shape': self.global_weights_original_shape[iteration],
                            'iteration': iteration,
                            'encrypted_weights' : self.global_weights[iteration], 
                            'encrypted_biases': self.global_biases[iteration],
                            'simulated_time': simulated_time}
                    
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    
                    arguments.append((clientObject, msg))
                returned_messages = returning_pool.map(client_weights_returner, arguments)
            
            
            simulated_time += find_slowest_time(returned_messages) # Tổng tg cho đến bước này 
            start_return_time = datetime.now()
            
            removing_clients = set()
            
            for message in returned_messages:
                if message.body['converged'] == True and message.sender not in converged_clients:
                    converged_clients[message.sender] = iteration
                    removing_clients.add(message.sender)
                    
            end_return_time = datetime.now()
            server_logic_time = end_return_time - start_return_time
            simulated_time += server_logic_time #Tổng thời gian cho đến bước này
            
            # bỏ client nếu nó hội tụ
            active_clients_list = [active_client for active_client in active_clients_list if active_client not in removing_clients]
            
            # nếu số client nhỏ hơn 2, dừng được rồi
            if len(active_clients_list) < 2:
                self.get_convergences(converged_clients)
                return
######################################    RETURN NEW WEIGHTS    ######################################################     


######################################    REMOVE CLIENTS IF NEEDED    ######################################################                 
            with ThreadPool(len(active_clients_list)) as calling_removing_pool:
                arguments = []
                
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    
                    body = {'iteration': iteration, 'removing_clients': removing_clients,\
                        'simulated_time': simulated_time + LATENCY_DICT[self.server_name][client_name]}
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                __ = calling_removing_pool.map(client_drop_caller, arguments)
            
            print("====================================== Kết thúc Iteration "+str(iteration)+" ======================================")
        
        print(converged_clients)
        return None    
######################################    REMOVE CLIENTS IF NEEDED    ######################################################                 


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
