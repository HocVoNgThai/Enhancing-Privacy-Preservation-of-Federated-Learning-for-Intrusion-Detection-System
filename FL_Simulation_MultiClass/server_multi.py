# server_multi.py
import sys
sys.path.append('..')
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.pool import ThreadPool
import gc
import tenseal as ts # <<< THÊM VÀO


from dp_mechanisms import laplace

num_iterations = 8
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return f"Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"


class Server:
    def __init__(self, server_name, active_clients_list):
        self.server_name = server_name
        self.global_weights = {} # Sẽ lưu trữ plaintext sau khi debugging (nếu cần)
        self.global_biases = {}
        # <<< THÊM VÀO: Lưu trữ encrypted global models >>>
        self.global_weights_encrypted = {}
        self.global_biases_encrypted = {}
        self.he_context = None # Server sẽ nhận context từ client
        # <<< KẾT THÚC THÊM VÀO >>>
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
            self.client_data_sizes[client_name] = client.steps_per_epoch * client.batch_size

    def get_av(self):
        return self.active_clients_list

    def get_agentsDict(self):
        return self.agents_dict

    def initIterations(self):
        return None

    # <<< CHỈNH SỬA HOÀN TOÀN: average_params để hoạt động trên dữ liệu mã hóa >>>
    def average_params(self, messages):
        if not messages:
            return None, None

        # Server không có context ban đầu, nó cần lấy từ một client
        # Vì tất cả client dùng chung context, lấy từ client đầu tiên là đủ
        if self.he_context is None:
            # client đã gửi context đã được serialize và không có secret key
            # Đây là context public mà server cần để tính toán
            a_client_msg_body = messages[0].body
            # Không thể deserialize context từ client, phải tạo lại từ init
            # Sửa lại: Server không cần context, nó sẽ load các vector mã hóa
            # với context được tạo sẵn ở init. Ta cần load context này vào server.
            # Cách tốt nhất là truyền nó vào server khi khởi tạo.
            # Tạm thời, ta sẽ load CKKSVector với context được tạo từ đầu.
            # Để đơn giản, ta giả định server có thể truy cập context này.
            # Trong một hệ thống thực, context public sẽ được phân phối an toàn.
            # Sửa lại logic init và server __init__ để server nhận context
            pass # Sẽ load trực tiếp bên dưới

        total_data = sum(self.client_data_sizes[m.sender] for m in messages)

        # Load encrypted vectors từ message đầu tiên và tính trọng số
        first_msg = messages[0]
        client_weight = self.client_data_sizes[first_msg.sender] / total_data
        
        # Load vector với context đã biết
        # Giả định: self.he_context đã được thiết lập.
        # Ta cần sửa Init và Server init để server có context.
        # Tạm thời: Lấy context từ client message đầu tiên để load các vector khác
        
        temp_ctx = ts.context_from(self.agents_dict['client']['client_0'].he_context.serialize(save_secret_key=False))

        agg_weights = ts.CKKSVector.load(temp_ctx, first_msg.body['encrypted_weights']) * client_weight
        agg_biases = ts.CKKSVector.load(temp_ctx, first_msg.body['encrypted_biases']) * client_weight

        # Cộng dồn các vector đã mã hóa từ các client còn lại
        for msg in messages[1:]:
            client_weight = self.client_data_sizes[msg.sender] / total_data
            enc_w = ts.CKKSVector.load(temp_ctx, msg.body['encrypted_weights'])
            enc_b = ts.CKKSVector.load(temp_ctx, msg.body['encrypted_biases'])
            
            agg_weights += (enc_w * client_weight)
            agg_biases += (enc_b * client_weight)

        return agg_weights, agg_biases
    # <<< KẾT THÚC CHỈNH SỬA >>>

    def InitLoop(self):
        converged_clients = {}  
        active_clients_list = self.active_clients_list

        for iteration in range(1, num_iterations + 1):
            print(f"====================================== Đang chạy Iteration {iteration} ======================================")
            weights = {}
            biases = {}

            m = multiprocessing.Manager()
            lock = m.Lock()

            # Chạy đồng thời với ThreadPool
            with ThreadPool(processes=min(len(active_clients_list), multiprocessing.cpu_count())) as calling_init_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {'iteration': iteration, 'lock': lock, 'simulated_time': LATENCY_DICT[self.server_name][client_name]}
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                calling_returned_messages = calling_init_pool.map(client_compute_caller, arguments)

            start_call_time = datetime.now()
            simulated_time = find_slowest_time(calling_returned_messages)

            # <<< CHỈNH SỬA: Lưu trữ kết quả mã hóa >>>
            self.global_weights_encrypted[iteration], self.global_biases_encrypted[iteration] = self.average_params(calling_returned_messages)

            if self.global_weights_encrypted[iteration] is None:
                print("Không nhận được tham số từ client, dừng lại.")
                break
            # <<< KẾT THÚC CHỈNH SỬA >>>
            
            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time

            with ThreadPool(processes=min(len(active_clients_list), multiprocessing.cpu_count())) as returning_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    # <<< CHỈNH SỬA: Gửi đi global model đã mã hóa >>>
                    body = {
                        'iteration': iteration,
                        'encrypted_global_weights': self.global_weights_encrypted[iteration].serialize(),
                        'encrypted_global_biases': self.global_biases_encrypted[iteration].serialize(),
                        'simulated_time': simulated_time
                    }
                    # <<< KẾT THÚC CHỈNH SỬA >>>
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

            with ThreadPool(processes=min(len(active_clients_list), multiprocessing.cpu_count())) as calling_removing_pool:
                arguments = []
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    body = {'iteration': iteration, 'removing_clients': removing_clients,
                            'simulated_time': simulated_time + LATENCY_DICT[self.server_name][client_name]}
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    arguments.append((clientObject, msg))
                __ = calling_removing_pool.map(client_drop_caller, arguments)
            
            if iteration > 1:
                # Xóa các tham số mã hóa của vòng lặp trước để tiết kiệm RAM
                if iteration-1 in self.global_weights_encrypted:
                    del self.global_weights_encrypted[iteration-1]
                    del self.global_biases_encrypted[iteration-1]
            print(f"====================================== Kết thúc Iteration {iteration} ======================================")
            gc.collect()

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
        gc.collect()  # Giải phóng bộ nhớ sau khi hoàn tất
        return None

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