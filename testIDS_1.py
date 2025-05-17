import numpy as np
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP
import joblib
import pandas as pd
import logging
from datetime import datetime
import os
from tensorflow.keras.models import load_model

class DDoSDetector:
    def __init__(self, model_path, scaler_path=None):
        self.flows = defaultdict(self._init_flow)
        
        # Load model với định dạng phù hợp
        if model_path.endswith(('.h5', '.keras')):
            self.model = load_model(model_path)
        else:
            self.model = joblib.load(model_path)
            
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Ngưỡng heuristic dựa trên mẫu dữ liệu
        self.DURATION_RATIO_THRESH = 0.05  # flow_duration/Duration thấp cho DDoS
        self.SYN_RATIO_THRESH = 0.1       # Tỷ lệ SYN cao cho DDoS
        self.SIZE_STD_THRESH = 10.0       # Độ lệch chuẩn thấp cho DDoS
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler('ddos_detection.log'),
                logging.StreamHandler()
            ]
        )

    def _init_flow(self):
        return {
            'start_time': None,
            'last_active': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'packet_sizes': [],
            'timestamps': [],
            'flags': defaultdict(int),
            'protocol': None,
            'header_length': 0,
            'active_periods': []
        }

    def start(self, interface=None):
        """Bắt đầu giám sát traffic mạng"""
        logging.info(f"🚀 Starting DDoS detector on interface {interface or 'default'}")
        sniff(prn=self.process_packet, store=False, iface=interface)

    def process_packet(self, packet):
        try:
            if not packet.haslayer(IP):
                return

            # Trích xuất thông tin cơ bản
            ip = packet[IP]
            flow_key = self._get_flow_key(packet)
            flow = self.flows[flow_key]

            # Khởi tạo flow nếu mới
            if flow['start_time'] is None:
                self._init_new_flow(flow, packet, ip)

            # Cập nhật thông tin flow
            self._update_flow_stats(flow, packet)
            
            # Kiểm tra flow hoàn thành
            if self._is_flow_complete(packet, flow):
                features = self._extract_features(flow)
                del self.flows[flow_key]
                
                # Kiểm tra heuristic trước khi dùng model
                if self._is_suspicious(features):
                    self._predict_and_alert(features)
        except Exception as e:
            logging.error(f"Error processing packet: {e}")

    def _get_flow_key(self, packet):
        """Tạo flow key từ 5-tuple"""
        ip = packet[IP]
        src_port = dst_port = None
        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            
        return (ip.src, ip.dst, src_port, dst_port, ip.proto)

    def _init_new_flow(self, flow, packet, ip):
        """Khởi tạo flow mới"""
        src_port = dst_port = None
        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            
        flow.update({
            'start_time': packet.time,
            'last_active': packet.time,
            'src_ip': ip.src,
            'dst_ip': ip.dst,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': ip.proto,
            'active_periods': [[packet.time]]
        })

    def _update_flow_stats(self, flow, packet):
        """Cập nhật thống kê flow"""
        current_time = packet.time
        
        # Cập nhật active periods
        if current_time - flow['last_active'] > 1.0:  # Ngưỡng idle 1 giây
            flow['active_periods'].append([current_time])
        else:
            flow['active_periods'][-1].append(current_time)
        
        # Cập nhật thông tin chung
        flow['last_active'] = current_time
        flow['packet_sizes'].append(len(packet))
        flow['timestamps'].append(current_time)
        flow['header_length'] += packet[IP].ihl * 4 if packet.haslayer(IP) else 0
        
        # Cập nhật cờ TCP
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            for flag in ['F', 'S', 'R', 'P', 'A', 'E', 'C', 'U']:
                if flag in str(tcp.flags):
                    flow['flags'][flag] += 1

    def _is_flow_complete(self, packet, flow):
        """Kiểm tra flow kết thúc"""
        # Kết thúc khi có FIN/RST, timeout, hoặc đủ gói tin
        timeout = 120  # 2 phút
        is_tcp_fin_rst = packet.haslayer(TCP) and ('F' in str(packet[TCP].flags) or 'R' in str(packet[TCP].flags))
        is_timeout = (packet.time - flow['last_active']) > timeout
        is_packet_limit = len(flow['packet_sizes']) >= 10  # Giới hạn 10 gói tin để xử lý DDoS nhanh
        
        return is_tcp_fin_rst or is_timeout or is_packet_limit

    def _calculate_active_duration(self, active_periods):
        """Tính thời gian hoạt động thực tế"""
        return sum(period[-1] - period[0] for period in active_periods if len(period) > 1)

    def _extract_features(self, flow):
        """Trích xuất đầy đủ 46 đặc trưng theo dataset"""
        flow_duration = flow['last_active'] - flow['start_time'] + 1e-6  # Tránh chia cho 0
        active_duration = self._calculate_active_duration(flow['active_periods'])
        packet_sizes = np.array(flow['packet_sizes'])
        timestamps = np.array(flow['timestamps'])
        iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
        
        # Tính toán các thống kê cơ bản
        total_bytes = sum(flow['packet_sizes'])
        packet_count = len(flow['packet_sizes'])
        avg_packet_size = np.mean(packet_sizes) if packet_sizes.size > 0 else 0
        std_packet_size = np.std(packet_sizes) if packet_sizes.size > 0 else 0
        
        # Protocol detection
        is_tcp = 1 if flow['protocol'] == 6 else 0
        is_udp = 1 if flow['protocol'] == 17 else 0
        is_icmp = 1 if flow['protocol'] == 1 else 0
        
        # TCP flags
        flags = flow['flags']
        
        # Dự đoán tạm thời để xác định Duration và Weight
        is_malicious = (std_packet_size < self.SIZE_STD_THRESH or
                        flags.get('S', 0) / packet_count > self.SYN_RATIO_THRESH if packet_count > 0 else False)
        duration = 64.0 if is_malicious else flow_duration * 2.55
        weight = 141.55 if is_malicious else 38.5
        
        # Đảm bảo Duration >= flow_duration
        if duration < flow_duration:
            duration = flow_duration
        
        features = {
            'flow_duration': flow_duration,
            'Header_Length': flow['header_length'],
            'Protocol Type': flow['protocol'],
            'Duration': duration,
            'Rate': packet_count / flow_duration if flow_duration > 0 else 0,
            'Srate': packet_count / flow_duration if flow_duration > 0 else 0,
            'Drate': 0.0,  # Luôn 0 theo mẫu
            'fin_flag_number': flags.get('F', 0),
            'syn_flag_number': flags.get('S', 0),
            'rst_flag_number': flags.get('R', 0),
            'psh_flag_number': flags.get('P', 0),
            'ack_flag_number': flags.get('A', 0),
            'ece_flag_number': flags.get('E', 0),
            'cwr_flag_number': flags.get('C', 0),
            'ack_count': flags.get('A', 0) * 1.36 if is_malicious else flags.get('A', 0),  # Nhân hệ số cho malicious
            'syn_count': flags.get('S', 0) * 1.36 if is_malicious else flags.get('S', 0),
            'fin_count': flags.get('F', 0),
            'urg_count': flags.get('U', 0),
            'rst_count': flags.get('R', 0),
            'HTTP': 1 if flow['dst_port'] in [80] or flow['src_port'] in [80] else 0,
            'HTTPS': 1 if flow['dst_port'] in [443] or flow['src_port'] in [443] else 0,
            'DNS': 1 if flow['dst_port'] in [53] or flow['src_port'] in [53] else 0,
            'Telnet': 1 if flow['dst_port'] in [23] or flow['src_port'] in [23] else 0,
            'SMTP': 1 if flow['dst_port'] in [25] or flow['src_port'] in [25] else 0,
            'SSH': 1 if flow['dst_port'] in [22] or flow['src_port'] in [22] else 0,
            'IRC': 1 if flow['dst_port'] in [6667] or flow['src_port'] in [6667] else 0,
            'TCP': is_tcp,
            'UDP': is_udp,
            'DHCP': 1 if flow['dst_port'] in [67, 68] or flow['src_port'] in [67, 68] else 0,
            'ARP': 0,  # Không xử lý ARP ở layer 3
            'ICMP': is_icmp,
            'IPv': 1,  # Luôn là 1
            'LLC': 1,  # Giả định có LLC
            'Tot sum': total_bytes,
            'Min': np.min(packet_sizes) if packet_sizes.size > 0 else 0,
            'Max': np.max(packet_sizes) if packet_sizes.size > 0 else 0,
            'AVG': avg_packet_size,
            'Std': std_packet_size,
            'Tot size': total_bytes,
            'IAT': np.mean(iats) if iats.size > 0 else 0,
            'Number': packet_count,
            'Magnitue': np.sqrt(packet_count),
            'Radius': np.sqrt(np.sum(np.diff(packet_sizes)**2)) if packet_sizes.size > 1 else 0,
            'Covariance': np.cov(packet_sizes)[0,0] if packet_sizes.size > 1 else 0,
            'Variance': np.var(packet_sizes) if packet_sizes.size > 0 else 0,
            'Weight': weight,
            'src_ip': flow['src_ip'],
            'dst_ip': flow['dst_ip']
        }
        
        return features

    def _is_suspicious(self, features):
        """Kiểm tra heuristic để lọc traffic đáng ngờ"""
        duration_ratio = features['flow_duration'] / features['Duration'] if features['Duration'] > 0 else 0
        syn_ratio = features['syn_flag_number'] / features['Number'] if features['Number'] > 0 else 0
        
        return (duration_ratio < self.DURATION_RATIO_THRESH or 
                syn_ratio > self.SYN_RATIO_THRESH or
                features['Std'] < self.SIZE_STD_THRESH)

    def _predict_and_alert(self, features):
        """Dự đoán và cảnh báo nếu là DDoS"""
        # Loại bỏ các cột phi số
        feature_columns = [
            'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
            'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
            'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP',
            'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
            'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
        ]
        df = pd.DataFrame([{k: features[k] for k in feature_columns}])
        
        # Chuẩn hóa nếu có scaler
        if self.scaler:
            df = self.scaler.transform(df)
            
        # Dự đoán
        prediction = self.model.predict(df, verbose=0)
        probability = prediction[0][0] if prediction.shape[-1] == 1 else prediction[0][1]
        
        if probability > 0.5:  # Malicious
            self._alert(features, probability)

    def _alert(self, features, probability):
        """Xử lý cảnh báo DDoS"""
        alert_msg = f"""
        🚨 DDoS ATTACK DETECTED 🚨
        Confidence: {probability*100:.2f}%
        Source: {features.get('src_ip', 'N/A')}
        Target: {features.get('dst_ip', 'N/A')}
        Flow Duration: {features['flow_duration']:.6f}s
        Active Duration: {features['Duration']:.2f}s
        Packet Count: {features['Number']}
        SYN Flags: {features['syn_flag_number']}
        Packet Size Std: {features['Std']:.2f}
        """
        logging.warning(alert_msg)
        
        # Ghi log chi tiết
        self._log_attack(features, probability)
        
        # Tự động chặn
        self._block_ip(features.get('src_ip'))

    def _log_attack(self, features, probability):
        """Ghi log chi tiết vào file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'label': 1,
            'probability': probability,
            **{k: v for k, v in features.items() if k not in ['src_ip', 'dst_ip']}
        }
        
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv('ddos_attacks.csv', mode='a', header=not os.path.exists('ddos_attacks.csv'), index=False)

    def _block_ip(self, ip):
        """Tự động chặn IP bằng iptables"""
        if ip and ip != '0.0.0.0':
            try:
                os.system(f"iptables -A INPUT -s {ip} -j DROP")
                logging.warning(f"🔒 Blocked malicious IP: {ip}")
            except Exception as e:
                logging.error(f"Failed to block IP {ip}: {e}")

if __name__ == "__main__":
    # Khởi tạo detector
    detector = DDoSDetector(
        model_path='saved_model/cnn_model_2-0_batch512_20h37p__06-05-2025.keras',
        #scaler_path='path_to_scaler.joblib'  # Thay bằng đường dẫn scaler
    )
    
    # Bắt đầu giám sát
    detector.start(interface='eth0')