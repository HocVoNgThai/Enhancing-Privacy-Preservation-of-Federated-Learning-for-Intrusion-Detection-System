import numpy as np
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import logging
from datetime import datetime
from tensorflow.keras.models import load_model

class DDoSDetector:
    def __init__(self, model_path):
        self.flows = defaultdict(self._init_flow)
        self.model = load_model(model_path)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[logging.FileHandler('ddos_detection.log'), logging.StreamHandler()]
        )

    def _init_flow(self):
        return {
            'start_time': None,
            'last_active': None,
            'src_ips': set(),  # Lưu tập hợp các src_ip
            'dst_ip': None,
            'packet_sizes': [],
            'timestamps': [],
            'flags': defaultdict(int),  # Mặc định giá trị là 0 cho các cờ
            'protocol': None
        }

    def start(self, interface=None):
        logging.info(f"🚀 Starting DDoS detector on interface {interface or 'default'}")
        sniff(prn=self.process_packet, store=False, iface=interface)

    def process_packet(self, packet):
        try:
            if not packet.haslayer(IP):
                return
            ip = packet[IP]
            flow_key = (ip.dst, ip.proto)  # Nhóm theo dst_ip và protocol
            flow = self.flows[flow_key]

            if flow['start_time'] is None:
                self._init_new_flow(flow, packet, ip)
            self._update_flow_stats(flow, packet)
            if self._is_flow_complete(packet, flow):
                features = self._extract_features(flow)
                del self.flows[flow_key]
                self._predict_and_alert(features)  # Gọi trực tiếp, bỏ _is_suspicious
        except Exception as e:
            logging.error(f"Error processing packet: {e}")

    def _init_new_flow(self, flow, packet, ip):
        flow.update({
            'start_time': packet.time,
            'last_active': packet.time,
            'src_ips': {ip.src},  # Lưu src_ip đầu tiên
            'dst_ip': ip.dst,
            'protocol': ip.proto
        })

    def _update_flow_stats(self, flow, packet):
        ip = packet[IP]
        flow['src_ips'].add(ip.src)  # Thêm src_ip vào tập hợp
        flow['last_active'] = packet.time
        flow['timestamps'].append(packet.time)
        flow['packet_sizes'].append(len(packet))
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            flow['flags']['S'] += 1 if tcp.flags & 0x02 else 0  # SYN
            flow['flags']['A'] += 1 if tcp.flags & 0x10 else 0  # ACK
            flow['flags']['F'] += 1 if tcp.flags & 0x01 else 0  # FIN
            flow['flags']['R'] += 1 if tcp.flags & 0x04 else 0  # RST
            flow['flags']['P'] += 1 if tcp.flags & 0x08 else 0  # PSH
            flow['flags']['U'] += 1 if tcp.flags & 0x20 else 0  # URG

    def _is_flow_complete(self, packet, flow):
        timeout = 120
        is_timeout = (packet.time - flow['last_active']) > timeout
        is_packet_limit = len(flow['timestamps']) >= 100
        return is_timeout or is_packet_limit

    def _extract_features(self, flow):
        timestamps = np.array(flow['timestamps'])
        flow_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        packet_sizes = np.array(flow['packet_sizes'])
        packet_count = len(flow['timestamps'])
        rate = packet_count / flow_duration if flow_duration > 0 else 0
        iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
        iat_mean = np.mean(iats) if iats.size > 0 else 0

        is_tcp = 1 if flow['protocol'] == 6 else 0
        is_udp = 1 if flow['protocol'] == 17 else 0
        is_icmp = 1 if flow['protocol'] == 1 else 0
        total_bytes = sum(flow['packet_sizes']) if packet_sizes.size > 0 else 0
        avg_packet_size = np.mean(packet_sizes) if packet_sizes.size > 0 else 0
        std_packet_size = np.std(packet_sizes) if packet_sizes.size > 0 else 0
        min_packet_size = np.min(packet_sizes) if packet_sizes.size > 0 else 0
        max_packet_size = np.max(packet_sizes) if packet_sizes.size > 0 else 0

        features = {
            'flow_duration': flow_duration,
            'Header_Length': 0,  # Placeholder (có thể lấy từ packet[IP].ihl * 4 nếu cần)
            'Protocol Type': flow['protocol'],
            'Duration': flow_duration,  # Sử dụng flow_duration làm placeholder
            'Rate': rate,
            'Srate': rate,  # Sử dụng Rate làm placeholder
            'Drate': 0,  # Placeholder
            'fin_flag_number': flow['flags']['F'],
            'syn_flag_number': flow['flags']['S'],
            'rst_flag_number': flow['flags']['R'],
            'psh_flag_number': flow['flags']['P'],
            'ack_flag_number': flow['flags']['A'],
            'ece_flag_number': 0,  # Placeholder
            'cwr_flag_number': 0,  # Placeholder
            'ack_count': flow['flags']['A'],  # Placeholder (có thể nhân với hệ số nếu cần)
            'syn_count': flow['flags']['S'],  # Placeholder
            'fin_count': flow['flags']['F'],
            'urg_count': flow['flags']['U'],
            'rst_count': flow['flags']['R'],
            'HTTP': 0,  # Placeholder (có thể kiểm tra port 80)
            'HTTPS': 0,  # Placeholder (có thể kiểm tra port 443)
            'DNS': 0,    # Placeholder (có thể kiểm tra port 53)
            'Telnet': 0, # Placeholder (có thể kiểm tra port 23)
            'SMTP': 0,   # Placeholder (có thể kiểm tra port 25)
            'SSH': 0,    # Placeholder (có thể kiểm tra port 22)
            'IRC': 0,    # Placeholder (có thể kiểm tra port 6667)
            'TCP': is_tcp,
            'UDP': is_udp,
            'DHCP': 0,   # Placeholder (có thể kiểm tra port 67, 68)
            'ARP': 0,    # Placeholder
            'ICMP': is_icmp,
            'IPv': 1,    # Placeholder
            'LLC': 1,    # Placeholder
            'Tot sum': total_bytes,
            'Min': min_packet_size,
            'Max': max_packet_size,
            'AVG': avg_packet_size,
            'Std': std_packet_size,
            'Tot size': total_bytes,
            'IAT': iat_mean,
            'Number': packet_count,
            'Magnitue': np.sqrt(packet_count),
            'Radius': np.sqrt(np.sum(np.diff(packet_sizes)**2)) if packet_sizes.size > 1 else 0,
            'Covariance': 0,  # Placeholder (có thể tính lại nếu cần)
            'Variance': np.var(packet_sizes) if packet_sizes.size > 0 else 0,
            'Weight': 38.5,  # Placeholder
            'src_ips': flow['src_ips'],
            'dst_ip': flow['dst_ip']
        }

        # Gán các giá trị cho cờ để sử dụng trong _alert
        features['urg_flag_number'] = flow['flags']['U']  # Đảm bảo key này tồn tại

        # Chuyển features thành DataFrame với 46 cột
        feature_df = pd.DataFrame([{
            'flow_duration': features['flow_duration'],
            'Header_Length': features['Header_Length'],
            'Protocol Type': features['Protocol Type'],
            'Duration': features['Duration'],
            'Rate': features['Rate'],
            'Srate': features['Srate'],
            'Drate': features['Drate'],
            'fin_flag_number': features['fin_flag_number'],
            'syn_flag_number': features['syn_flag_number'],
            'rst_flag_number': features['rst_flag_number'],
            'psh_flag_number': features['psh_flag_number'],
            'ack_flag_number': features['ack_flag_number'],
            'ece_flag_number': features['ece_flag_number'],
            'cwr_flag_number': features['cwr_flag_number'],
            'ack_count': features['ack_count'],
            'syn_count': features['syn_count'],
            'fin_count': features['fin_count'],
            'urg_count': features['urg_count'],
            'rst_count': features['rst_count'],
            'HTTP': features['HTTP'],
            'HTTPS': features['HTTPS'],
            'DNS': features['DNS'],
            'Telnet': features['Telnet'],
            'SMTP': features['SMTP'],
            'SSH': features['SSH'],
            'IRC': features['IRC'],
            'TCP': features['TCP'],
            'UDP': features['UDP'],
            'DHCP': features['DHCP'],
            'ARP': features['ARP'],
            'ICMP': features['ICMP'],
            'IPv': features['IPv'],
            'LLC': features['LLC'],
            'Tot sum': features['Tot sum'],
            'Min': features['Min'],
            'Max': features['Max'],
            'AVG': features['AVG'],
            'Std': features['Std'],
            'Tot size': features['Tot size'],
            'IAT': features['IAT'],
            'Number': features['Number'],
            'Magnitue': features['Magnitue'],
            'Radius': features['Radius'],
            'Covariance': features['Covariance'],
            'Variance': features['Variance'],
            'Weight': features['Weight']
        }])

        features['feature_df'] = feature_df
        return features

    def _predict_and_alert(self, features):
        prediction = self.model.predict(features['feature_df'], verbose=0)
        probability = float(prediction[0][0] if prediction.shape[-1] == 1 else prediction[0][1])
        if probability > 0.5:
            self._alert(features, probability)

    def _alert(self, features, probability):
        flags_str = ', '.join([f"{k}={v}" for k, v in {
            'SYN': features['syn_flag_number'],
            'ACK': features['ack_flag_number'],
            'FIN': features['fin_flag_number'],
            'RST': features['rst_flag_number'],
            'PSH': features['psh_flag_number'],
            'URG': features['urg_flag_number']
        }.items() if v > 0])
        src_ip_display = list(features['src_ips'])[0] if len(features['src_ips']) == 1 else f"Multiple sources ({len(features['src_ips'])} IPs)"
        alert_msg = f"""
        🚨 DDoS ATTACK DETECTED 🚨
        Confidence: {probability*100:.2f}%
        Source: {src_ip_display}
        Target: {features['dst_ip']}
        Flow Duration: {features['flow_duration']:.6f}s
        Packet Count: {features['Number']}
        Flags: {flags_str if flags_str else 'None'}
        """
        logging.warning(alert_msg)
        self._log_attack(features, probability)

    def _log_attack(self, features, probability):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'label': 1,
            'probability': probability,
            **{k: v for k, v in features.items() if k not in ['src_ips', 'dst_ip', 'feature_df']}
        }
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv('ddos_attacks.csv', mode='a', header=not os.path.exists('ddos_attacks.csv'), index=False)

    def _block_ip(self, ip):
        pass  # Bỏ qua phần chặn IP

if __name__ == "__main__":
    detector = DDoSDetector('saved_model/cnn_model_2-0_batch512_20h37p__06-05-2025.keras')
    detector.start(interface='eth0')