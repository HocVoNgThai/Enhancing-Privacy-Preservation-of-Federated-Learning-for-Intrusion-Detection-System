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
        self.DURATION_RATIO_THRESH = 0.05
        self.SYN_RATIO_THRESH = 0.1
        self.RATE_THRESH = 1000  # Ngưỡng tần suất gói tin (packets/s) cho ICMP/UDP
        self.PACKET_COUNT_THRESH = 500  # Ngưỡng số lượng gói tin cho ICMP/UDP
        
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
            'flags': defaultdict(int),
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
                if self._is_suspicious(features):
                    self._predict_and_alert(features)
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
        packet_count = len(flow['timestamps'])
        rate = packet_count / flow_duration if flow_duration > 0 else 0
        
        features = {
            'flow_duration': flow_duration,
            'syn_flag_number': flow['flags']['S'],
            'ack_flag_number': flow['flags']['A'],
            'fin_flag_number': flow['flags']['F'],
            'rst_flag_number': flow['flags']['R'],
            'psh_flag_number': flow['flags']['P'],
            'urg_flag_number': flow['flags']['U'],
            'packet_count': packet_count,
            'rate': rate,
            'protocol': flow['protocol'],
            'src_ips': flow['src_ips'],
            'dst_ip': flow['dst_ip']
        }
        return features

    def _is_suspicious(self, features):
        if features['packet_count'] <= 1:
            return False
        
        # Phát hiện SYN flood (TCP)
        if features['protocol'] == 6:  # TCP
            syn_ratio = features['syn_flag_number'] / features['packet_count'] if features['packet_count'] > 0 else 0
            return syn_ratio > self.SYN_RATIO_THRESH
        
        # Phát hiện ICMP/UDP flood
        if features['protocol'] in [1, 17]:  # ICMP hoặc UDP
            return (features['rate'] > self.RATE_THRESH or 
                    features['packet_count'] > self.PACKET_COUNT_THRESH)
        
        return False

    def _predict_and_alert(self, features):
        feature_columns = [
            'flow_duration', 'syn_flag_number', 'ack_flag_number', 'fin_flag_number',
            'rst_flag_number', 'psh_flag_number', 'urg_flag_number', 'packet_count', 'rate'
        ]
        df = pd.DataFrame([{k: features[k] for k in feature_columns}])
        prediction = self.model.predict(df, verbose=0)
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
        # Hiển thị tất cả src_ips hoặc thông báo nếu có nhiều nguồn
        src_ip_display = list(features['src_ips'])[0] if len(features['src_ips']) == 1 else f"Multiple sources ({len(features['src_ips'])} IPs)"
        alert_msg = f"""
        🚨 DDoS ATTACK DETECTED 🚨
        Confidence: {probability*100:.2f}%
        Source: {src_ip_display}
        Target: {features['dst_ip']}
        Flow Duration: {features['flow_duration']:.6f}s
        Packet Count: {features['packet_count']}
        Flags: {flags_str if flags_str else 'None'}
        """
        logging.warning(alert_msg)
        self._log_attack(features, probability)

    def _log_attack(self, features, probability):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'label': 1,
            'probability': probability,
            **{k: v for k, v in features.items() if k not in ['src_ips', 'dst_ip']}
        }
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv('ddos_attacks.csv', mode='a', header=not os.path.exists('ddos_attacks.csv'), index=False)

    def _block_ip(self, ip):
        pass  # Bỏ qua phần chặn IP

if __name__ == "__main__":
    detector = DDoSDetector('saved_model/cnn_model_2-0_batch512_20h37p__06-05-2025.keras')
    detector.start(interface='eth0')