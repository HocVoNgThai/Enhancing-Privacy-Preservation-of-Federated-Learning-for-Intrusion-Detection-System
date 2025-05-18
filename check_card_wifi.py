import pywifi
from pywifi import const

def get_wifi_info():
    wifi = pywifi.PyWiFi()
    interfaces = wifi.interfaces()
    
    for i, iface in enumerate(interfaces):
        print(f"[{i}] Interface name: {iface.name()}")
        print(f"    Status: {iface.status()}")  # 0: disconnected, 4: connected

get_wifi_info()
