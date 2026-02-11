"""
Virtual Smart Meter Bot 🤖
Generates realistic, context-aware electricity data for simulation.
"""

import random
import datetime
import pandas as pd
import numpy as np

class VirtualSmartMeter:
    def __init__(self):
        self.state = {
            'ac_on': False,
            'heater_on': False,
            'washing_machine_on': False,
            'ev_charger_on': False,
            'lights_on': False
        }
        # Base loads (kW)
        self.loads = {
            'fridge': 0.15,
            'fan': 0.08,
            'tv': 0.12,
            'lights': 0.05,
            'ac': 1.5,
            'heater': 2.0,
            'washing_machine': 0.5,
            'ev_charger': 3.0,
            'microwave': 1.2
        }
    
    def _is_peak_hours(self, hour):
        # Peak: 7-9 AM (Morning routine) and 6-10 PM (Evening routine)
        return (7 <= hour <= 9) or (18 <= hour <= 22)
    
    def _get_base_load(self, hour):
        """Calculate base load depending on time of day"""
        load = self.loads['fridge'] # Always running cycle
        
        # Day vs Night logic
        if 9 <= hour <= 17:
             load += 0.2 # Home office / general day usage
        elif 0 <= hour <= 5:
             load = 0.1 # Sleeping (just fridge cycling)
             
        # Peak randomness
        if self._is_peak_hours(hour):
            load += random.uniform(0.1, 0.4) # Random lights/fans
            
        return load

    def generate_reading(self):
        """
        Generate a single data point representing the current instant.
        Returns: dict with Voltage, Current, Power, PowerFactor, Frequency
        """
        now = datetime.datetime.now()
        hour = now.hour
        
        # 1. Calculate Active Power (kW)
        power_kw = self._get_base_load(hour)
        
        # Add manual state loads
        if self.state['ac_on']: power_kw += self.loads['ac']
        if self.state['heater_on']: power_kw += self.loads['heater']
        if self.state['washing_machine_on']: power_kw += self.loads['washing_machine']
        if self.state['ev_charger_on']: power_kw += self.loads['ev_charger']
        
        # Add random noise (fluctuation)
        noise = random.uniform(-0.02, 0.02)
        power_kw = max(0.05, power_kw + noise)
        
        # 2. Simulate Voltage (V) - Indian Grid Standard (230V ± 10%)
        # Voltage dips slightly when load is high
        voltage_base = 230.0
        voltage_sag = (power_kw / 5.0) * 5.0 # Max 5V sag at 5kW
        voltage = voltage_base - voltage_sag + random.uniform(-2.0, 2.0)
        
        # 3. Simulate Power Factor (0.8 - 0.99)
        pf = random.uniform(0.85, 0.98)
        
        # 4. Calculate Current (I = P / (V * PF))
        # power_kw * 1000 = V * I * PF
        current = (power_kw * 1000) / (voltage * pf)
        
        # 5. Frequency (Indian Grid 50Hz ± 0.5)
        freq = 50.0 + random.uniform(-0.1, 0.1)
        
        return {
            'timestamp': now.isoformat(timespec='seconds'),
            'voltage_v': round(voltage, 2),
            'current_a': round(current, 2),
            'power_kw': round(power_kw, 3),
            'power_factor': round(pf, 2),
            'frequency_hz': round(freq, 2),
            'state': self.state.copy()
        }

    def toggle_device(self, device):
        if device in self.state:
            self.state[device] = not self.state[device]
            return f"{device} is now {'ON' if self.state[device] else 'OFF'}"
        return "Unknown device"
