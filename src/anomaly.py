from datetime import datetime

class AnomalyDetector:
    def __init__(self):
        self.history = []
        # Configuration Thresholds
        self.SPIKE_THRESHOLD = 6.0  # kW
        self.VAMPIRE_THRESHOLD = 1.0 # kW (Night time)
        self.SUSTAINED_THRESHOLD = 4.0 # kW
        self.sustained_count = 0

    def check_anamoly(self, timestamp, power_kw):
        """
        Check for anomalies in the current reading.
        Returns: "Alert Message" or None
        """
        hour = timestamp.hour
        
        # 1. Instant Spike Detection
        if power_kw > self.SPIKE_THRESHOLD:
            return f"⚠️ Power Spike Detected: {power_kw:.2f} kW exceeds safety limit!"

        # 2. Vampire Load (Night Monitoring: 12 AM - 5 AM)
        if 0 <= hour < 5:
            if power_kw > self.VAMPIRE_THRESHOLD:
                return f"🧛 High Night Usage: {power_kw:.2f} kW detected during sleep hours."

        # 3. Sustained High Load (Simple counter based)
        # Assuming this is called approx every 1 second
        if power_kw > self.SUSTAINED_THRESHOLD:
            self.sustained_count += 1
            if self.sustained_count > 10: # Alert after ~10 seconds of high load
                self.sustained_count = 0 # Reset to avoid spamming
                return f"📈 Sustained High Load: Device running at {power_kw:.2f} kW for extended period."
        else:
            self.sustained_count = 0
            
        return None
