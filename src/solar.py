
import numpy as np
from datetime import datetime

class SolarSimulator:
    def __init__(self, system_size_kw=5.0, efficiency=0.18, performance_ratio=0.75):
        """
        :param system_size_kw:  Rated capacity (e.g., 5kW system)
        :param efficiency:      Panel efficiency (std 18-20%)
        :param performance_ratio: Real-world loss factor (heat, wiring, inverter) ~0.75
        """
        self.system_size_kw = system_size_kw
        self.efficiency = efficiency
        # Boost PR slightly for premium efficiency (better wiring/inverter assumption)
        if efficiency > 0.20:
            self.performance_ratio = performance_ratio * 1.05 # 5% boost
        else:
            self.performance_ratio = performance_ratio

    def calculate_instant_power(self, weather_data):
        """
        Calculate estimated power output (kW) right now based on weather.
        """
        if not weather_data:
            return 0.0

        # 1. Time of Day Factor (Sun Position)
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        
        # Simple Solar Elevation Model: Peak at 13:00 (1 PM), zero before 6 and after 19
        if 6 <= hour <= 19:
            # Normalized Bell Curve (0 to 1)
            time_factor = np.sin((hour - 6) * np.pi / 13)
            time_factor = max(0, time_factor) 
        else:
            time_factor = 0.0

        if time_factor <= 0:
            return 0.0

        # 2. Weather Impact (Cloud Cover)
        # Cloud cover 0-100. 
        # IMPROVEMENT: Higher efficiency panels often capture diffuse light better.
        # Base loss is 80% at full cloud. Premium panels might only lose 70%.
        base_cloud_loss_factor = 0.8
        if self.efficiency > 0.20: # Premium
            base_cloud_loss_factor = 0.7 
        elif self.efficiency < 0.15: # Thin Film/Old
            base_cloud_loss_factor = 0.9

        cloud_cover = weather_data.get('Cloud', 20.0)
        cloud_factor = 1.0 - (cloud_cover / 100.0 * base_cloud_loss_factor)

        # 3. Weather Impact (UV / Clarity) - slightly boosts efficiency if high UV
        uv_index = weather_data.get('UV', 5.0)
        uv_factor = 1.0 + (uv_index / 10.0 * 0.1) # Up to 10% boost on very clear high UV days

        # 4. Temperature Derating (Panels lose efficiency as they get hot)
        # Standard: -0.4% / °C
        # Premium: -0.3% / °C (Better temp coefficient)
        temp_coeff = 0.004
        if self.efficiency > 0.20:
            temp_coeff = 0.003
            
        temp_c = weather_data.get('Temperature', 25.0)
        temp_loss = max(0, (temp_c - 25) * temp_coeff)
        temp_factor = 1.0 - temp_loss

        # Total Estimated Power
        # Formula: Rated_KW * Irradiance_Factor * Losses
        estimated_kw = self.system_size_kw * time_factor * cloud_factor * uv_factor * temp_factor * self.performance_ratio
        
        return max(0.0, round(estimated_kw, 2))

    def estimate_daily_production(self, weather_data):
        """
        Estimate total kWh for the *whole day* assuming current weather roughly holds.
        (Simplified integration)
        """
        # Average peak sun hours approx 5 hours equiv
        # Adjusted by current weather factor
        # Adjusted by current weather factor
        # Reuse the cloud logic from instant power
        base_cloud_loss_factor = 0.6 # Daily average loss factor (lower than instant)
        if self.efficiency > 0.20:
             base_cloud_loss_factor = 0.5 # Better collection
        elif self.efficiency < 0.15:
             base_cloud_loss_factor = 0.7

        cloud_cover = weather_data.get('Cloud', 20.0)
        weather_efficiency = 1.0 - (cloud_cover / 100.0 * base_cloud_loss_factor)
        
        peak_sun_hours = 5.0 # Average for many locations
        estimated_kwh = self.system_size_kw * peak_sun_hours * weather_efficiency * self.performance_ratio
        
        return round(estimated_kwh, 1)

    def calculate_roi(self, monthly_bill, current_tariff_kwh, system_cost=5000):
        """
        Estimate ROI based on bill savings.
        :param monthly_bill: User's current avg monthly bill ($)
        :param current_tariff_kwh: Cost per kWh ($)
        :param system_cost: Total installation cost ($)
        """
        # Est. Monthly Generation
        daily_avg_kwh = self.system_size_kw * 4.5 * self.performance_ratio # Conservative 4.5 peak hours
        monthly_generation_kwh = daily_avg_kwh * 30
        
        potential_savings = monthly_generation_kwh * current_tariff_kwh
        
        # Capped at current bill (assuming Net Metering usually zeros out bill but rarely pays massive cash back)
        actual_savings = min(monthly_bill, potential_savings)
        
        years_to_breakeven = system_cost / (actual_savings * 12) if actual_savings > 0 else 99
        
        return {
            'monthly_production_kwh': int(monthly_generation_kwh),
            'monthly_savings': round(actual_savings, 2),
            'breakeven_years': round(years_to_breakeven, 1),
            'new_bill': max(0, monthly_bill - actual_savings)
        }
