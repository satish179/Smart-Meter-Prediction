class EnergyChatbot:
    def __init__(self, df, current_cost_kwh=0.15):
        """
        :param df: The main dataframe with 'energy_kwh' and time index
        """
        self.df = df
        self.cost_per_kwh = current_cost_kwh

    def get_response(self, user_input):
        user_input = user_input.lower()
        
        # 1. High Bill Analysis
        if "bill" in user_input and ("high" in user_input or "expensive" in user_input):
            return self.analyze_high_bill()

        # 1.1 Budget planning
        if "budget" in user_input or "limit" in user_input:
            return self.budget_tip()
            
        # 2. Washing Machine / Best Time
        if "washing machine" in user_input or "best time" in user_input:
            return "Based on your forecast, solar potential is highest between **11 AM and 2 PM**. Running heavy appliances then can save the most money."

        # 2.1 EV charging
        if "ev" in user_input or "car charging" in user_input or "charger" in user_input:
            return "For lower peak stress, prefer EV charging in late night or early afternoon windows. Avoid evening peak hours if possible."
            
        # 3. Savings / Dimming
        if "save" in user_input or "saving" in user_input:
            return "Small changes add up! Dimming lights by 20% can save about $5/month. Unplugging 'vampire' devices like chargers could save another $10!"

        # 3.1 Carbon footprint
        if "carbon" in user_input or "co2" in user_input or "emission" in user_input:
            return self.carbon_summary()
            
        # 4. Current Usage
        if "current" in user_input or "now" in user_input:
            # We would need live data passed in, but generic response for now
            return "Check the 'Live Telemetry' tab. If you see spikes, try turning off the AC or heater for a while."

        # 4.1 Peak hour
        if "peak" in user_input or "highest usage" in user_input:
            return self.peak_hour_summary()

        # 4.2 Forecast style question
        if "forecast" in user_input or "tomorrow" in user_input or "next 24" in user_input:
            return self.forecast_style_response()

        # 4.3 Night load / vampire load
        if "night" in user_input or "vampire" in user_input or "standby" in user_input:
            return self.night_usage_tip()

        # 4.4 Solar
        if "solar" in user_input or "panel" in user_input:
            return "Solar contribution is strongest around midday. If you have flexible loads, run them near noon to maximize self-consumption."

        # 4.5 Anomaly
        if "anomaly" in user_input or "spike" in user_input or "alert" in user_input:
            return "In the Live Telemetry tab, spikes usually come from high-load devices like HVAC, heater, or EV charging. Check those first."
            
        # 5. General Greeting
        if "hello" in user_input or "hi" in user_input:
            return "Hello! I'm Lumina. Ask me about your energy usage, bills, or how to save money!"

        return "I'm still learning! Try asking: 'Why is my bill high?' or 'When should I run laundry?'"

    def analyze_high_bill(self):
        """
        Simple logic to look at yesterday's data vs average
        """
        if self.df is None or self.df.empty:
            return "I don't have enough data yet to analyze your bill. Try again later!"
            
        # Get yesterday's consumption
        # Simplified: just looking at total average vs recent
        total_avg = self.df['energy_kwh'].mean() * 24 # Daily approx
        
        return f"Your average daily usage is about **{total_avg:.1f} kWh** (${total_avg * self.cost_per_kwh:.2f}). If your bill seems high, check if you ran the AC or Heater longer than usual yesterday!"

    def budget_tip(self):
        if self.df is None or self.df.empty:
            return "Set a daily target and monitor Live Telemetry to keep heavy devices under control during peak times."
        daily_kwh = self.df['energy_kwh'].mean() * 24
        daily_cost = daily_kwh * self.cost_per_kwh
        suggested = max(0.0, daily_cost * 0.9)
        return f"Your estimated daily cost is about **${daily_cost:.2f}**. A practical budget target is around **${suggested:.2f}/day** with load shifting."

    def carbon_summary(self):
        if self.df is None or self.df.empty:
            return "I need more data to estimate carbon footprint accurately."
        daily_kwh = self.df['energy_kwh'].mean() * 24
        daily_co2 = daily_kwh * 0.82
        return f"Your estimated daily carbon footprint is **{daily_co2:.1f} kg CO2** based on current usage patterns."

    def peak_hour_summary(self):
        if self.df is None or self.df.empty:
            return "I don't have enough data to estimate your peak hour yet."
        by_hour = self.df.groupby(self.df.index.hour)['energy_kwh'].mean()
        peak_hour = int(by_hour.idxmax())
        peak_val = float(by_hour.max())
        return f"Your typical peak hour is around **{peak_hour:02d}:00**, with average usage near **{peak_val:.2f} kWh**."

    def forecast_style_response(self):
        if self.df is None or self.df.empty:
            return "Forecast is available in the Forecast tab once location/weather is set."
        last_24 = self.df['energy_kwh'].tail(24).sum()
        return f"Based on recent patterns, your next-day usage may stay near **{last_24:.1f} kWh**. Open the Forecast tab for hour-wise prediction."

    def night_usage_tip(self):
        if self.df is None or self.df.empty:
            return "Try checking 12 AM–5 AM usage in the Overview chart and turn off unnecessary standby devices."
        night = self.df[self.df.index.hour.isin([0, 1, 2, 3, 4])]['energy_kwh']
        if night.empty:
            return "I don't have enough night-time samples yet."
        avg_night = float(night.mean())
        return f"Your average night-time usage is about **{avg_night:.2f} kWh** per hour. Reduce standby loads to improve this."
