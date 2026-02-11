
# ... existing code ...

with tab_solar:
    st.subheader("☀️ Solar Power Estimator")
    
    col_solar1, col_solar2 = st.columns([1, 2])
    
    with col_solar1:
        st.info("Simulate how a Rooftop Solar System would perform right now based on current weather.")
        
        system_size = st.slider("System Size (kW)", 1.0, 10.0, 5.0, 0.5)
        panel_efficiency = st.selectbox("Panel Type", ["Standard (18%)", "Premium (21%)", "Thin Film (14%)"], index=0)
        
        eff_map = {"Standard (18%)": 0.18, "Premium (21%)": 0.21, "Thin Film (14%)": 0.14}
        
        # Instantiate Simulator
        solar_sim = SolarSimulator(system_size_kw=system_size, efficiency=eff_map[panel_efficiency])
        
        # Real-time Calculation
        current_solar_kw = solar_sim.calculate_instant_power(live_weather)
        
        # Gauge Visualization (Simple Metric for now)
        st.metric(
            label="Current Generation Potential",
            value=f"{current_solar_kw} kW",
            delta=f"{last_power_reading:.1f} kW Usage" if 'last_power_reading' in locals() else None,
            delta_color="normal" # Green if generation > usage? logic usually inverse for delta, but here context matters
        )
        
        if current_solar_kw > 0.1:
            st.success(f"Converting Sunlight! ({live_weather.get('Cloud', 0)}% Cloud Cover)")
        else:
            st.warning("Low/No Production (Night or Heavy Cloud)")

    with col_solar2:
        st.markdown("### Daily Estimation")
        
        predicted_daily_kwh = solar_sim.estimate_daily_production(live_weather)
        
        # Compare with average daily usage
        coverage = (predicted_daily_kwh / avg_daily * 100) if avg_daily > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. Daily Production", f"{predicted_daily_kwh} kWh")
        c2.metric("Grid Offset", f"{min(100, coverage):.1f}%")
        c3.metric("Est. Daily Savings", f"${predicted_daily_kwh * cost_per_kwh:.2f}")
        
        st.progress(min(1.0, coverage/100))
        
        st.divider()
        st.markdown("### Net Metering Simulation (Monthly)")
        
        # Calculate ROI
        roi_data = solar_sim.calculate_roi(monthly_bill=total_cost, current_tariff_kwh=cost_per_kwh, system_cost=system_size * 1000) # Approx $1000/kW
        
        roi_c1, roi_c2, roi_c3 = st.columns(3)
        roi_c1.metric("New Monthly Bill", f"${roi_data['new_bill']:.2f}", delta=f"-${roi_data['monthly_savings']:.2f}")
        roi_c2.metric("ROI Breakeven", f"{roi_data['breakeven_years']} Years")
        roi_c3.metric("CO2 Saved/Month", f"{roi_data['monthly_production_kwh'] * 0.82:.0f} kg")
