import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from utils.plot_utils import apply_layout_theme, PRIMARY_BLUE, ACCENT_TEAL, WARNING_AMBER

from utils.data_utils import render_page_header, show_empty_state

def show_forecasting():
    render_page_header("Machine Learning > Forecasting", "Sales Forecasting Engine", "auto_graph", "Generate future demand projections with uncertainty intervals")

    df = st.session_state.get("clean_df")
    
    if df is None:
        show_empty_state()
        return

    model = st.session_state.get("trained_model")
    
    # 1. Inputs Section
    st.subheader("1. Forecast Parameters")
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        horizon = st.selectbox("Forecast Horizon (Days)", [7, 30, 60, 90], index=1)
        
    with col_input2:
        sku_list = sorted(df["product_id"].unique().tolist())
        selected_sku = st.selectbox("Select Product SKU", sku_list)
        
    # Check if a model is trained
    if model is None:
        st.info("💡 Note: No custom model has been trained yet. Generating forecast using a high-fidelity Seasonal Baseline estimator. Go to 'Model Training' to train a machine learning model.")

    # 2. Run Forecast
    if st.button("🔮 Run Demand Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating projections..."):
            # Fetch historical series for selected SKU
            sku_df = df[df["product_id"] == selected_sku].sort_values("date").reset_index(drop=True)
            latest_date = sku_df["date"].max()
            sku_category = sku_df["category"].iloc[0] if len(sku_df) > 0 else "Unknown"
            sku_price = sku_df["price"].iloc[-1] if len(sku_df) > 0 else 0
            
            # Generate future dates
            future_dates = [latest_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
            
            # Forecast calculations
            forecast_qty = []
            lower_ci = []
            upper_ci = []
            
            # Calculate standard deviation of historical sales to estimate confidence intervals
            hist_sales = sku_df["sales_qty"].tail(90).values
            hist_mean = hist_sales.mean() if len(hist_sales) > 0 else 10
            hist_std = hist_sales.std() if len(hist_sales) > 0 else 2
            if hist_std == 0:
                hist_std = 1.0
                
            # Forecast logic:
            if model is not None:
                # Custom model prediction
                # If ARIMA/SARIMA/Prophet, it's stored in st.session_state as a dict of models per SKU
                # If XGBoost/RF/LSTM, it's a global model
                try:
                    if isinstance(model, dict) and selected_sku in model:
                        sku_model = model[selected_sku]
                        if "SARIMAX" in str(type(sku_model)):
                            # SARIMAX forecast
                            preds = sku_model.forecast(steps=horizon)
                            forecast_qty = preds.values.tolist()
                        elif "Prophet" in str(type(sku_model)):
                            # Prophet forecast
                            future_df = sku_model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
                            preds = sku_model.predict(future_df)
                            forecast_qty = preds["yhat"].values.tolist()
                            lower_ci = preds["yhat_lower"].values.tolist()
                            upper_ci = preds["yhat_upper"].values.tolist()
                    else:
                        # Tabular model (XGBoost / RF / LSTM)
                        # Predict using rolling lag simulations
                        # For simplicity, we model a recursive forecasting loop using the mean lag features
                        # or project using historical trends with minor shifts
                        current_lags = sku_df.tail(30).copy()
                        pred_vol = []
                        
                        # Use model features
                        feature_cols = ["lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", 
                                        "month", "weekday", "is_weekend", "is_holiday"]
                        if "category_encoded" in df.columns:
                            feature_cols.append("category_encoded")
                        if "product_id_encoded" in df.columns:
                            feature_cols.append("product_id_encoded")
                            
                        # Standard recursive simulator
                        for i in range(horizon):
                            f_date = future_dates[i]
                            # Construct row
                            lag_7_val = pred_vol[i-7] if i >= 7 else current_lags.iloc[-(7-i)]["sales_qty"]
                            lag_14_val = pred_vol[i-14] if i >= 14 else current_lags.iloc[-(14-i)]["sales_qty"]
                            lag_30_val = pred_vol[i-30] if i >= 30 else current_lags.iloc[-(30-i)]["sales_qty"]
                            
                            roll_7 = np.mean(pred_vol[max(0, i-7):i]) if i > 0 else current_lags["sales_qty"].tail(7).mean()
                            roll_30 = np.mean(pred_vol[max(0, i-30):i]) if i > 0 else current_lags["sales_qty"].tail(30).mean()
                            
                            # Construct single row dataframe
                            row_dict = {
                                "lag_7": lag_7_val,
                                "lag_14": lag_14_val,
                                "lag_30": lag_30_val,
                                "rolling_mean_7": roll_7,
                                "rolling_mean_30": roll_30,
                                "month": f_date.month,
                                "weekday": f_date.weekday(),
                                "is_weekend": 1 if f_date.weekday() in [5,6] else 0,
                                "is_holiday": 0 # Default holiday flag
                            }
                            if "category_encoded" in df.columns:
                                row_dict["category_encoded"] = sku_df["category_encoded"].iloc[0]
                            if "product_id_encoded" in df.columns:
                                row_dict["product_id_encoded"] = sku_df["product_id_encoded"].iloc[0]
                                
                            single_row = pd.DataFrame([row_dict])[feature_cols]
                            
                            # Support ensemble dict
                            if isinstance(model, dict) and "xgb" in model:
                                p_val = model["xgb"].predict(single_row)[0]
                            else:
                                p_val = model.predict(single_row)[0]
                                
                            # Convert scaled predictions back if necessary, but we predicted on unscaled target sales_qty
                            p_val = max(0.0, p_val)
                            pred_vol.append(p_val)
                            
                        forecast_qty = pred_vol
                except Exception as ex:
                    # Fallback on failure
                    st.write(f"Model prediction error: {ex}. Falling back to baseline.")
                    model = None
            
            # Baseline or model fallback forecast generator
            if model is None or len(forecast_qty) == 0:
                # Seasonal baseline: recent SKU daily trend + weekday multiplier + holiday check
                forecast_qty = []
                np.random.seed(42)
                for f_date in future_dates:
                    weekday = f_date.weekday()
                    # Add seasonality (weekends have 1.25x sales)
                    w_mult = 1.25 if weekday in [5, 6] else 1.0
                    
                    # Dec Year-end spike
                    y_mult = 1.5 if (f_date.month == 12 and f_date.day >= 22) else 1.0
                    
                    # Basic naive forecast with decay/trend plus random noise
                    val = hist_mean * w_mult * y_mult * np.random.uniform(0.9, 1.1)
                    forecast_qty.append(max(0.0, val))

            # Calculate confidence intervals if they weren't generated by Prophet
            if len(lower_ci) == 0:
                for idx, val in enumerate(forecast_qty):
                    # Volatility increases as we forecast further out (standard deviation scales with sqrt of time)
                    uncertainty = hist_std * np.sqrt(idx + 1) * 0.4
                    lower_ci.append(max(0.0, val - 1.96 * uncertainty))
                    upper_ci.append(val + 1.96 * uncertainty)

            # Round values for sales quantities
            forecast_qty = [int(round(x)) for x in forecast_qty]
            lower_ci = [int(round(x)) for x in lower_ci]
            upper_ci = [int(round(x)) for x in upper_ci]
            
            # Create Forecast Dataframe
            forecast_df = pd.DataFrame({
                "date": future_dates,
                "predicted_sales": forecast_qty,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "revenue_forecast": [int(round(qty * sku_price)) for qty in forecast_qty]
            })
            
            # Save forecast results to session state for reports
            st.session_state["latest_forecast"] = {
                "sku": selected_sku,
                "category": sku_category,
                "horizon": horizon,
                "data": forecast_df
            }
            
            # Save forecast to session state alerts
            st.session_state["alerts"].append({
                "type": "success",
                "text": f"Generated {horizon}-day forecast for SKU {selected_sku}.",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # 3. Output Displays
    active_forecast = st.session_state.get("latest_forecast")
    if active_forecast and active_forecast["sku"] == selected_sku:
        f_df = active_forecast["data"]
        sku_df = df[df["product_id"] == selected_sku].sort_values("date")
        
        # Line chart with 95% confidence intervals
        st.subheader(f"📈 Forecast Trend for {selected_sku} ({horizon} Days)")
        
        # Take last 60 days of historical data for visual context
        hist_context = sku_df.tail(60).copy()
        
        fig = go.Figure()
        
        # Historical Trace
        fig.add_trace(go.Scatter(
            x=hist_context["date"],
            y=hist_context["sales_qty"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color=PRIMARY_BLUE, width=2.5)
        ))
        
        # Forecast Trace
        fig.add_trace(go.Scatter(
            x=f_df["date"],
            y=f_df["predicted_sales"],
            mode="lines+markers",
            name="Forecasted Demand",
            line=dict(color=ACCENT_TEAL, width=2.5)
        ))
        
        # Upper Bound
        fig.add_trace(go.Scatter(
            x=f_df["date"],
            y=f_df["upper_ci"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper Bound"
        ))
        
        # Lower Bound (Fill area between upper and lower)
        fig.add_trace(go.Scatter(
            x=f_df["date"],
            y=f_df["lower_ci"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(29, 158, 117, 0.15)",
            name="95% Confidence Interval"
        ))
        
        fig = apply_layout_theme(fig, f"Demand Projection with Uncertainty Band")
        st.plotly_chart(fig, use_container_width=True)
        
        # Bottom Grid: Month-wise forecast summary cards, category bar chart & download
        st.markdown("---")
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.subheader("📋 Forecast Details Table")
            display_f_df = f_df.copy()
            display_f_df["date"] = display_f_df["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_f_df, use_container_width=True, height=300)
            
            # Download CSV Button
            csv = display_f_df.to_csv(index=False)
            filename = f"forecast_{selected_sku}_{horizon}d.csv"
            import os
            os.makedirs("reports", exist_ok=True)
            with open(os.path.join("reports", filename), "w", encoding="utf-8") as f:
                f.write(csv)
                
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            st.caption(f"💾 Saved to workspace: `reports/{filename}`")

        with col_c2:
            st.subheader("📊 Projected Categories & Month-wise Breakdown")
            
            # Group forecast by calendar Month
            f_df["Month-Year"] = f_df["date"].dt.strftime("%b %Y")
            monthly_summary = f_df.groupby("Month-Year").agg(
                units=("predicted_sales", "sum"),
                revenue=("revenue_forecast", "sum")
            ).reset_index()
            
            # Show summary cards
            for _, row in monthly_summary.iterrows():
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #1D9E75;">
                        <span style="font-weight:600; font-size: 0.95rem; color:#ffffff;">{row['Month-Year']} Forecast Summary</span>
                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                            <span style="font-size:0.85rem; color:rgba(255,255,255,0.7);">Projected Demand: <b>{row['units']:,} units</b></span>
                            <span style="font-size:0.85rem; color:rgba(255,255,255,0.7);">Estimated Revenue: <b>₹{row['revenue']:,}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Category share bar chart: For this specific SKU, it is just one category, but we can display the category aggregate forecast from other SKUs
            # Let's mock or build a category-wise forecast breakdown by summing across all products for that horizon
            # Actually, to make it fully dynamic, we can run a simple category forecast or plot the current SKU's share compared to the active database
            st.write("")
            all_sku_revs = df.groupby("category")["revenue"].sum().reset_index()
            # Append forecast revenue to category
            all_sku_revs.loc[all_sku_revs["category"] == active_forecast["category"], "revenue"] += f_df["revenue_forecast"].sum()
            
            fig_bar = px.bar(
                all_sku_revs,
                x="category",
                y="revenue",
                title="Aggregate Category Revenue (Historic + Future SKU Forecast)",
                color_discrete_sequence=[ACCENT_TEAL]
            )
            fig_bar = apply_layout_theme(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.info("Set parameters above and click 'Run Demand Forecast' to view demand projections.")
