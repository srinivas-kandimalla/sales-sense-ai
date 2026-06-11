import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.plot_utils import apply_layout_theme, COLOR_SEQUENCE, PRIMARY_BLUE, ACCENT_TEAL, WARNING_AMBER, DANGER_RED

from utils.data_utils import render_page_header, show_empty_state, tag_html

def show_inventory():
    render_page_header("Operations > Inventory", "Inventory Optimization & Safety Stock", "inventory_2", "Compute optimal safety stocks, reorder thresholds, and EOQs")

    df = st.session_state.get("clean_df")
    
    if df is None:
        show_empty_state()
        return

    # 1. Input parameters
    st.subheader("1. Inventory Parameters")
    col_inv1, col_inv2, col_inv3 = st.columns(3)
    
    with col_inv1:
        lead_time = st.slider("Lead Time (Days)", 1, 30, 7)
        service_level = st.select_slider(
            "Target Service Level (%)", 
            options=[80, 85, 90, 95, 98, 99],
            value=95
        )
        
    with col_inv2:
        holding_cost_day = st.number_input("Holding Cost per Unit/Day (₹)", min_value=0.01, max_value=500.0, value=2.50, step=0.50)
        ordering_cost = st.number_input("Ordering/Setup Cost per Order (₹)", min_value=1.0, max_value=10000.0, value=500.0, step=50.0)

    with col_inv3:
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border-top: 4px solid #378ADD;">
                <span style="font-weight:600; font-size: 0.9rem; color:#ffffff;">Optimization Standard</span>
                <p style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:5px; line-height: 1.3;">
                    Using the stochastic <b>Continuous Review (Q, R) model</b>. Z-score mapped dynamically based on selected confidence thresholds.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Map service level to Z-score
    z_scores = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.64, 98: 2.05, 99: 2.33}
    z_val = z_scores[service_level]

    # 2. Calculation logic per SKU
    skus = sorted(df["product_id"].unique().tolist())
    inv_records = []
    
    total_baseline_cost = 0.0
    total_optimal_cost = 0.0
    
    for sku in skus:
        sku_df = df[df["product_id"] == sku].sort_values("date")
        if len(sku_df) == 0:
            continue
            
        # Get historical demand statistics (last 90 days of sales)
        recent_sales = sku_df["sales_qty"].tail(90).values
        avg_daily_demand = recent_sales.mean() if len(recent_sales) > 0 else 10.0
        std_daily_demand = recent_sales.std() if len(recent_sales) > 0 else 2.0
        if pd.isna(avg_daily_demand) or avg_daily_demand == 0:
            avg_daily_demand = 1.0
        if pd.isna(std_daily_demand):
            std_daily_demand = 0.0
            
        # Current Inventory level
        latest_row = sku_df.iloc[-1]
        current_inv = latest_row["inventory_qty"]
        if pd.isna(current_inv):
            current_inv = 0
            
        # Safety Stock = Z * std(demand) * sqrt(L)
        safety_stock = int(round(z_val * std_daily_demand * np.sqrt(lead_time)))
        safety_stock = max(0, safety_stock)
        
        # Reorder Point = (avg_demand * L) + safety_stock
        reorder_point = int(round((avg_daily_demand * lead_time) + safety_stock))
        
        # Economic Order Quantity (EOQ) = sqrt( 2 * D * S / H )
        # D = Annual demand
        annual_demand = avg_daily_demand * 365
        holding_cost_year = holding_cost_day * 365
        eoq = int(round(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_year)))
        eoq = max(1, eoq)
        
        # Days of stock remaining = Current Inventory / avg_daily_demand
        days_stock = current_inv / avg_daily_demand
        
        # Status mapping
        if current_inv < reorder_point and days_stock <= 3:
            status = "Critical"
        elif current_inv < reorder_point:
            status = "Low"
        else:
            status = "Healthy"
            
        # Cost saving calculation: average inventory level comparison
        # Baseline average inventory = historical average inventory level
        hist_avg_inv = sku_df["inventory_qty"].mean()
        if pd.isna(hist_avg_inv):
            hist_avg_inv = current_inv
            
        # Optimal average inventory = Safety stock + (EOQ / 2)
        optimal_avg_inv = safety_stock + (eoq / 2.0)
        
        # Annual holding cost of baseline vs optimal
        baseline_holding_cost = hist_avg_inv * holding_cost_year
        optimal_holding_cost = optimal_avg_inv * holding_cost_year
        
        total_baseline_cost += baseline_holding_cost
        total_optimal_cost += optimal_holding_cost
        
        inv_records.append({
            "SKU": sku,
            "Category": sku_df["category"].iloc[0],
            "Daily Demand (Avg)": round(avg_daily_demand, 2),
            "Current Stock": int(current_inv),
            "Safety Stock (optimal)": safety_stock,
            "Reorder Point": reorder_point,
            "EOQ": eoq,
            "Days Remaining": round(days_stock, 1),
            "Status": status,
            "baseline_holding": baseline_holding_cost,
            "optimal_holding": optimal_holding_cost
        })

    inv_rep_df = pd.DataFrame(inv_records)

    # 3. Cost Savings Calculations & KPI
    carrying_savings = total_baseline_cost - total_optimal_cost
    # If carrying savings is negative (i.e. optimal requires more stock because safety thresholds were historically too low),
    # it implies avoiding stockouts, which is also a net benefit, but we display the actual cost comparison
    
    st.markdown("---")
    st.subheader("2. Optimization Results")
    
    save_col1, save_col2 = st.columns([2, 1])
    
    with save_col1:
        st.markdown("**SKU Optimization Status Table:**")
        display_df = inv_rep_df.drop(columns=["baseline_holding", "optimal_holding"])
        
        # Create copy for HTML display and apply tag_html
        html_display_df = display_df.copy()
        html_display_df["Status"] = html_display_df["Status"].apply(tag_html)
        
        # Render table as HTML
        html_table = html_display_df.to_html(escape=False, index=False, classes="dataframe inventory-table")
        st.markdown(
            f"""
            <div style="overflow-x: auto; max-height: 320px; border: 1px solid rgba(128,128,128,0.15); border-radius: 8px; margin-bottom: 15px;">
                {html_table}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Download Button
        csv_reorder = display_df.to_csv(index=False)
        import os
        os.makedirs("reports", exist_ok=True)
        with open("reports/reorder_recommendations.csv", "w", encoding="utf-8") as f:
            f.write(csv_reorder)
            
        st.download_button(
            label="📥 Download Reorder Recommendations CSV",
            data=csv_reorder,
            file_name="reorder_recommendations.csv",
            mime="text/csv"
        )
        st.caption("💾 Saved to workspace: `reports/reorder_recommendations.csv`")
        
    with save_col2:
        st.markdown("**Carrying Costs Optimization Summary**")
        savings_text = f"₹{carrying_savings:,.2f}" if carrying_savings >= 0 else f"-₹{abs(carrying_savings):,.2f}"
        card_class = "teal" if carrying_savings >= 0 else "amber"
        subtitle = "Annual holding capital saved" if carrying_savings >= 0 else "Capital adjustment to prevent stockouts"
        
        st.markdown(
            f"""
            <div class="kpi-card {card_class}" style="margin-bottom: 20px;">
                <div class="kpi-title">Carrying Capital Saved</div>
                <div class="kpi-value">{savings_text}</div>
                <div class="kpi-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Breakdown statistics
        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; font-size: 0.85rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                    <span>Baseline Annual Holding Cost:</span>
                    <b>₹{total_baseline_cost:,.0f}</b>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                    <span>Optimal Annual Holding Cost:</span>
                    <b>₹{total_optimal_cost:,.0f}</b>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Target Service Level Z-Score:</span>
                    <b>{z_val}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Bar chart
    st.markdown("---")
    st.subheader("📊 Stock Levels vs Reorder Thresholds per SKU")
    
    # We create a double bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=inv_rep_df["SKU"],
        y=inv_rep_df["Current Stock"],
        name="Current Stock Qty",
        marker_color=PRIMARY_BLUE
    ))
    fig.add_trace(go.Bar(
        x=inv_rep_df["SKU"],
        y=inv_rep_df["Reorder Point"],
        name="Reorder Point Threshold",
        marker_color=WARNING_AMBER
    ))
    
    fig = apply_layout_theme(fig, "Stock Adequacy Benchmark")
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # Save optimized data for reports
    st.session_state["inventory_report_data"] = inv_rep_df
