import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

def generate_sample_sales_data(filepath="data/sample_sales_data.csv"):
    """
    Generates a realistic 3-year daily sales dataset for 10 SKUs
    across 4 categories. Includes Diwali & Year-end seasonality,
    missing values, and outliers.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    # 10 SKUs configuration
    skus_config = {
        "SKU_001": {"category": "Electronics", "base_qty": 8, "price": 15000, "max_inv": 200, "replenish_threshold": 100},
        "SKU_002": {"category": "Electronics", "base_qty": 5, "price": 28000, "max_inv": 150, "replenish_threshold": 75},
        "SKU_003": {"category": "Apparel", "base_qty": 25, "price": 1200, "max_inv": 800, "replenish_threshold": 350},
        "SKU_004": {"category": "Apparel", "base_qty": 35, "price": 850, "max_inv": 500, "replenish_threshold": 120},
        "SKU_005": {"category": "FMCG", "base_qty": 120, "price": 120, "max_inv": 4000, "replenish_threshold": 2200},
        "SKU_006": {"category": "FMCG", "base_qty": 180, "price": 75, "max_inv": 2500, "replenish_threshold": 600},
        "SKU_007": {"category": "FMCG", "base_qty": 90, "price": 210, "max_inv": 3000, "replenish_threshold": 1200},
        "SKU_008": {"category": "Home", "base_qty": 12, "price": 4500, "max_inv": 300, "replenish_threshold": 160},
        "SKU_009": {"category": "Home", "base_qty": 15, "price": 3200, "max_inv": 400, "replenish_threshold": 200},
        "SKU_010": {"category": "Home", "base_qty": 10, "price": 5800, "max_inv": 150, "replenish_threshold": 35},
    }
    
    records = []
    
    for sku, cfg in skus_config.items():
        category = cfg["category"]
        base_qty = cfg["base_qty"]
        price = cfg["price"]
        max_inv = cfg["max_inv"]
        threshold = cfg["replenish_threshold"]
        
        # Initialize inventory
        current_inv = max_inv
        
        for dt in date_range:
            month = dt.month
            day = dt.day
            weekday = dt.weekday() # 0 = Monday, 6 = Sunday
            
            # 1. Base seasonality
            season_factor = 1.0
            
            # Diwali Peak (mid-Oct to mid-Nov, varying slightly, lets say Oct 15 - Nov 20)
            if (month == 10 and day >= 15) or (month == 11 and day <= 20):
                # Major sales spike in India
                season_factor *= np.random.uniform(1.8, 2.6)
            
            # Year-end Spike (Dec 22 to Dec 31)
            elif month == 12 and day >= 22:
                season_factor *= np.random.uniform(1.4, 2.0)
            
            # Weekend effect (Fri, Sat, Sun)
            if weekday in [4, 5, 6]:
                season_factor *= np.random.uniform(1.15, 1.35)
                
            # Random weekly fluctuations
            random_noise = np.random.normal(1.0, 0.15)
            
            # Promotion flag (realistic frequency, say 5% chance)
            promotion_flag = 1 if np.random.rand() < 0.05 else 0
            promo_factor = 1.4 if promotion_flag == 1 else 1.0
            
            # Final sales quantity calculation
            sales_qty = int(base_qty * season_factor * promo_factor * random_noise)
            sales_qty = max(0, sales_qty) # Cannot be negative
            
            # Inventory calculation (sawtooth pattern)
            # Sales reduce inventory
            current_inv -= sales_qty
            
            # Replenishment check (simple lead time replenishment modeling)
            if current_inv <= threshold:
                # Stock is replenished (comes in 3 days later, but for data simplicity, we replenish it here to max_inv)
                # Let's model a realistic replenishment amount
                replenish_qty = max_inv - current_inv
                current_inv = max_inv
            else:
                replenish_qty = 0
                
            # Make sure inventory doesn't fall below zero in the historic record
            current_inv = max(0, current_inv)
            
            # Calculate revenue
            revenue = sales_qty * price
            
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": sku,
                "category": category,
                "sales_qty": sales_qty,
                "revenue": revenue,
                "price": price,
                "inventory_qty": current_inv,
                "promotion_flag": promotion_flag
            })
            
    df = pd.DataFrame(records)
    
    # Introduce synthetic missing values (~1% for sales_qty, price, and inventory_qty)
    mask_sales = np.random.rand(*df["sales_qty"].shape) < 0.01
    mask_price = np.random.rand(*df["price"].shape) < 0.008
    mask_inv = np.random.rand(*df["inventory_qty"].shape) < 0.005
    
    df.loc[mask_sales, "sales_qty"] = np.nan
    df.loc[mask_price, "price"] = np.nan
    df.loc[mask_inv, "inventory_qty"] = np.nan
    
    # Introduce synthetic outliers in sales_qty (~0.5% of rows, spike of 6x to 10x base_qty)
    outlier_mask = np.random.rand(*df["sales_qty"].shape) < 0.005
    for idx in df[outlier_mask].index:
        sku = df.loc[idx, "product_id"]
        base = skus_config[sku]["base_qty"]
        df.loc[idx, "sales_qty"] = int(base * np.random.uniform(6.0, 10.0))
        # Recalculate revenue for outliers
        if not pd.isna(df.loc[idx, "price"]):
            df.loc[idx, "revenue"] = df.loc[idx, "sales_qty"] * df.loc[idx, "price"]
            
    df.to_csv(filepath, index=False)
    print(f"Generated sample sales data saved to {filepath}")
    return df

def get_sales_data(filepath="data/sample_sales_data.csv"):
    """Loads sample data, generating it first if not present."""
    if not os.path.exists(filepath):
        return generate_sample_sales_data(filepath)
    return pd.read_csv(filepath)

def render_page_header(breadcrumb, title_text, icon, subtitle):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_html = f"""
    <div style="margin-bottom: 25px;">
        <div style="font-size: 0.8rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">{breadcrumb}</div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span class="material-symbols-outlined" style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; color: var(--text-color, inherit);">{icon}</span>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; color: var(--text-color, inherit); display: inline-block; vertical-align: middle;">{title_text}</h1>
            <span style="background: rgba(29, 158, 117, 0.15); color: #1D9E75; border: 1px solid rgba(29, 158, 117, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-left: 10px;">Live</span>
        </div>
        <div style="font-size: 0.95rem; color: #a1a1aa; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; margin-top: 4px;">
            <span>{subtitle}</span>
            <span style="font-size: 0.8rem; opacity: 0.7;">Last Updated: {timestamp}</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def show_empty_state():
    st.markdown(
        f"""
        <div style="text-align: center; padding: 50px 20px; margin: 20px 0; border: 2px dashed rgba(128,128,128,0.2); border-radius: 12px; background: rgba(128,128,128,0.02);">
            <div style="margin-bottom: 15px;">
                <span class="material-symbols-outlined" style="font-size: 4rem; color: var(--text-color, inherit); opacity: 0.6;">cloud_upload</span>
            </div>
            <h3 style="margin-bottom: 10px; color: var(--text-color, inherit); font-size: 1.35rem; font-weight: 600;">No data yet — go to Data Upload to get started</h3>
            <p style="font-size: 0.9rem; color: #a1a1aa; margin-bottom: 25px; max-width: 500px; margin-left: auto; margin-right: auto;">
                You need to upload a dataset or load standard sample data before you can access this module's features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Go to Data Upload", use_container_width=True, type="primary"):
        st.session_state["selected_page"] = "Data Upload"
        st.rerun()

def tag_html(status):
    if status == "Critical":
        bg_color = "#E24B4A"
    elif status == "Low":
        bg_color = "#EF9F27"
    elif status == "Healthy":
        bg_color = "#1D9E75"
    else:
        bg_color = "#6b7280"
    return f'<span style="background-color: {bg_color}; color: #ffffff; padding: 4px 12px; border-radius: 50px; font-size: 0.75rem; font-weight: 600; display: inline-block; text-align: center; text-transform: uppercase; letter-spacing: 0.5px;">{status}</span>'
