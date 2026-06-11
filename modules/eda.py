import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.plot_utils import (
    create_line_chart, 
    create_bar_chart, 
    create_heatmap, 
    create_histogram,
    PRIMARY_BLUE, 
    ACCENT_TEAL
)

from utils.data_utils import render_page_header, show_empty_state

def show_eda():
    render_page_header("Analytics > EDA", "Exploratory Data Analysis", "analytics", "Deep dive into sales trends, seasonality, and patterns")

    df = st.session_state.get("clean_df")
    
    if df is None:
        show_empty_state()
        return

    # Check date type
    df["date"] = pd.to_datetime(df["date"])

    st.subheader("📋 Descriptive Statistics Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # 1. Monthly Sales Trend Line Chart & YoY Comparison
    st.markdown("---")
    col_trend, col_yoy = st.columns(2)
    
    with col_trend:
        st.subheader("Monthly Sales Trend")
        # Group by Year-Month
        df_monthly = df.copy()
        df_monthly["year_month"] = df_monthly["date"].dt.to_period("M").astype(str)
        monthly_trend = df_monthly.groupby("year_month")["sales_qty"].sum().reset_index()
        fig_trend = create_line_chart(
            monthly_trend, 
            "year_month", 
            "sales_qty", 
            title="Total Monthly Sales Quantity",
            labels={"year_month": "Month", "sales_qty": "Units Sold"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_yoy:
        st.subheader("Year-over-Year (YoY) Sales Comparison")
        # Check if multi-year data exists
        years = df["date"].dt.year.unique()
        if len(years) > 1:
            yoy_df = df.copy()
            yoy_df["Year"] = yoy_df["date"].dt.year.astype(str)
            yoy_df["Month"] = yoy_df["date"].dt.month
            
            # Group by Year and Month
            yoy_grouped = yoy_df.groupby(["Year", "Month"])["sales_qty"].sum().reset_index()
            # Map month numbers to short names for readability
            month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                         7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            yoy_grouped["Month Name"] = yoy_grouped["Month"].map(month_map)
            # Sort by Month number to maintain correct calendar order
            yoy_grouped = yoy_grouped.sort_values(["Year", "Month"])
            
            fig_yoy = create_line_chart(
                yoy_grouped, 
                "Month Name", 
                "sales_qty", 
                color_col="Year",
                title="Monthly Sales Qty compared across Years",
                labels={"sales_qty": "Units Sold", "Month Name": "Calendar Month"}
            )
            st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.info("YoY chart requires at least two distinct years of data. Active dataset only contains one year.")

    # 2. Seasonality Index & Correlation Matrix
    st.markdown("---")
    col_season, col_corr = st.columns(2)
    
    with col_season:
        st.subheader("Monthly Seasonality Index")
        # Index = average sales in Month X / average sales overall
        df_season = df.copy()
        df_season["Month"] = df_season["date"].dt.month
        monthly_avg = df_season.groupby("Month")["sales_qty"].mean()
        overall_avg = df_season["sales_qty"].mean()
        seasonality_idx = (monthly_avg / overall_avg).reset_index()
        
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        seasonality_idx["Month Name"] = seasonality_idx["Month"].map(month_names)
        seasonality_idx = seasonality_idx.sort_values("Month")
        
        fig_season = create_bar_chart(
            seasonality_idx, 
            "Month Name", 
            "sales_qty", 
            title="Seasonality Multiplier (Average Month / Overall Average)",
            labels={"sales_qty": "Seasonality Index", "Month Name": "Month"}
        )
        # Add a baseline indicator line at 1.0 (average index)
        fig_season.add_hline(y=1.0, line_dash="dash", line_color="#E24B4A", annotation_text="Baseline (1.0)")
        st.plotly_chart(fig_season, use_container_width=True)

    with col_corr:
        st.subheader("Feature Correlation Heatmap")
        # Filter for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        # Remove categorical encodings or standard target columns that might clutter the map
        cols_to_corr = [c for c in numeric_df.columns if not c.endswith("_encoded")]
        
        if len(cols_to_corr) > 1:
            corr_matrix = numeric_df[cols_to_corr].corr()
            fig_corr = create_heatmap(corr_matrix, title="Correlation Matrix of Features")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Heatmap requires at least two numeric features.")

    # 3. Top SKUs & Distribution
    st.markdown("---")
    col_sku, col_dist = st.columns(2)
    
    with col_sku:
        st.subheader("Top 5 SKUs by Revenue")
        sku_revenue = df.groupby("product_id")["revenue"].sum().reset_index()
        top_5_skus = sku_revenue.sort_values("revenue", ascending=False).head(5)
        
        fig_sku = create_bar_chart(
            top_5_skus, 
            "product_id", 
            "revenue", 
            title="Highest Revenue Generating SKUs (₹)",
            labels={"product_id": "SKU", "revenue": "Total Revenue (₹)"}
        )
        st.plotly_chart(fig_sku, use_container_width=True)
        
    with col_dist:
        st.subheader("Sales Distribution Histogram")
        # Filter out extremely large values for visualization zoom if desired, but we show the histogram of raw sales
        fig_dist = create_histogram(
            df, 
            "sales_qty", 
            title="Frequency Distribution of Daily Sales Quantities",
            labels={"sales_qty": "Sales Units"}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
