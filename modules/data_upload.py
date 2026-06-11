import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_utils import render_page_header

def show_data_upload():
    render_page_header("Data > Data Ingestion", "Data Upload & Ingestion", "cloud_upload", "Import your sales and inventory records (CSV/XLSX)")

    # Display current active dataset
    active_source = st.session_state.get("data_source", "None")
    if active_source:
        st.info(f"**Active Data Source:** {active_source}")

    # Uploader
    uploaded_file = st.file_uploader(
        "Upload Sales Data File",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing daily sales transactions and inventory logs."
    )

    # If file is uploaded, process it
    if uploaded_file is not None:
        try:
            with st.spinner("Parsing file..."):
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded file '{uploaded_file.name}' with {len(df_raw)} rows!")
            
            # Fuzzy match columns
            col_options = list(df_raw.columns)
            
            # Helper to find closest match
            def find_match(targets, options):
                for target in targets:
                    for opt in options:
                        if target.lower() in opt.lower() or opt.lower() in target.lower():
                            return opt
                return options[0] if options else None

            # Detect defaults
            detected_date = find_match(["date", "dt", "time", "day", "period"], col_options)
            detected_sku = find_match(["product_id", "sku", "item_id", "product", "sku_id"], col_options)
            detected_sales = find_match(["sales_qty", "qty", "quantity", "sales", "volume", "units"], col_options)
            detected_price = find_match(["price", "unit_price", "rate", "mrp"], col_options)
            detected_cat = find_match(["category", "cat", "dept", "department", "group"], col_options)
            detected_inv = find_match(["inventory_qty", "inventory", "stock", "stock_qty", "stock_level", "on_hand"], col_options)

            st.subheader("Map Columns to Standard Schema")
            st.write("Ensure your columns match the required fields. We have auto-detected them where possible:")

            # Form to confirm mappings
            col1, col2, col3 = st.columns(3)
            with col1:
                date_map = st.selectbox("Date Column", col_options, index=col_options.index(detected_date) if detected_date in col_options else 0)
                sku_map = st.selectbox("Product ID (SKU) Column", col_options, index=col_options.index(detected_sku) if detected_sku in col_options else 0)
            with col2:
                sales_map = st.selectbox("Sales Qty Column", col_options, index=col_options.index(detected_sales) if detected_sales in col_options else 0)
                price_map = st.selectbox("Price Column", col_options, index=col_options.index(detected_price) if detected_price in col_options else 0)
            with col3:
                cat_map = st.selectbox("Category Column", col_options, index=col_options.index(detected_cat) if detected_cat in col_options else 0)
                inv_map = st.selectbox("Inventory Qty Column", col_options, index=col_options.index(detected_inv) if detected_inv in col_options else 0)

            # Optional mapping confirm button
            if st.button("Confirm Column Mapping"):
                df_mapped = df_raw.rename(columns={
                    date_map: "date",
                    sku_map: "product_id",
                    sales_map: "sales_qty",
                    price_map: "price",
                    cat_map: "category",
                    inv_map: "inventory_qty"
                })
                
                # Keep only these columns and any other existing columns
                standard_cols = ["date", "product_id", "sales_qty", "price", "category", "inventory_qty"]
                for col in standard_cols:
                    if col not in df_mapped.columns:
                        # Create with NaNs if not present
                        df_mapped[col] = np.nan
                        
                # Ensure date is parsed to datetime
                try:
                    df_mapped["date"] = pd.to_datetime(df_mapped["date"]).dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                
                # If revenue is not present, calculate it
                if "revenue" not in df_mapped.columns:
                    df_mapped["revenue"] = df_mapped["sales_qty"] * df_mapped["price"]
                
                # Save to session_state
                st.session_state["df"] = df_mapped
                # Refresh clean_df too
                st.session_state["clean_df"] = df_mapped.dropna().copy()
                st.session_state["clean_df"]["date"] = pd.to_datetime(st.session_state["clean_df"]["date"])
                
                st.session_state["data_source"] = uploaded_file.name
                
                # Add system alert
                st.session_state["alerts"].append({
                    "type": "success",
                    "text": f"New dataset '{uploaded_file.name}' successfully mapped and loaded.",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success("Data successfully processed and mapping applied!")
                st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Section to restore sample data
    st.markdown("---")
    st.subheader("💡 Sandbox Mode / Default Sample Data")
    st.write("If you do not have a dataset handy, you can either load our realistic sample data directly or download it to test the uploader:")
    
    col_sandbox1, col_sandbox2 = st.columns(2)
    with col_sandbox1:
        if st.button("Load Standard Sample Data", use_container_width=True):
            df = pd.read_csv('data/sample_sales_data.csv', parse_dates=['date'])
            st.session_state['raw_df'] = df
            st.session_state['df'] = df  # Keep for module compatibility
            st.session_state['clean_df'] = df.copy()
            st.session_state['uploaded_filename'] = 'sample_sales_data.csv'
            st.session_state['data_source'] = 'sample_sales_data.csv'
            st.success(f"Sample data loaded — {len(df):,} rows ready!")
            st.rerun()
            
    with col_sandbox2:
        from utils.data_utils import get_sales_data
        sample_df = get_sales_data()
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV File",
            data=sample_csv,
            file_name="sample_sales_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Show Data validation Report & Preview
    df_current = st.session_state.get("df")
    if df_current is not None:
        st.markdown("---")
        st.subheader("👁️ Data Preview (First 10 rows)")
        st.dataframe(df_current.head(10), use_container_width=True)
        
        # Validation report
        st.subheader("📋 Data Validation Report")
        
        row_count = len(df_current)
        col_count = len(df_current.columns)
        
        null_counts = df_current.isnull().sum()
        dtypes = df_current.dtypes
        
        # Date range detection
        date_col = "date"
        date_range_str = "Unknown"
        if date_col in df_current.columns:
            try:
                date_vals = pd.to_datetime(df_current[date_col])
                min_dt = date_vals.min().strftime("%Y-%m-%d")
                max_dt = date_vals.max().strftime("%Y-%m-%d")
                days_span = (date_vals.max() - date_vals.min()).days
                date_range_str = f"{min_dt} to {max_dt} ({days_span} days)"
            except Exception:
                date_range_str = "Invalid Date Format"

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.write(f"**Total Records:** {row_count:,} rows")
            st.write(f"**Total Features:** {col_count} columns")
            st.write(f"**Date Range:** {date_range_str}")
        with col_rep2:
            unique_skus = df_current["product_id"].nunique() if "product_id" in df_current.columns else 0
            unique_cats = df_current["category"].nunique() if "category" in df_current.columns else 0
            st.write(f"**Distinct SKUs (Products):** {unique_skus}")
            st.write(f"**Distinct Categories:** {unique_cats}")
            
        # Detailed null & type analysis table
        report_df = pd.DataFrame({
            "Data Type": [str(d) for d in dtypes],
            "Missing Values": null_counts,
            "Missing Value %": (null_counts / row_count) * 100
        })
        
        st.markdown("**Field Summary Table:**")
        st.table(report_df.style.format({"Missing Value %": "{:.2f}%"}))
    else:
        st.info("No data active. Please upload a file or click 'Load Standard Sample Data' to begin.")
