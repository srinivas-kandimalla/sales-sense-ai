import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import holidays
import plotly.express as px
import plotly.graph_objects as go
from utils.plot_utils import apply_layout_theme, COLOR_SEQUENCE, PRIMARY_BLUE, ACCENT_TEAL
from datetime import datetime

from utils.data_utils import render_page_header, show_empty_state

def show_preprocessing():
    render_page_header("Data > Preprocessing", "Data Preprocessing Pipeline", "settings_suggest", "Prepare your dataset for Machine Learning modeling")

    raw_df = st.session_state.get("df")
    
    if raw_df is None:
        show_empty_state()
        return
        
    df = raw_df.copy()
    st.write(f"**Current Dataset Shape:** {df.shape[0]:,} rows, {df.shape[1]} columns")

    # We will use steps/tabs to guide the user
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Impute Missing Values",
        "2. Outlier Detection",
        "3. Feature Engineering",
        "4. Scaling & Encoding"
    ])

    # 1. Missing Value Imputation Tab
    with tab1:
        st.subheader("Step 1 — Missing Value Handler")
        
        # Calculate missing counts
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        if not cols_with_missing:
            st.success("🎉 No missing values detected in your dataset!")
        else:
            st.warning(f"Detected missing values in columns: {', '.join(cols_with_missing)}")
            
            # Show a beautiful null heatmap using Plotly
            # We sample if the dataset is too large, to keep it fast
            sample_size = min(len(df), 2000)
            sample_df = df.sample(sample_size, random_state=42).sort_index()
            null_matrix = sample_df.isnull().astype(int)
            
            fig_heatmap = px.imshow(
                null_matrix,
                labels=dict(x="Columns", y="Row Index", color="Is Null"),
                color_continuous_scale=[[0, "#1e1e2e"], [1, "#E24B4A"]],
                title=f"Missing Value Heatmap (Sample of {sample_size} rows - Red represents nulls)"
            )
            fig_heatmap.update_layout(coloraxis_showscale=False)
            fig_heatmap = apply_layout_theme(fig_heatmap)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("Choose imputation strategy for each column:")
            imputation_methods = {}
            for col in cols_with_missing:
                default_idx = 0
                if col == "sales_qty":
                    default_idx = 1
                elif col == "price":
                    default_idx = 2
                elif col == "inventory_qty":
                    default_idx = 3
                
                imputation_methods[col] = st.selectbox(
                    f"Imputation for '{col}' (Nulls: {missing_counts[col]} | {missing_counts[col]/len(df)*100:.2f}%)",
                    options=["Drop Rows", "Mean Imputation", "Median Imputation", "Forward Fill (ffill)"],
                    index=default_idx,
                    key=f"imp_{col}"
                )

    # 2. Outlier Detection Tab
    with tab2:
        st.subheader("Step 2 — Outlier Detection")
        st.write("We use the IQR (Interquartile Range) method to identify anomalies in numeric columns.")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target promotion flags if present
        if "promotion_flag" in numeric_cols:
            numeric_cols.remove("promotion_flag")
            
        selected_outlier_col = st.selectbox("Select Column to Analyze for Outliers", numeric_cols)
        
        # Calculate IQR
        q1 = df[selected_outlier_col].quantile(0.25)
        q3 = df[selected_outlier_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = ((df[selected_outlier_col] < lower_bound) | (df[selected_outlier_col] > upper_bound)).sum()
        st.write(f"**IQR Bounds for '{selected_outlier_col}':** {lower_bound:.2f} to {upper_bound:.2f}")
        st.write(f"**Outliers detected:** {outliers_count:,} rows ({outliers_count/len(df)*100:.2f}%)")
        
        # Option to cap or remove
        outlier_strategy = st.radio(
            "Outlier Treatment Strategy:",
            ["Keep as is (No treatment)", "Cap at IQR thresholds (Windsorization)", "Remove rows containing outliers"],
            index=1
        )
        
        # Display Boxplots
        col_box1, col_box2 = st.columns(2)
        
        with col_box1:
            fig_before = px.box(df, y=selected_outlier_col, title=f"Before Treatment: {selected_outlier_col}", color_discrete_sequence=[PRIMARY_BLUE])
            fig_before = apply_layout_theme(fig_before)
            st.plotly_chart(fig_before, use_container_width=True)
            
        with col_box2:
            # Generate temporary treated data for visualization
            temp_col = df[selected_outlier_col].copy()
            if outlier_strategy == "Cap at IQR thresholds (Windsorization)":
                temp_col = temp_col.clip(lower=lower_bound, upper=upper_bound)
                title_text = "After Treatment (Capped)"
            elif outlier_strategy == "Remove rows containing outliers":
                temp_col = temp_col[(temp_col >= lower_bound) & (temp_col <= upper_bound)]
                title_text = "After Treatment (Removed)"
            else:
                title_text = "After Treatment (No Change)"
                
            fig_after = px.box(y=temp_col, title=f"{title_text}: {selected_outlier_col}", color_discrete_sequence=[ACCENT_TEAL])
            fig_after = apply_layout_theme(fig_after)
            st.plotly_chart(fig_after, use_container_width=True)

    # 3. Feature Engineering Tab
    with tab3:
        st.subheader("Step 3 — Feature Engineering")
        st.write("We will automatically enrich the dataset with the following historical and temporal features:")
        
        features_to_create = {
            "Lags": "lag_7, lag_14, lag_30 (historical sales volume from 7, 14, and 30 days ago)",
            "Rolling Averages": "rolling_mean_7, rolling_mean_30 (smoothed historical averages)",
            "Calendar Features": "month, weekday (0-6), is_weekend (0/1)",
            "Holiday Flags": "is_holiday (1 if date is a public holiday in India, else 0)"
        }
        
        for name, desc in features_to_create.items():
            st.markdown(f"- **{name}**: {desc}")
            
        enable_lags = st.checkbox("Create Lag & Rolling features (Requires date and product_id columns)", value=True)
        enable_holidays = st.checkbox("Create India Public Holiday flags", value=True)

    # 4. Scaling & Encoding Tab
    with tab4:
        st.subheader("Step 4 — Normalisation & Encoding")
        st.write("Prepare the data scale and categories for deep learning and regression algorithms.")
        
        scale_cols = st.multiselect(
            "Select Numeric Columns to Scale (MinMaxScaler to [0, 1])",
            options=["sales_qty", "price", "revenue", "inventory_qty"],
            default=["sales_qty", "price", "inventory_qty"]
        )
        
        encode_cols = st.multiselect(
            "Select Categorical Columns to Encode (LabelEncoder)",
            options=["category", "product_id"],
            default=["category", "product_id"]
        )
        
        st.markdown("---")
        
        # Run Button
        if st.button("🔥 Run Preprocessing Pipeline", type="primary"):
            p_bar = st.progress(0, text="Starting pipeline...")
            
            try:
                # Step 1: Missing value imputation
                p_bar.progress(15, text="Applying missing value treatments...")
                df_clean = df.copy()
                
                # Check for columns missing
                if 'cols_with_missing' in locals() or 'cols_with_missing' in globals() or True:
                    # Recalculate cols with missing
                    temp_missing = df_clean.isnull().sum()
                    cols_missing = temp_missing[temp_missing > 0].index.tolist()
                    for col in cols_missing:
                        default_method = "Drop Rows"
                        if col == "sales_qty":
                            default_method = "Mean Imputation"
                        elif col == "price":
                            default_method = "Median Imputation"
                        elif col == "inventory_qty":
                            default_method = "Forward Fill (ffill)"
                        
                        method = st.session_state.get(f"imp_{col}", default_method)
                        if method == "Drop Rows":
                            df_clean = df_clean.dropna(subset=[col])
                        elif method == "Mean Imputation":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        elif method == "Median Imputation":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        elif method == "Forward Fill (ffill)":
                            # Sort by product and date before ffilling
                            if "product_id" in df_clean.columns and "date" in df_clean.columns:
                                df_clean = df_clean.sort_values(["product_id", "date"])
                            df_clean[col] = df_clean.groupby("product_id", group_keys=False)[col].apply(lambda x: x.ffill().bfill())
                
                # Step 2: Outlier treatment
                p_bar.progress(40, text="Treating outliers...")
                if outlier_strategy != "Keep as is (No treatment)":
                    q1 = df_clean[selected_outlier_col].quantile(0.25)
                    q3 = df_clean[selected_outlier_col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    if outlier_strategy == "Cap at IQR thresholds (Windsorization)":
                        df_clean[selected_outlier_col] = df_clean[selected_outlier_col].clip(lower=lower_bound, upper=upper_bound)
                    elif outlier_strategy == "Remove rows containing outliers":
                        df_clean = df_clean[(df_clean[selected_outlier_col] >= lower_bound) & (df_clean[selected_outlier_col] <= upper_bound)]
                
                # Step 3: Feature Engineering
                p_bar.progress(65, text="Generating features...")
                # Sort first
                df_clean["date"] = pd.to_datetime(df_clean["date"])
                df_clean = df_clean.sort_values(["product_id", "date"]).reset_index(drop=True)
                
                if enable_lags:
                    # Group by product to calculate lag and rolling features safely
                    df_clean["lag_7"] = df_clean.groupby("product_id")["sales_qty"].shift(7)
                    df_clean["lag_14"] = df_clean.groupby("product_id")["sales_qty"].shift(14)
                    df_clean["lag_30"] = df_clean.groupby("product_id")["sales_qty"].shift(30)
                    
                    # Rolling mean requires shift(1) to avoid data leakage!
                    df_clean["rolling_mean_7"] = df_clean.groupby("product_id")["sales_qty"].shift(1).rolling(window=7, min_periods=1).mean()
                    df_clean["rolling_mean_30"] = df_clean.groupby("product_id")["sales_qty"].shift(1).rolling(window=30, min_periods=1).mean()
                    
                # Time features
                df_clean["month"] = df_clean["date"].dt.month
                df_clean["weekday"] = df_clean["date"].dt.weekday
                df_clean["is_weekend"] = df_clean["weekday"].isin([5, 6]).astype(int)
                
                if enable_holidays:
                    in_holidays = holidays.India(years=df_clean["date"].dt.year.unique().tolist())
                    df_clean["is_holiday"] = df_clean["date"].apply(lambda d: 1 if d in in_holidays else 0)
                else:
                    df_clean["is_holiday"] = 0
                
                # Fill new NaNs from lag features
                df_clean = df_clean.fillna(0)
                
                # Step 4: Normalisation and Encoding
                p_bar.progress(85, text="Applying scaling and label encoding...")
                
                # Apply MinMaxScaler
                scalers = {}
                for col in scale_cols:
                    if col in df_clean.columns:
                        scaler = MinMaxScaler()
                        # We keep scale parameters if needed, but for now we direct fit_transform
                        # We save a copy of original values if needed, but standard preprocessing usually replaces
                        df_clean[col + "_scaled"] = scaler.fit_transform(df_clean[[col]])
                        scalers[col] = scaler
                        
                # Apply LabelEncoder
                encoders = {}
                for col in encode_cols:
                    if col in df_clean.columns:
                        le = LabelEncoder()
                        df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
                        encoders[col] = le
                
                # Complete
                p_bar.progress(100, text="Pipeline execution complete!")
                
                # Show report
                st.success("🎉 Preprocessing successfully completed!")
                st.write(f"**Shape Before:** {df.shape[0]:,} rows, {df.shape[1]} columns")
                st.write(f"**Shape After:** {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
                
                # Preview clean dataset
                st.subheader("Cleaned Dataset Preview")
                st.dataframe(df_clean.head(5), use_container_width=True)
                
                # Save to session state
                st.session_state["clean_df"] = df_clean
                
                # Add system alert
                st.session_state["alerts"].append({
                    "type": "success",
                    "text": f"Preprocessed dataset saved with {df_clean.shape[1]} features.",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as ex:
                st.error(f"Failed to process pipeline: {ex}")
                import traceback
                st.write(traceback.format_exc())
