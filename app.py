import streamlit as st
import pandas as pd
import os
from datetime import datetime

if not os.path.exists('data/sample_sales_data.csv'):
    exec(open('generate_sample_data.py').read())

# Import modules
from modules.dashboard import show_dashboard
from modules.data_upload import show_data_upload
from modules.preprocessing import show_preprocessing
from modules.eda import show_eda
from modules.model_training import show_model_training
from modules.forecasting import show_forecasting
from modules.inventory import show_inventory
from modules.reports import show_reports
from utils.data_utils import get_sales_data

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="SalesSense - AI Sales Forecasting & Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and apply custom CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply custom CSS stylesheet
local_css("assets/style.css")

# Initialize Session State
def init_session_state():
    # Load default data if not present
    if "df" not in st.session_state:
        try:
            st.session_state["df"] = get_sales_data()
            st.session_state["data_source"] = "Default Sample Data"
        except Exception as e:
            st.session_state["df"] = None
            st.session_state["data_source"] = None
            st.error(f"Error loading sample data: {e}")

    if "clean_df" not in st.session_state:
        # Prepopulate clean_df with a raw copy initially (to make sure other screens don't crash before preprocessing)
        if st.session_state["df"] is not None:
            # Drop null rows for quick baseline compatibility
            st.session_state["clean_df"] = st.session_state["df"].dropna().copy()
            # Ensure proper datetime type
            st.session_state["clean_df"]["date"] = pd.to_datetime(st.session_state["clean_df"]["date"])
        else:
            st.session_state["clean_df"] = None

    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None
        
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}  # {model_name: model_object}
        
    if "model_metrics" not in st.session_state:
        st.session_state["model_metrics"] = {}  # {model_name: {metric_name: value}}

    if "alerts" not in st.session_state:
        st.session_state["alerts"] = [
            {"type": "info", "text": "Default sample data successfully pre-loaded.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"type": "warning", "text": "No custom ML model trained yet. Run model training for tailored forecasts.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]

    if "reports_history" not in st.session_state:
        st.session_state["reports_history"] = []

# Initialize state
init_session_state()

# Sidebar branding & Navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 15px;'>
        <h1 style='color: #378ADD; font-size: 2.2rem; margin-bottom: 0px;'>SalesSense</h1>
        <p style='color: #1D9E75; font-size: 0.9rem; margin-top: 0px;'>Intelligent Sales Forecasting</p>
    </div>
    """,
    unsafe_allow_html=True
)

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Dashboard"

navigation_pages = {
    "Dashboard": show_dashboard,
    "Data Upload": show_data_upload,
    "Data Preprocessing": show_preprocessing,
    "EDA Analysis": show_eda,
    "Model Training": show_model_training,
    "Sales Forecasting": show_forecasting,
    "Inventory Optimization": show_inventory,
    "Reports": show_reports
}

# Group navigation configurations
groups = {
    "Main": [
        {"label": "Dashboard", "icon": "dashboard", "color": "#378ADD"},
        {"label": "Data Upload", "icon": "cloud_upload", "color": "#1D9E75"},
        {"label": "Data Preprocessing", "icon": "settings_suggest", "color": "#EF9F27"}
    ],
    "Analysis": [
        {"label": "EDA Analysis", "icon": "analytics", "color": "#AB63FA"},
        {"label": "Model Training", "icon": "smart_toy", "color": "#FFA15A"}
    ],
    "Output": [
        {"label": "Sales Forecasting", "icon": "auto_graph", "color": "#19D3F3"},
        {"label": "Inventory Optimization", "icon": "inventory_2", "color": "#E24B4A"},
        {"label": "Reports", "icon": "description", "color": "#FF6692"}
    ]
}

# Render Grouped Navigation
for group_name, items in groups.items():
    st.sidebar.markdown(
        f"""
        <div style="margin-top: 15px; margin-bottom: 8px; font-weight: 700; font-size: 0.85rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 3px;">
            {group_name}
        </div>
        """,
        unsafe_allow_html=True
    )
    for item in items:
        label = item["label"]
        icon = item["icon"]
        color = item["color"]
        active = st.session_state["selected_page"] == label
        
        # Display colored icon box before each nav label
        st.sidebar.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px; margin-bottom: 4px;">
                <div style="background-color: {color}; padding: 4px; border-radius: 6px; display: flex; align-items: center; justify-content: center; width: 26px; height: 26px; color: white;">
                    <span class="material-symbols-outlined" style="font-size: 1.15rem; line-height: 1;">{icon}</span>
                </div>
                <span style="font-weight: 500; font-size: 0.95rem; color: var(--text-color, inherit);">{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.sidebar.button(f"Go to {label}", key=f"nav_btn_{label}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state["selected_page"] = label
            st.rerun()

selected_page = st.session_state["selected_page"]

# Sidebar alerts history footer
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Notification History")

alerts_html = []
for alert in st.session_state["alerts"][-4:]: # Show last 4 alerts
    color_class = "info"
    if alert["type"] == "warning":
        color_class = "amber"
    elif alert["type"] == "success":
        color_class = "success"
    elif alert["type"] == "danger":
        color_class = "danger"
    
    alerts_html.append(
        f"""
        <div class="sidebar-alert-panel {color_class}" style="color: inherit;">
            <div class="sidebar-alert-text" style="color: inherit;">{alert['text']}</div>
            <div class="sidebar-alert-time" style="color: inherit; opacity: 0.65;">{alert['time']}</div>
        </div>
        """
    )

st.sidebar.markdown("".join(alerts_html), unsafe_allow_html=True)

# Run chosen module function
navigation_pages[selected_page]()
