# SalesSense

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sales-sense-ai-4yhu.onrender.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Active-green?style=for-the-badge&logo=google-chrome&logoColor=white)](https://sales-sense-ai-4yhu.onrender.com)

**Live Demo URL:** [https://sales-sense-ai-4yhu.onrender.com](https://sales-sense-ai-4yhu.onrender.com)

## Problem Statement
Traditional inventory management systems struggle to predict consumer demand fluctuations, resulting in expensive stockouts or wasteful capital tied up in excess inventory. SalesSense addresses this by combining advanced machine learning forecasts with a dynamic, safety-stock-driven inventory optimization workflow to streamline operations.

## End-to-End Workflow

This flowchart outlines how raw transaction data is ingested, processed, modeled, and transformed into actionable inventory optimization choices and live dashboard statuses:

```mermaid
graph TD
    A["📥 1. Data Ingestion (Upload CSV/XLSX)"] -->|Raw Sales Data| B["⚙️ 2. Data Preprocessing"]
    B -->|Cleaned Datasets| C["📈 3. Exploratory Data Analysis (EDA)"]
    B -->|Engineered ML Lags & LSTMs Features| D["🤖 4. Model Training Workbench"]
    D -->|Best-fit Model Object| E["🔮 5. Demand Forecasting Engine"]
    E -->|SKU Demand Projections & 95% CI| F["📦 6. Safety Stock & EOQ Optimization"]
    F -->|Optimized Reorder points & Stock alerts| H["📊 7. Executive Dashboard Monitoring"]
    F -->|Operations Log| G["📄 8. Automated PDF & Excel Reports"]
    G --> H
    
    style A fill:#1D9E75,stroke:#fff,stroke-width:2px,color:#fff
    style B fill:#EF9F27,stroke:#fff,stroke-width:2px,color:#fff
    style C fill:#AB63FA,stroke:#fff,stroke-width:2px,color:#fff
    style D fill:#FFA15A,stroke:#fff,stroke-width:2px,color:#fff
    style E fill:#19D3F3,stroke:#fff,stroke-width:2px,color:#fff
    style F fill:#E24B4A,stroke:#fff,stroke-width:2px,color:#fff
    style G fill:#FF6692,stroke:#fff,stroke-width:2px,color:#fff
    style H fill:#378ADD,stroke:#fff,stroke-width:2px,color:#fff
```

## Features
- **Dashboard**: Real-time business overview with high-level KPI cards, interactive sales trends, category distribution charts, and recent operational alerts.
- **Data Upload**: Seamless CSV/XLSX file ingestion with column mapping detection, comprehensive validation reporting, and data type checks.
- **Data Preprocessing**: Interactive multi-step wizard for handling missing values, identifying/capping outliers via the IQR method, automating feature engineering (lag, rolling, calendar, holiday indicators), and normalization.
- **EDA Analysis**: In-depth exploratory data analysis including summary stats, monthly trend breakdowns, correlation heatmaps, top-revenue SKUs, and year-over-year sales performance.
- **Model Training**: Custom machine learning pipeline featuring ARIMA, SARIMA, Prophet, XGBoost, Random Forest, and Deep Learning (LSTM-based) architectures, complete with hyperparameter tuning, model comparison charts, and metrics (MAPE, RMSE, MAE, R²).
- **Sales Forecasting**: Interactive future projection charts showing 95% confidence intervals, category forecasts, SKU-level filtering, and CSV download capabilities.
- **Inventory Optimization**: Automatic calculations for safety stock, reorder points, economic order quantity (EOQ), and days of stock remaining with color-coded warning systems and carrying cost savings logs.
- **Reports**: Quick one-click automated summaries exported as professionally styled PDF (via `fpdf2`) and Excel (via `openpyxl`) spreadsheets.

## Core Optimization Mathematics

SalesSense translates demand forecasts into mathematical inventory decisions using industry-standard Operations Research (OR) formulas:

### 1. Safety Stock ($SS$)
Protects the supply chain from lead-time demand variability:
$$SS = z \times \sigma_d \times \sqrt{L}$$
*Where:*
- $z$: Service level factor (1.65 for a 95% target fulfillment rate)
- $\sigma_d$: Standard deviation of daily sales volume demand
- $L$: Replenishment lead time in days

### 2. Reorder Point ($ROP$)
The inventory level that triggers a stock replenishment order:
$$ROP = (\mu_d \times L) + SS$$
*Where:*
- $\mu_d$: Average daily demand (forecasted volume)
- $L$: Replenishment lead time in days
- $SS$: Safety Stock buffer

### 3. Economic Order Quantity ($EOQ$)
The optimal order size that minimizes total holding and ordering costs:
$$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$
*Where:*
- $D$: Annual demand volume (forecasted run-rate)
- $S$: Order setup cost (shipping, handling, and procurement fees)
- $H$: Annual holding cost per unit (storage, insurance, and carrying costs)

## Tech Stack

| Component | Technology |
|---|---|
| Frontend / App framework | Streamlit |
| Data Manipulation | Pandas, Numpy |
| Machine Learning Models | Scikit-learn, XGBoost, Prophet, Statsmodels |
| Data Visualisation | Plotly |
| Export Formats | openpyxl (Excel), fpdf2 (PDF) |
| Holidays Integration | holidays |

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sales-sense-ai.git
   cd sales-sense-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Folder Structure

```text
sales-sense-ai/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── README.md
├── data/
│   └── sample_sales_data.csv       # Preloaded realistic sample data
├── modules/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── data_upload.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── forecasting.py
│   ├── inventory.py
│   └── reports.py
├── models/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── plot_utils.py
└── assets/
    └── style.css
```

## Project Screenshots

### 1. Dashboard Overview
![Dashboard](screenshots/01_dashboard_overview.png)

### 2. Data Upload Module
![Data Upload](screenshots/02_data_upload_module.png)

### 3. Data Preprocessing
![Data Preprocessing](screenshots/03_data_preprocessing.png)

### 4. Exploratory Data Analysis
![EDA](screenshots/04_exploratory_data_analysis.png)

### 5. Model Training Results
![Model Training](screenshots/05_model_training_results.png)

### 6. Sales Forecasting
![Sales Forecasting](screenshots/06_sales_forecasting.png)

### 7. Inventory Optimization
![Inventory](screenshots/07_inventory_optimization.png)

### 8. Report Export
![Report Export](screenshots/08_report_export.png)

### 9. Forecast PDF Report
![PDF Report](screenshots/09_monthly_forecast_pdf_report.png)

### 10. Forecast Excel Report
![Excel Report](screenshots/10_monthly_forecast_excel_report.png)

## 🚀 Future Roadmap & Enhancements
- [ ] **LLM AI Agent Assistant**: Chat directly with your sales data and forecast charts.
- [ ] **Multi-Warehouse Support**: Expand inventory calculations across multiple storage locations.
- [ ] **Real-Time Data Streams**: Connect directly to live e-commerce databases (e.g. Shopify, PostgreSQL).
- [ ] **Dockerization**: Complete containerization for the ML training server and Streamlit frontend.

## 🤝 Contributing & Support
Contributions, issues, and feature requests are welcome! Feel free to open a pull request or check the issues page.

**Show your support:** Give a ⭐️ if this project helped you!

## License
MIT License
