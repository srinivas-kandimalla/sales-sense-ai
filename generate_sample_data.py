# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

random.seed(42)
np.random.seed(42)

os.makedirs('data', exist_ok=True)

start = datetime(2022, 1, 1)
end   = datetime(2024, 12, 31)
dates = pd.date_range(start, end, freq='D')

products = {
    'SKU-0821': ('Electronics', 200),
    'SKU-1134': ('FMCG',         50),
    'SKU-0562': ('Apparel',     200),
    'SKU-2201': ('Home',        400),
    'SKU-3305': ('Electronics', 350),
    'SKU-4412': ('FMCG',         30),
    'SKU-5501': ('Apparel',     150),
    'SKU-6623': ('Home',        250),
    'SKU-7734': ('Electronics', 500),
    'SKU-8891': ('FMCG',         80),
}

rows = []
for date in dates:
    month = date.month
    dow   = date.weekday()

    season  = {1:0.8,2:0.82,3:0.9,4:0.95,5:1.0,6:1.05,
               7:1.0,8:0.98,9:1.05,10:1.3,11:1.5,12:1.4}[month]
    weekend = 1.15 if dow >= 5 else 1.0
    diwali  = 1.4  if (month == 10 and 15 <= date.day <= 25) else 1.0
    yearend = 1.3  if (month == 12 and date.day >= 20)       else 1.0

    for sku, (cat, base_price) in products.items():
        promo       = 1 if random.random() < 0.12 else 0
        promo_boost = 1.25 if promo else 1.0

        base_qty = {'Electronics':30,'FMCG':200,'Apparel':60,'Home':25}[cat]
        noise    = np.random.normal(1.0, 0.12)
        qty      = max(1, int(base_qty * season * weekend * diwali
                              * yearend * promo_boost * noise))

        price_var = base_price * np.random.uniform(0.95, 1.05)
        discount  = 0.85 if promo else 1.0
        price     = round(price_var * discount, 2)
        revenue   = round(qty * price, 2)

        inv_base  = {'Electronics':300,'FMCG':1500,
                     'Apparel':500,'Home':200}[cat]
        inventory = max(10, int(inv_base * np.random.uniform(0.6, 2.0)))

        rows.append({
            'date':           date.strftime('%Y-%m-%d'),
            'product_id':     sku,
            'category':       cat,
            'sales_qty':      qty,
            'revenue':        revenue,
            'price':          price,
            'inventory_qty':  inventory,
            'promotion_flag': promo
        })

df = pd.DataFrame(rows)

# Inject realistic nulls
null_sales = np.random.choice(df.index, size=int(len(df)*0.011), replace=False)
null_price = np.random.choice(df.index, size=int(len(df)*0.008), replace=False)
null_inv   = np.random.choice(df.index, size=int(len(df)*0.006), replace=False)
df.loc[null_sales, 'sales_qty']     = np.nan
df.loc[null_price, 'price']         = np.nan
df.loc[null_inv,   'inventory_qty'] = np.nan

# Inject outliers for preprocessing demo
outlier_idx = np.random.choice(df.index, size=30, replace=False)
df.loc[outlier_idx, 'sales_qty'] = (
    df.loc[outlier_idx, 'sales_qty'] * np.random.uniform(4, 7, 30)
)

df = df.sort_values(['date','product_id']).reset_index(drop=True)
df.to_csv('data/sample_sales_data.csv', index=False)
print(f"Generated: {len(df):,} rows | "
      f"Nulls -> sales_qty:{df['sales_qty'].isna().sum()} "
      f"price:{df['price'].isna().sum()} "
      f"inventory_qty:{df['inventory_qty'].isna().sum()}")
