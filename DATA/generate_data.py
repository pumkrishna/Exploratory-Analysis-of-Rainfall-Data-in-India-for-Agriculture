import pandas as pd
import numpy as np

np.random.seed(42)

regions = {
    'North India': {'base': 800, 'std': 200, 'states': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Himachal Pradesh']},
    'South India': {'base': 1200, 'std': 300, 'states': ['Tamil Nadu', 'Karnataka', 'Kerala', 'Andhra Pradesh']},
    'East India': {'base': 1500, 'std': 400, 'states': ['West Bengal', 'Odisha', 'Bihar', 'Jharkhand']},
    'West India': {'base': 600, 'std': 250, 'states': ['Rajasthan', 'Gujarat', 'Maharashtra', 'Goa']},
    'Central India': {'base': 900, 'std': 220, 'states': ['Madhya Pradesh', 'Chhattisgarh']},
    'Northeast India': {'base': 2000, 'std': 500, 'states': ['Assam', 'Meghalaya', 'Arunachal Pradesh', 'Manipur']},
}

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Monthly distribution weights (monsoon-heavy)
month_weights = [0.02, 0.02, 0.03, 0.04, 0.05, 0.12, 0.20, 0.22, 0.15, 0.08, 0.04, 0.03]

years = list(range(1990, 2024))

records = []
for region, info in regions.items():
    for state in info['states']:
        for year in years:
            # Add climate trend (slight increase in variability over years)
            trend_factor = 1 + (year - 1990) * 0.002
            annual_rain = np.random.normal(info['base'] * trend_factor, info['std'])
            annual_rain = max(annual_rain, 100)
            
            monthly_rains = np.random.dirichlet(np.array(month_weights) * 10) * annual_rain
            
            row = {
                'Year': year,
                'Region': region,
                'State': state,
                'Annual_Rainfall': round(annual_rain, 2),
            }
            for i, month in enumerate(months):
                row[month] = round(monthly_rains[i], 2)
            
            # Crop yield index (correlated with rainfall + noise)
            optimal_rain = info['base']
            deviation = abs(annual_rain - optimal_rain) / optimal_rain
            crop_yield = max(0.3, 1.0 - deviation * 0.8 + np.random.normal(0, 0.1))
            row['Crop_Yield_Index'] = round(crop_yield, 3)
            
            # Drought/flood flags
            row['Drought_Year'] = 1 if annual_rain < info['base'] * 0.75 else 0
            row['Flood_Year'] = 1 if annual_rain > info['base'] * 1.35 else 0
            
            records.append(row)

df = pd.DataFrame(records)
df.to_csv('/home/claude/rainfall_india/data/india_rainfall.csv', index=False)
print(f"Dataset created: {df.shape[0]} records, {df.shape[1]} columns")
print(df.head())
