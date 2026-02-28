# 🌧 India Rainfall Analysis for Agriculture

An end-to-end data science web application for **Exploratory Data Analysis (EDA)** and **Machine Learning** on India's historical rainfall data (1990–2023), built with Python, Flask, and Scikit-learn.

---

## 📁 Project Structure

```
rainfall_india/
├── app.py                  # Flask web application (main entry point)
├── requirements.txt        # Python dependencies
├── data/
│   ├── generate_data.py    # Dataset generation script
│   └── india_rainfall.csv  # Generated dataset (748 records × 19 cols)
├── utils/
│   ├── eda.py              # EDA & visualization functions
│   └── ml_models.py        # ML training, evaluation, and prediction
├── models/
│   └── best_model.pkl      # Saved best-performing model (auto-generated)
├── templates/
│   ├── base.html           # Shared navbar & layout
│   ├── index.html          # Dashboard with KPI stats
│   ├── eda.html            # Full EDA charts page
│   ├── ml.html             # ML results + live predictor
│   └── data.html           # Dataset browser + data dictionary
└── static/
    └── charts/             # Auto-generated chart PNGs
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
```bash
python data/generate_data.py
```

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
```
http://localhost:5000
```

---

## 📊 Features

### Dashboard
- Key statistics: total records, years covered, states/regions, avg rainfall, drought/flood events
- Annual trend chart with variability band
- Regional comparison & crop yield correlation

### EDA Charts
- **Annual Trend** – Yearly average with ±1 std deviation and regression trendline
- **Monthly Pattern** – Region-wise seasonal curves highlighting monsoon peak
- **Heatmaps** – Monthly rainfall by region; state-wise rainfall over decades
- **Drought & Flood Events** – Bar chart of extreme years by frequency
- **Crop Yield Correlation** – Scatter plots and regional correlation bars

### Machine Learning
| Task | Models Used | Target |
|------|-------------|--------|
| Rainfall Prediction | Linear Regression, Ridge, Random Forest, Gradient Boosting | Annual Rainfall (mm) |
| Drought Classification | Decision Tree | Drought Year (0/1) |

- Model comparison by R², RMSE, MAE
- Feature importance visualization
- **Live API predictor** (`/api/predict`) accepts early-season data and returns annual forecast

### Data Explorer
- Statistical summary (count, mean, std, percentiles)
- Paginated data table (first 50 records)
- Full data dictionary

---

## 🌾 Use Cases

| Scenario | Beneficiary | How |
|----------|------------|-----|
| Crop Planning | Farmers | Select crops/planting dates based on regional rainfall patterns |
| Irrigation Management | Agricultural Engineers | Optimize water scheduling using monsoon variability analysis |
| Risk Assessment | Policymakers & Insurers | Use drought/flood predictions for insurance and disaster planning |

---

## 🛠 Technical Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web Framework | Flask 2.x |
| Data Analysis | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Frontend | Jinja2 templates, Vanilla JS |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/eda` | GET | All EDA charts |
| `/ml` | GET | ML results |
| `/data` | GET | Dataset explorer |
| `/api/region/<region>` | GET | Region-wise yearly data (JSON) |
| `/api/predict` | POST | Predict annual rainfall from JSON body |

### Predict API Example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"year":2024,"region":"South India","state":"Kerala","Jan":10,"Feb":15,"Mar":20,"Apr":50,"May":100,"Jun":250}'
```

---

## 📜 Dataset Description

**748 records** — 18 Indian states × 34 years (1990–2023)

| Column | Type | Description |
|--------|------|-------------|
| Year | int | Calendar year |
| Region | str | North/South/East/West/Central/Northeast India |
| State | str | Indian state name |
| Annual_Rainfall | float | Total annual rainfall (mm) |
| Jan–Dec | float | Monthly rainfall (mm) |
| Crop_Yield_Index | float | Normalized crop yield (0–1+) |
| Drought_Year | bool | 1 if rainfall < 75% of regional normal |
| Flood_Year | bool | 1 if rainfall > 135% of regional normal |
