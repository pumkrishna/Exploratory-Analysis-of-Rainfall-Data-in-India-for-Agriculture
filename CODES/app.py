"""
Rainfall Data Analysis Web Application - Flask Backend
"""
from flask import Flask, render_template, jsonify, request
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.eda import (plot_annual_trend, plot_regional_comparison, plot_monthly_pattern,
                       plot_heatmap, plot_drought_flood, plot_crop_correlation,
                       plot_state_heatmap, get_summary_stats, load_data)
from utils.ml_models import run_all_ml

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

_ml_cache = None
_eda_charts = None


def get_eda_charts():
    global _eda_charts
    if _eda_charts is None:
        _eda_charts = {
            'annual_trend': plot_annual_trend(),
            'regional': plot_regional_comparison(),
            'monthly': plot_monthly_pattern(),
            'heatmap': plot_heatmap(),
            'drought_flood': plot_drought_flood(),
            'crop_corr': plot_crop_correlation(),
            'state_heatmap': plot_state_heatmap(),
        }
    return _eda_charts


def get_ml_results():
    global _ml_cache
    if _ml_cache is None:
        _ml_cache = run_all_ml()
    return _ml_cache


@app.route('/')
def index():
    stats = get_summary_stats()
    charts = get_eda_charts()
    return render_template('index.html', stats=stats, charts=charts)


@app.route('/eda')
def eda():
    charts = get_eda_charts()
    return render_template('eda.html', charts=charts)


@app.route('/ml')
def ml():
    ml_results = get_ml_results()
    return render_template('ml.html', ml=ml_results)


@app.route('/data')
def data_view():
    df = load_data()
    df_sample = df.head(50)
    columns = df_sample.columns.tolist()
    rows = df_sample.values.tolist()
    stats = df.describe().round(2).to_dict()
    return render_template('data.html', columns=columns, rows=rows, stats=stats, total=len(df))


@app.route('/api/region/<region>')
def api_region(region):
    df = load_data()
    subset = df[df['Region'] == region]
    yearly = subset.groupby('Year')['Annual_Rainfall'].mean().reset_index()
    return jsonify({
        'years': yearly['Year'].tolist(),
        'rainfall': yearly['Annual_Rainfall'].tolist(),
        'avg': round(subset['Annual_Rainfall'].mean(), 2),
        'states': subset['State'].unique().tolist()
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        import pickle
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON body received', 'status': 'error'})

        # Validate required fields
        required = ['year', 'region', 'state']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'})

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
        if not os.path.exists(model_path):
            # Re-train if model file missing
            from utils.ml_models import run_all_ml
            run_all_ml()

        with open(model_path, 'rb') as f:
            saved = pickle.load(f)

        df = load_data()

        # Fit encoders on same data as training
        le_region = LabelEncoder().fit(df['Region'])
        le_state  = LabelEncoder().fit(df['State'])

        # Validate region/state values
        if data['region'] not in le_region.classes_:
            return jsonify({'error': f"Unknown region: '{data['region']}'. Valid: {list(le_region.classes_)}", 'status': 'error'})
        if data['state'] not in le_state.classes_:
            return jsonify({'error': f"Unknown state: '{data['state']}'. Valid: {list(le_state.classes_)}", 'status': 'error'})

        region_enc = int(le_region.transform([data['region']])[0])
        state_enc  = int(le_state.transform([data['state']])[0])

        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        feature_cols = saved['features']  # ['Year', 'Region_enc', 'State_enc', 'Jan'...'Jun']

        row = {
            'Year':       int(data.get('year', 2024)),
            'Region_enc': region_enc,
            'State_enc':  state_enc,
        }
        for m in month_cols:
            row[m] = float(data.get(m, 0))

        # Build DataFrame with exact column names model was trained on
        X = pd.DataFrame([row])[feature_cols]

        model = saved['model']
        prediction = float(model.predict(X)[0])

        # Drought/flood flag based on regional average
        region_avg = df[df['Region'] == data['region']]['Annual_Rainfall'].mean()
        risk = 'Normal'
        if prediction < region_avg * 0.75:
            risk = 'Drought Risk 🟠'
        elif prediction > region_avg * 1.35:
            risk = 'Flood Risk 🔵'

        return jsonify({
            'predicted_rainfall': round(prediction, 2),
            'regional_average':   round(region_avg, 2),
            'risk_assessment':    risk,
            'model_used':         saved['name'],
            'status': 'success'
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc(), 'status': 'error'})


if __name__ == '__main__':
    print("Pre-generating charts and training models...")
    get_eda_charts()
    get_ml_results()
    print("Ready! Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
