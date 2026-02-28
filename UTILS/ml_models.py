"""
Machine Learning Models for Rainfall Analysis and Prediction
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score
import os, pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'charts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    return f'/static/charts/{name}'


def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'india_rainfall.csv'))
    return df


def prepare_features(df):
    le = LabelEncoder()
    df = df.copy()
    df['Region_enc'] = le.fit_transform(df['Region'])
    df['State_enc'] = le.fit_transform(df['State'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    features = ['Year', 'Region_enc', 'State_enc'] + months[:6]  # Use first 6 months to predict annual
    return df, features


def train_regression_models():
    df = load_data()
    df, features = prepare_features(df)
    
    X = df[features]
    y = df['Annual_Rainfall']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        if name in ['Linear Regression', 'Ridge Regression']:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 4)}
        predictions[name] = (y_test, y_pred)
    
    # Save best model
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'features': features,
                     'name': best_model_name}, f)
    
    return results, predictions, best_model_name


def plot_model_comparison(results):
    names = list(results.keys())
    r2_vals = [results[n]['R2'] for n in names]
    rmse_vals = [results[n]['RMSE'] for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    bars = axes[0].bar(names, r2_vals, color=colors)
    axes[0].set_title('Model Comparison – R² Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R² Score'); axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    bars2 = axes[1].bar(names, rmse_vals, color=colors)
    axes[1].set_title('Model Comparison – RMSE', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE (mm)'); axes[1].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars2, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return save_fig('model_comparison.png')


def plot_predictions(predictions, best_name):
    y_test, y_pred = predictions[best_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='#1a6b3c', s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[0].set_title(f'{best_name} – Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Rainfall (mm)'); axes[0].set_ylabel('Predicted Rainfall (mm)')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    residuals = np.array(y_test) - np.array(y_pred)
    axes[1].hist(residuals, bins=30, color='#2980b9', alpha=0.75, edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residuals (mm)'); axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return save_fig('predictions.png')


def plot_feature_importance():
    try:
        with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        features = data['features']
        
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(features)))
        ax.bar(range(len(features)), importances[indices], color=colors)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([features[i] for i in indices], rotation=40, ha='right')
        ax.set_title('Feature Importance – Random Forest / Gradient Boosting', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score'); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        return save_fig('feature_importance.png')
    except Exception as e:
        return None


def train_drought_classifier():
    df = load_data()
    le = LabelEncoder()
    df = df.copy()
    df['Region_enc'] = le.fit_transform(df['Region'])
    df['State_enc'] = le.fit_transform(df['State'])
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    features = ['Year', 'Region_enc', 'State_enc'] + months
    
    X = df[features]
    y = df['Drought_Year']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Drought'], output_dict=True)
    
    return round(acc, 4), report


def run_all_ml():
    results, predictions, best_name = train_regression_models()
    chart1 = plot_model_comparison(results)
    chart2 = plot_predictions(predictions, best_name)
    chart3 = plot_feature_importance()
    drought_acc, drought_report = train_drought_classifier()
    
    return {
        'regression_results': results,
        'best_model': best_name,
        'drought_accuracy': drought_acc,
        'drought_report': drought_report,
        'charts': {'comparison': chart1, 'predictions': chart2, 'importance': chart3}
    }
