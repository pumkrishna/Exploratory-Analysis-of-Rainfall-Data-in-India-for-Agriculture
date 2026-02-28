"""
Exploratory Data Analysis Utilities for India Rainfall Data
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    'primary': '#1a6b3c',
    'secondary': '#2980b9',
    'accent': '#e67e22',
    'danger': '#c0392b',
    'light': '#ecf0f1',
}

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    return f'/static/charts/{name}'


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(base, 'data', 'india_rainfall.csv'))
    return df


def plot_annual_trend():
    df = load_data()
    yearly = df.groupby('Year')['Annual_Rainfall'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(yearly['Year'],
                    yearly['mean'] - yearly['std'],
                    yearly['mean'] + yearly['std'],
                    alpha=0.2, color=PALETTE['secondary'], label='±1 Std Dev')
    ax.plot(yearly['Year'], yearly['mean'], color=PALETTE['primary'], linewidth=2.5, marker='o', markersize=4, label='Mean Annual Rainfall')
    
    # Trend line
    z = np.polyfit(yearly['Year'], yearly['mean'], 1)
    p = np.poly1d(z)
    ax.plot(yearly['Year'], p(yearly['Year']), '--', color=PALETTE['accent'], linewidth=1.5, label='Trend')
    
    ax.set_title('Annual Rainfall Trend Across India (1990–2023)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Rainfall (mm)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return save_fig('annual_trend.png')


def plot_regional_comparison():
    df = load_data()
    region_data = df.groupby('Region')['Annual_Rainfall'].mean().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(region_data)))
    axes[0].barh(region_data.index, region_data.values, color=colors)
    axes[0].set_title('Average Annual Rainfall by Region', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rainfall (mm)')
    for i, v in enumerate(region_data.values):
        axes[0].text(v + 10, i, f'{v:.0f}mm', va='center', fontsize=9)
    
    df_box = df[['Region', 'Annual_Rainfall']]
    regions_order = region_data.index.tolist()
    data_by_region = [df_box[df_box['Region'] == r]['Annual_Rainfall'].values for r in regions_order]
    bp = axes[1].boxplot(data_by_region, labels=[r.replace(' India', '') for r in regions_order],
                         patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title('Rainfall Distribution by Region', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Rainfall (mm)')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return save_fig('regional_comparison.png')


def plot_monthly_pattern():
    df = load_data()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    regions = df['Region'].unique()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    
    for region, color in zip(regions, colors):
        subset = df[df['Region'] == region][months].mean()
        ax.plot(months, subset.values, marker='o', linewidth=2, label=region, color=color)
    
    ax.axvspan(5, 8, alpha=0.08, color='blue', label='Monsoon Season')
    ax.set_title('Monthly Rainfall Pattern by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Average Rainfall (mm)')
    ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    return save_fig('monthly_pattern.png')


def plot_heatmap():
    df = load_data()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = df.groupby('Region')[months].mean()
    pivot.index = [r.replace(' India', '') for r in pivot.index]
    
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrBr',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Rainfall (mm)'})
    ax.set_title('Monthly Rainfall Heatmap by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Region')
    plt.tight_layout()
    return save_fig('heatmap.png')


def plot_drought_flood():
    df = load_data()
    yearly = df.groupby('Year').agg(
        droughts=('Drought_Year', 'sum'),
        floods=('Flood_Year', 'sum')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.35
    x = np.arange(len(yearly))
    ax.bar(x - width/2, yearly['droughts'], width, label='Drought Events', color=PALETTE['accent'], alpha=0.85)
    ax.bar(x + width/2, yearly['floods'], width, label='Flood Events', color=PALETTE['secondary'], alpha=0.85)
    ax.set_xticks(x[::3]); ax.set_xticklabels(yearly['Year'].iloc[::3], rotation=45)
    ax.set_title('Drought and Flood Events Across Regions (1990–2023)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Number of States/Regions Affected')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    return save_fig('drought_flood.png')


def plot_crop_correlation():
    df = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].scatter(df['Annual_Rainfall'], df['Crop_Yield_Index'],
                    alpha=0.4, color=PALETTE['primary'], s=20)
    z = np.polyfit(df['Annual_Rainfall'], df['Crop_Yield_Index'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(df['Annual_Rainfall'].min(), df['Annual_Rainfall'].max(), 300)
    axes[0].plot(x_line, p(x_line), color=PALETTE['accent'], linewidth=2, label='Poly fit')
    corr = df['Annual_Rainfall'].corr(df['Crop_Yield_Index'])
    axes[0].set_title(f'Rainfall vs Crop Yield Index (r={corr:.2f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Annual Rainfall (mm)'); axes[0].set_ylabel('Crop Yield Index')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    region_corr = df.groupby('Region').apply(
        lambda x: x['Annual_Rainfall'].corr(x['Crop_Yield_Index'])
    ).sort_values()
    colors = [PALETTE['danger'] if v < 0 else PALETTE['primary'] for v in region_corr.values]
    axes[1].barh(region_corr.index, region_corr.values, color=colors)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title('Correlation: Rainfall vs Yield by Region', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pearson Correlation'); axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    return save_fig('crop_correlation.png')


def plot_state_heatmap():
    df = load_data()
    pivot = df.groupby(['State', 'Year'])['Annual_Rainfall'].mean().unstack()
    pivot = pivot.iloc[:, ::5]  # Every 5 years for readability
    
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(pivot, cmap='Blues', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Rainfall (mm)'}, annot=False)
    ax.set_title('State-wise Annual Rainfall Over Years', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('State')
    plt.tight_layout()
    return save_fig('state_heatmap.png')


def get_summary_stats():
    df = load_data()
    return {
        'total_records': len(df),
        'years': f"{df['Year'].min()} – {df['Year'].max()}",
        'states': df['State'].nunique(),
        'regions': df['Region'].nunique(),
        'avg_rainfall': round(df['Annual_Rainfall'].mean(), 1),
        'max_rainfall': round(df['Annual_Rainfall'].max(), 1),
        'min_rainfall': round(df['Annual_Rainfall'].min(), 1),
        'drought_events': int(df['Drought_Year'].sum()),
        'flood_events': int(df['Flood_Year'].sum()),
        'avg_yield_index': round(df['Crop_Yield_Index'].mean(), 3),
    }
