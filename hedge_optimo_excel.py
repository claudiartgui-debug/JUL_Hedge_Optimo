import sys 
import os 
import pandas as pd 
import calendar
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import streamlit as st 
import requests
import io

plt.rcParams.update({
    "axes.titlesize": 14,     
    "axes.titleweight": "bold",
    "axes.labelsize": 10, 
    "grid.linestyle": "--",
    "grid.alpha": 0.3
})

# ==== FUNCTIONS ===== #
# 38 - 42.5  : steps = 0.25
@st.cache_data
def margen_ppa(name: str , PPA_price: list) -> pd.DataFrame:
    cost = 1.5 
    df_list = []
    df_margen = pd.read_excel(name, sheet_name='margen_EEP', dtype={'sample': str}).rename(columns={"value_musd": "margin_EEP_musd"})
    df_cmg = pd.read_excel(name, sheet_name='CMG', dtype={'sample': str}).rename(columns={"precio_medio": "spot_price"})
    df_margenPPA = df_margen.copy()[['simulation', 'scenario', 'sample', 'year', 'month']]
    df_margenPPA ["days_in_month"] = df_margenPPA.apply(lambda x: calendar.monthrange(x["year"], x["month"])[1], axis=1)

    for df in [df_margen, df_cmg]:
        df_margenPPA = pd.merge(df_margenPPA, df, on=['simulation', 'scenario', 'sample', 'year', 'month'], how='inner')
    df_margenPPA = df_margenPPA[df_margenPPA['sample'] != "Mean"]
    
    for price in PPA_price:
        df_temp = df_margenPPA.copy()
        df_temp["ppa_price"] = price
        df_list.append(df_temp)
    
    df_margenPPA = pd.concat(df_list, ignore_index=True)
    
    capacidad = list(range(1,400,1))
    
    new_cols = {}
    for c in capacidad:
        new_cols[f'margen+PPA_{c}'] = (
            df_margenPPA['margin_EEP_musd'] 
            + (c * df_margenPPA['days_in_month'] * 24) 
            * (df_margenPPA['ppa_price'] - cost - df_margenPPA['spot_price']) / 1000000
        )
        
    df_margenPPA = pd.concat([df_margenPPA, pd.DataFrame(new_cols)], axis=1)
    agg_dict = {'margin_EEP_musd': 'sum' ,"spot_price":'mean'}
    agg_dict.update({f'margen+PPA_{c}': 'sum'  for c in capacidad })
    
    df_margenPPA_annually = df_margenPPA.groupby(
        ['simulation', 'scenario', 'sample', 'year','ppa_price'], as_index=False
    ).agg(agg_dict)
    
    return df_margen, df_cmg, df_margenPPA, df_margenPPA_annually

@st.cache_data
def margen_ppa_withoutChilca2(name: str , PPA_price: list) -> pd.DataFrame:
    cost = 1.5 
    df_list = []
    
    df_margen = pd.read_excel(name, sheet_name='margen_EEP_withoutChilca2', dtype={'sample': str}).rename(columns={"value_musd": "margin_EEP_musd"})
    df_margen["sample"] = df_margen["sample"].astype(str)
    
    df_margen = df_margen.groupby(['simulation', 'scenario', 'sample', 'year', 'month'], as_index=False)['margin_EEP_musd'].sum()
    df_cmg = pd.read_excel(name , sheet_name='CMG', dtype={'sample': str}).rename(columns={"precio_medio": "spot_price"})
    
    df_margenPPA = df_margen.copy()[['simulation', 'scenario', 'sample', 'year', 'month']]
    df_margenPPA ["days_in_month"] = df_margenPPA.apply(lambda x: calendar.monthrange(x["year"], x["month"])[1], axis=1)

    for df in [df_margen, df_cmg]:
        df_margenPPA = pd.merge(df_margenPPA, df, on=['simulation', 'scenario', 'sample', 'year', 'month'], how='inner')
        
    df_margenPPA = df_margenPPA[df_margenPPA['sample'] != "Mean"]
    
    for price in PPA_price:
        df_temp = df_margenPPA.copy()
        df_temp["ppa_price"] = price
        df_list.append(df_temp)

    df_margenPPA = pd.concat(df_list, ignore_index=True)
    
    capacidad = list(range(1,400,1))
    
    new_cols = {}
    for c in capacidad:
        new_cols[f'margen+PPA_{c}'] = (
            df_margenPPA['margin_EEP_musd'] 
            + (c * df_margenPPA['days_in_month'] * 24) 
            * (df_margenPPA['ppa_price'] - cost - df_margenPPA['spot_price']) / 1000000)
        
    df_margenPPA = pd.concat([df_margenPPA, pd.DataFrame(new_cols)], axis=1)
    agg_dict = {'margin_EEP_musd': 'sum' ,"spot_price":'mean'}
    agg_dict.update({f'margen+PPA_{c}': 'sum'  for c in capacidad })
    
    df_margenPPA_annually = df_margenPPA.groupby(
        ['simulation', 'scenario', 'sample', 'year','ppa_price'], as_index=False
    ).agg(agg_dict)
    
    return df_margen, df_cmg , df_margenPPA, df_margenPPA_annually

@st.cache_data
def resumen_stats(df: pd.DataFrame, columns_mw: list , PPA_price:list ) -> pd.DataFrame:
    if not isinstance(PPA_price, (list, tuple, set)):
        PPA_price = [PPA_price]
    years = df['year'].unique()
    stats = {}
    for price in PPA_price:
        df_price = df[df['ppa_price'] == price]
        for y in years:
            df_year = df_price[df_price['year'] == y]

            for col_mw in columns_mw: 
                if col_mw not in df_year.columns:
                    print(f"⚠ Columna no encontrada -> {col_mw}")
                    continue

                serie = df_year[col_mw].dropna().values

                if len(serie) == 0:
                    print(f"⚠ Serie vacía -> price={price}, year={y}, col={col_mw}")
                    continue

                #median_val = np.median(serie)
                mean_val = np.mean(serie)
                std_val = np.std(serie)
                p5 = np.percentile(serie, 95)
                p10 = np.percentile(serie, 10)
                #p25 = np.percentile(serie, 25)
                p50 = np.percentile(serie, 50)
                #p75 = np.percentile(serie, 75)
                p90 = np.percentile(serie, 90)
                p95 = np.percentile(serie, 5)
                min_val = np.min(serie)
                max_val = np.max(serie)
                downside = np.mean(serie[serie < p10])
                upside = np.mean(serie[serie > p90])
                spread = upside - downside
                at_risk = p50 - downside
                stats[f"{price}_{y}_{col_mw}"] = {
                        'Mean': mean_val,
                        'SD': std_val,
                        'P5': p5,
                        'P95': p95,
                        'Min': min_val,
                        'Max': max_val,
                        'P95-Min': p95 - min_val,
                        'P5-P95': p95 - p5,
                        'Max-P5': max_val - p5,
                        'Downside': downside,
                        'Upside': upside,
                        'Spread': spread,
                        'At Risk@P90': at_risk
                    }
    df_stats = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'capacity'})
    df_stats[['ppa_price','year','name', 'capacity']] = df_stats['capacity'].str.split('_', n=3, expand=True)
    df_stats.drop(columns=['name'], inplace=True)
    df_stats['capacity'] = pd.to_numeric(df_stats['capacity'], errors='coerce')
    df_stats['year'] = pd.to_numeric(df_stats['year'], errors='coerce')
    df_stats['ppa_price'] = pd.to_numeric(df_stats['ppa_price'], errors='coerce')
    return df_stats

def plot_hist(df, columna, year, tittle, PPA_price):
    df['year'] = df['year'].astype(int)
    df = df[df['ppa_price'] == PPA_price]
    df = df[df['year'] == year]
    data = df[columna].dropna().values
    bins = 10
    counts, bin_edges = np.histogram(data, bins=bins)

    fig, ax = plt.subplots(figsize= (10,6)) 
    bars = ax.bar(range(len(counts)), counts, width=0.95, align='center')

    labels = [f"[{int(bin_edges[i])}, {int(bin_edges[i+1])}]" for i in range(len(bin_edges)-1)]
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, count, str(count),
                    ha='center', va='bottom', fontsize=10) 

    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"{tittle} - {year}", fontsize=14)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.2)
    plt.tight_layout()
    
    return fig

def scatter_plot(df: pd.DataFrame, col1: str, col2: str, year: int , PPA_price: float) -> plt.Figure:
    df = df[df['year'] == year]
    df = df[df['ppa_price'] == PPA_price]
    col1 = f'{col1}'
    col2 = f'{col2}'

    fig, ax = plt.subplots(figsize=(10, 6))  

    ax.scatter(df[col1], df[col2], alpha=0.6, color='teal')
    x = df[col1].dropna().values
    y = df[col2].dropna().values

    if len(x) > 1: 
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='blue', alpha=0.6, linewidth=1, label='Tendencia')
        ax.text(
            x.min(), y.max(),
            f'y = {m:.2f}x + {b:.2f}', fontsize=12, color='blue', ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    if (col2 == 'margin_EEP_musd') & (col1 == 'spot_price'):
        ax.set_xlabel('Spot Price (USD/MWh)')
        ax.set_ylabel('Margin EEP (MUSD)')
        ax.set_title(f'Spot Price vs Margin EEP - {year}')
    else:
        ax.set_xlabel(col1, fontsize=12)
        ax.set_ylabel(col2, fontsize=12)
        ax.set_title(f'{col1} ($/MW) vs {col2} - {year}', fontsize=14, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.2)

    fig.tight_layout()
    return fig   

def stats_graphic(df_stats: pd.DataFrame, col: str, year: int, PPA_price: float) -> plt.Figure: 
    df_stats = df_stats[(df_stats['ppa_price'] == PPA_price) & (df_stats['year'] == year)] 
    df_stats = df_stats[['capacity', col]]

    fig, ax = plt.subplots(figsize=(8,5))  

    x = df_stats["capacity"].values
    y = df_stats[col].values

    if len(x) > 1: 
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        ax.plot(x_smooth, y_smooth, color='teal', linewidth=2)
    else:
        ax.plot(x, y, 'o-', color='teal')

    ax.set_title("S-P at Risk", fontweight='bold')
    ax.set_xlabel("Capacity (MW)")
    ax.set_ylabel("S-P (MUSD)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.2)

    if len(y) > 0:
        ax.set_ylim(0, max(y)*1.1) 

    ax.legend(["S-P at Risk (ES@P90)"], loc="lower right")
    fig.tight_layout()
    return fig











