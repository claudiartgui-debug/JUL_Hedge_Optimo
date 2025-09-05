import sys 
import os 
import streamlit as st
import pandas as pd
import plotly.express as px
from hedge_optimo_excel import margen_ppa, resumen_stats, plot_hist, scatter_plot, stats_graphic, margen_ppa_withoutChilca2
import matplotlib.pyplot as plt
import numpy as np
import requests
import io

# ==== CONFIGURACIÓN ===== # 

st.set_page_config(layout="wide")

# inputs
name = "https://raw.githubusercontent.com/claudiartgui-debug/hedge-optimo/main/hedge_github_risk.xlsx"
name1 = "https://raw.githubusercontent.com/claudiartgui-debug/hedge-optimo/main/hedge_github_margin_aset.parquet"

fecha = "JUL-25"
PPA_price = list(np.arange(38, 42.5, 0.25).round(2))

# tablas inputs 
df_margen, df_cmg, df_margenPPA, df_margenPPA_annually = margen_ppa(name, PPA_price)
df_margen_chilca2, df_cmg_chilca2 , df_margenPPA_chilca2, df_margenPPA_annually_chilca2 = margen_ppa_withoutChilca2(name, name1 , PPA_price)

columns_mw = df_margenPPA_annually.columns.tolist()[7:]
# redondeo
for df in [df_margenPPA_annually , df_margenPPA_annually_chilca2]:
    for col in df.columns[5:]:
        df[col] = df[col].round(1)

# =========== FILTROS ========== #
st.sidebar.title("Summary")

years = sorted(df_margenPPA_annually['year'].unique())
year = st.sidebar.selectbox("Year", years, index=0)

ppa_prices = sorted(df_margenPPA_annually['ppa_price'].unique())
ppa_price  = st.sidebar.selectbox("PPA price ($/MWh)", ppa_prices, index=0)

scenarios = [ 'ALL', 'Normal' ,'Without Chilca1']
scenario = st.sidebar.selectbox("Scenario", scenarios, index=0)

dict_sens  = {
    'Normal': ['base', 'high', 'low'],
    'Without Chilca1': ['basewithoutchilca1', 'highwithoutchilca1', 'lowwithoutchilca1'],
    'ALL': ['basewithoutchilca1', 'highwithoutchilca1', 'lowwithoutchilca1','base', 'high', 'low' ]
    }

scenario_list = dict_sens[scenario]

# margin and cmg
temps = {}
for name, df in {
    "margin_EEP_month": df_margenPPA,
    "margin_withoutChilca2_month": df_margenPPA_chilca2
}.items():
    df_temp = df[
        (df['year'] == year) & (df['ppa_price'] == ppa_price) & (df['scenario'].isin(scenario_list))
    ][['simulation', 'scenario', 'sample', 'year', 'month', 'margin_EEP_musd', 'spot_price', 'ppa_price']].rename(
        columns={
            "margin_EEP_musd": "Margin EEP (MUSD)",
            "spot_price": "Spot Price ($/MWh)"})
    temps[name] = df_temp



for name, df in {
    "margin_EEP_anual": df_margenPPA_annually,
    "margin_withoutChilca2_anual": df_margenPPA_annually_chilca2
}.items():
    df_temp = df[
        (df['year'] == year) & (df['ppa_price'] == ppa_price) & (df['scenario'].isin(scenario_list))
    ][['simulation', 'scenario', 'sample', 'year', 'margin_EEP_musd', 'spot_price', 'ppa_price']].rename(
        columns={
            "margin_EEP_musd": "Margin EEP (MUSD)",
            "spot_price": "Spot Price ($/MWh)"})
    temps[name] = df_temp

# statistics
for name, df in {
    "margin_EEP_anual_stats": df_margenPPA_annually,
    "margin_withoutChilca2_anual_stats": df_margenPPA_annually_chilca2
}.items():
    df_temp = df[
        (df['scenario'].isin(scenario_list))]
    temps[name] = df_temp

df_stats = resumen_stats(temps['margin_EEP_anual_stats'], columns_mw , PPA_price )	
df_stats_chilca2 = resumen_stats(temps['margin_withoutChilca2_anual_stats'], columns_mw , PPA_price)	

for df in [df_stats , df_stats_chilca2]:
    for col in df.columns[1:-2]:
        df[col] = df[col].round(1)
        
for name, df in {
    'stats_EEP':df_stats,
    'stats_withoutChilca2':df_stats_chilca2
}.items():
    df = df[(df['year'] == year) & (df['ppa_price'] == ppa_price)][[
    'capacity', 'At Risk@P90', 'Mean', 'SD', 'P5', 'P95',
    'Downside', 'Upside', 'Min', 'Max', 'P95-Min',
    'P5-P95', 'Max-P5', 'Spread', 'ppa_price', 'year']]
    temps[name] = df

# ========== ALL ASSET EEP ========== #
col1 = 'Spot Price ($/MWh)'
col2 = 'Margin EEP (MUSD)'

st.title(f"Hedge Optimo - {fecha}")
colA, colB = st.columns(2)
with colA:
    st.title(f"ALL asset EEP")
    
    fig1 = plot_hist( temps['margin_EEP_anual'], col1, year,"Spot Price ($/MWh)", ppa_price)
    st.pyplot(fig1)

    fig2 = plot_hist(temps['margin_EEP_anual'], col2, year, "Margin EEP (MUSD)",ppa_price)
    st.pyplot(fig2)

    fig3 = scatter_plot(temps['margin_EEP_anual'], col1, col2 , year, ppa_price )
    st.pyplot(fig3)

    st.subheader("CMG ($/MWh) and S-P (MUSD)")
    st.dataframe(temps["margin_EEP_month"])

    fig4 = stats_graphic(temps['stats_EEP'], "At Risk@P90", year, ppa_price)
    st.pyplot(fig4)

    st.subheader("Statistics: ")
    st.dataframe(temps['stats_EEP'])


# ========== ALL ASSET EEP without Chilca 2 after Jan 27========== #
with colB:
        
    st.title(f"without Chilca2")

    fig5 = plot_hist( temps['margin_withoutChilca2_anual'], col1, year,"Spot Price ($/MWh)", ppa_price)
    st.pyplot(fig5)

    fig6 = plot_hist(temps['margin_withoutChilca2_anual'], col2, year, "Margin EEP (MUSD)",ppa_price)
    st.pyplot(fig6)

    fig7 = scatter_plot(temps['margin_withoutChilca2_anual'], col1, col2 , year, ppa_price)
    st.pyplot(fig7)

    st.subheader("CMG ($/MWh) and S-P (MUSD)")
    st.dataframe(temps["margin_withoutChilca2_month"])

    fig8 = stats_graphic(temps['stats_withoutChilca2'], "At Risk@P90", year, ppa_price )
    st.pyplot(fig8)

    st.subheader("Statistics:")
    st.dataframe(temps['stats_withoutChilca2'])


#cd "C:\Users\ZJ6638\OneDrive - ENGIE\MC\Streamlit\hedge-optimo-main"
#python -m streamlit run main_hedge_excel.py



# 38 - 42.5  : steps = 0.25










