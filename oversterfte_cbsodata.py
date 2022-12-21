import pandas as pd
import cbsodata


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import platform
from oversterfte_helpers import *
import get_rioolwater
from streamlit import caching

# 70895ned = https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1659307527578
# Overledenen; geslacht en leeftijd, per week

# Downloaden van tabeloverzicht
# toc = pd.DataFrame(cbsodata.get_table_list())

# Downloaden van gehele tabel (kan een halve minuut duren)
@st.cache(ttl=60 * 60 * 24)
def get_sterftedata():
    if platform.processor() != "":
        # file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overledenen_cbs.csv"
        # data  = pd.read_csv(
        #     file,
        #     delimiter=",",
            
        #     low_memory=False,
        # )
        data = pd.DataFrame(cbsodata.get_data('70895ned'))
    else: 
        data = pd.DataFrame(cbsodata.get_data('70895ned'))

        # name_ = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overledenen_cbs.csv"
        # compression_opts = dict(method=None, archive_name=name_)
        # data.to_csv(name_, index=False, compression=compression_opts)
        #print("--- Saving " + name_ + " ---")

    data[['jaar','week_']] = data.Perioden.str.split(" week ",expand=True,)
    data['week_'] = data['week_']+" _"
    data['week_'] = data['week_'].replace(' dag','_',  regex=True)
    data['week_'] = data['week_'].replace('\) _',')',  regex=True)
    
    data[['week','aantal_dagen']] = data.week_.str.split(" ",expand=True,)
    data = data[data['week'].notnull()] #remove the year-totals
    data['aantal_dagen'] = data['aantal_dagen'].replace('_','7')
    data['aantal_dagen'] = data['aantal_dagen'].replace('\(','', regex=True)
    data['aantal_dagen'] = data['aantal_dagen'].replace('_\)','', regex=True)
    data = data[data['aantal_dagen'] == '7'] # only complete weeks
    data["weeknr"] = data["jaar"].astype(str) +"_" + data["week"].astype(str).str.zfill(2)
    import math
    data["week_int"]=data['week'].astype(int)
    #data["week_int"].apply(lambda x: float(x))
    data["virtuele_maand"] = ((data["week_int"]-1)/4)+1
    data["virtuele_maand"]=data['virtuele_maand'].astype(int)
    data["virtuele_maandnr"] = data["jaar"].astype(str) +"_" + data["virtuele_maand"].astype(str).str.zfill(2)
    #data = data.round({'virtuele_maand': 0})

    data['Geslacht'] = data['Geslacht'].replace(['Totaal mannen en vrouwen'],'m_v_')
    data['Geslacht'] = data['Geslacht'].replace(['Mannen'],'m_')
    data['Geslacht'] = data['Geslacht'].replace(['Vrouwen'],'v_')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['Totaal leeftijd'],'0_999')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['0 tot 65 jaar'],'0_64')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['65 tot 80 jaar'],'65_79')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['80 jaar of ouder'],'80_999')
    data['categorie'] = data['Geslacht']+data['LeeftijdOp31December']

    # print (data.dtypes)
    # Downloaden van metadata
    # metadata = pd.DataFrame(cbsodata.get_meta('70895ned', 'DataProperties'))
    # print(metadata[['Key','Title']])
    df = data.pivot(index=['weeknr', "jaar", "week"], columns='categorie', values = 'Overledenen_1').reset_index()
    df["week"] = df["week"].astype(int)
    df["jaar"] = df["jaar"].astype(int)
  
    return df

def get_all_data():
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df_herfstprik = get_herfstprik()
    df_rioolwater_dag, df_rioolwater = get_rioolwater.scrape_rioolwater()
    df_ = get_sterftedata()

    return df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_



def interface():
    how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte", "year_minus_avg", "p_score"], index = 0)
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how == "p_score"):
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik", "herfstprik", "rioolwater", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "bb")
    else:
        rightax = None
        mergetype = None
    return how,yaxis_to_zero,rightax,mergetype

def main():
    # serienames = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
    #                 "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
    #                 "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
                    "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
                    "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]


    st.header("Overstefte - minder leeftijdscategorieen")
    st.write("Dit script heeft minder leeftijdscategorieen, maar de sterftedata wordt opgehaald van het CBS. Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019")
    how, yaxis_to_zero, rightax, mergetype = interface()
    df_boosters, df_herhaalprik, df_herfstprik,df_rioolwater, df_sterfte = get_all_data()

    plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, serienames, how, yaxis_to_zero, rightax, mergetype)
    
    footer()

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
