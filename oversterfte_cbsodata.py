import pandas as pd
import cbsodata


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
# import platform
from oversterfte_helpers import *
from sterfte_rivm import *
import get_rioolwater
# from streamlit import caching

# 70895ned = https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1659307527578
# Overledenen; geslacht en leeftijd, per week

# Downloaden van tabeloverzicht
# toc = pd.DataFrame(cbsodata.get_table_list())

# Downloaden van gehele tabel (kan een halve minuut duren)
@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata():
    data = pd.DataFrame(cbsodata.get_data('70895ned'))
 
    data[['jaar','week']] = data.Perioden.str.split(" week ",expand=True,)
    #data['week_'] = data['week_']+" _"
    #data['week_'] = data['week_'].replace(' dagen','_',  regex=True)
    #data['week_'] = data['week_'].replace('\)_',')',  regex=True)
    # Remove rows where 'Perioden' contains 'dagen'
    data = data[~data['Perioden'].str.contains('dagen')]
    data = data[~data['Perioden'].str.contains('dag')]
    # print (data)
    data = data.reset_index()

    

    #data[['week','aantal_dagen']] = data.week_.str.split(" ",expand=True,)
    data = data[data['week'].notnull()] #remove the year-totals
    # data['aantal_dagen'] = data['aantal_dagen'].replace('_','7')
    # data['aantal_dagen'] = data['aantal_dagen'].replace('\(','', regex=True)
    # data['aantal_dagen'] = data['aantal_dagen'].replace('_\)','', regex=True)
    # data = data[data['aantal_dagen'] == '7'] # only complete weeks
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
    """_summary_

    Returns:
        _type_: df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_
    """    
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df_herfstprik = get_herfstprik()
    #df_rioolwater_dag, df_rioolwater = None, None # get_rioolwater.scrape_rioolwater()
    df_kobak = get_kobak()
    df_rioolwater = get_rioolwater_simpel()
    df_ = get_sterftedata()


    return df_, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak

def get_rioolwater_simpel():
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaarde2024.csv"
    # else: 
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwaarde2024.csv"
    df_rioolwater = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
        )
    df_rioolwater["weeknr"] = df_rioolwater["jaar"].astype(int).astype(str) +"_"+df_rioolwater["week"].astype(int).astype(str)
    df_rioolwater["value_rivm_official_sma"] =  df_rioolwater["rioolwaarde"].rolling(window = 5, center = False).mean().round(1)
    # print (df_rioolwater)
    return df_rioolwater


def interface():
    how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte", "year_minus_avg", "p_score"], index = 0)
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how == "p_score") or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") :
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik", "herfstprik", "rioolwater", "kobak", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "bb")
        sec_y = st.sidebar.selectbox("Secondary Y axis", [True, False], index = 0, key = "cc")
    else:
        rightax = None
        mergetype = None
        sec_y = None
    return how,yaxis_to_zero,rightax,mergetype, sec_y

def main():
    # serienames = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
    #                 "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
    #                 "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames_ = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
                    "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
                    "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames = st.sidebar.multiselect("Leeftijden", serienames_, ["m_v_0_999"])
    st.header("Overstefte - minder leeftijdscategorieen")
    st.write("Dit script heeft minder leeftijdscategorieen, maar de sterftedata wordt opgehaald van het CBS. Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019")
    how, yaxis_to_zero, rightax, mergetype, sec_y = interface()
    df_sterfte, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak = get_all_data()
    plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, serienames, how, yaxis_to_zero, rightax, mergetype, sec_y)
   
    footer()

    
def comparison():
    st.subheader("Comparison")
    df_sterfte, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak = get_all_data()
    #plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, ["m_v_0_999"],"quantiles", False, None, None, None) 
    series_name = "m_v_0_999"
    df_data = get_data_for_series(df_sterfte, series_name).copy(deep=True)
    df_corona, df_quantile = make_df_qantile(series_name, df_data) 
   
    df_rivm = sterfte_rivm(df_sterfte, series_name)
    plot_graph_rivm(df_rivm, series_name, False)

    # Merge de dataframes, waarbij we de kolomnaam van df_quantile aanpassen tijdens de merge
    df_merged = df_corona.merge(df_quantile, left_on='weeknr', right_on='week_').merge(df_rivm, on='weeknr')

    # Verwijder de extra 'week_' kolom uit het eindresultaat
    df_merged = df_merged.drop(columns=['week_'])
    
    columns = [[series_name, "aantal_overlijdens"],
                ["avg", "verw_cbs"],
                ["low05", "low_cbs"],
                ["high95", "high_cbs"],
                ["voorspeld", "verw_rivm"],
                ["lower_ci", "high_rivm"],
                ["upper_ci", "low_rivm"]]
    for c in columns:
        print (c)
        df_merged = df_merged.rename(columns={c[0]:c[1]})
    
    show_difference(df_merged, "weeknr")
    
    df_merged["oversterfte_cbs"] = df_merged["aantal_overlijdens"] - df_merged["verw_cbs"]
    df_merged["oversterfte_rivm"] = df_merged["aantal_overlijdens"] - df_merged["verw_rivm"]
    df_merged["oversterfte_cbs_cumm"] = df_merged["oversterfte_cbs"].cumsum()
    df_merged["oversterfte_rivm_cumm"] = df_merged["oversterfte_rivm"].cumsum()
    fig = go.Figure()
    for n in ['cbs', 'rivm']:
        # Voeg de werkelijke data toe
        fig.add_trace(go.Scatter(
            x=df_merged['weeknr'],
            y=df_merged[f'oversterfte_{n}_cumm'],
            mode='lines',
            name=f'cummulatieve oversterfte {n}'
        ))
    
    # Titel en labels toevoegen
    fig.update_layout(
        title='Cumm oversterfte (simpel)',
        xaxis_title='Tijd',
        yaxis_title='Aantal'
    )

    st.plotly_chart(fig)
   

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    #main()
    comparison()
