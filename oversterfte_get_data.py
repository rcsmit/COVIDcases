import pandas as pd
import cbsodata
import streamlit as st

@st.cache_data(ttl=60 * 60 * 24)
def get_all_data():
    """_summary_

    Returns:
        _type_: df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_
    """
    print ("get all data")
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df_herfstprik = get_herfstprik()
    df_kobak = get_baseline_kobak()
    df_rioolwater = get_rioolwater_simpel()  
    print ("Loading cbs data")
    cbs_data_ruw = pd.DataFrame(cbsodata.get_data("70895ned"))
    return df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_kobak, cbs_data_ruw



def get_baseline_kobak():
    """Load the csv with the baseline as calculated by Ariel Karlinsky and Dmitry Kobak
        https://elifesciences.org/articles/69336#s4
        https://github.com/dkobak/excess-mortality/

    Returns:
        _type_: _description_
    """
    
    url = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/kobak_baselines.csv"
    # url ="C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\kobak_baselines.csv"     # Maak een interactieve plot met Plotly
    df_ = pd.read_csv(
        url,
        delimiter=",",
        low_memory=False,
    )

    df_["periodenr"] = df_["jaar"].astype(str) + "_" + df_["week"].astype(str).str.zfill(2)
    df_ = df_[["periodenr", "baseline_kobak"]]
    return df_

def get_df_offical():
    """Laad de waardes zoals door RIVM en CBS zijn bepaald. Gedownload dd 11 juni 2024
    jaar_z,week_z,datum,Overledenen_z,verw_cbs_official,low_cbs_official,high_cbs_official,aantal_overlijdens_z,low_rivm_official,high_rivm_official,opgehoogd

    Returns:
        _df
    """
    file = "C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overl_cbs_vs_rivm.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overl_cbs_vs_rivm.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_["periodenr_z"] = (
        df_["jaar_z"].astype(str) + "_" + df_["week_z"].astype(str).str.zfill(2)
    )
    df_["verw_rivm_official"] = (
        df_["low_rivm_official"] + df_["high_rivm_official"]
    ) / 2

    return df_

def get_boosters():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\boosters_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/boosters_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )
    df_["week"] = df_["weeknr"]
    df_=df_.drop("weeknr", axis=1)

    df_["periodenr"] = (
        df_["jaar"].astype(str) + "_" + df_["week"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    df_["boosters_m_v_0_64"] = df_["boosters_m_v_0_49"] + df_["boosters_m_v_50_64"]
    df_["boosters_m_v_80_999"] = df_["boosters_m_v_80_89"] + df_["boosters_m_v_90_999"]

    return df_

def get_herhaalprik():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herhaalprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )
    df_["week"] = df_["weeknr"]
    df_=df_.drop("weeknr", axis=1)
    df_["herhaalprik_m_v_0_64"] = (
        df_["herhaalprik_m_v_0_49"] + df_["herhaalprik_m_v_50_64"]
    )
    df_["herhaalprik_m_v_80_999"] = (
        df_["herhaalprik_m_v_80_89"] + df_["herhaalprik_m_v_90_999"]
    )

    df_["periodenr"] = (
        df_["jaar"].astype(str) + "_" + df_["week"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    return df_

def get_herfstprik():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herfstprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herfstprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_["week"] = df_["weeknr"]
    df_=df_.drop("weeknr", axis=1)
    df_["herfstprik_m_v_0_64"] = (
        df_["herfstprik_m_v_0_49"] + df_["herfstprik_m_v_50_64"]
    )
    df_["herfstprik_m_v_80_999"] = (
        df_["herfstprik_m_v_80_89"] + df_["herfstprik_m_v_90_999"]
    )

    df_["periodenr"] = (
        df_["jaar"].astype(str) + "_" + df_["week"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    return df_

def get_rioolwater_simpel():
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaarde2024.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwater_2024okt.csv"
    
    df_rioolwater = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )
    
    df_rioolwater["rioolwaarde"] = df_rioolwater["RNA_flow_per_100000"]

    df_rioolwater = df_rioolwater.drop("RNA_flow_per_100000", axis=1)
    
    df_rioolwater["periodenr"] = (
        df_rioolwater["jaar"].astype(int).astype(str)
        + "_"
        + df_rioolwater["week"].astype(int).astype(str)
    )
    df_rioolwater["rioolwater_sma"] = (
        df_rioolwater["rioolwaarde"].rolling(window=5, center=False).mean().round(1)
    )

    return df_rioolwater

def get_df_offical():
    """Laad de waardes zoals door RIVM en CBS zijn bepaald. Gedownload dd 11 juni 2024 RIVM updated 27/10/2024
    jaar_z,week_z,datum,Overledenen_z,verw_cbs_official,low_cbs_official,high_cbs_official,aantal_overlijdens_z,low_rivm_official,high_rivm_official,opgehoogd

    Returns:
        _df
    """
    file = "C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overl_cbs_vs_rivm.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overl_cbs_vs_rivm.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_["periodenr_z"] = (
        df_["jaar_z"].astype(str) + "_" + df_["week_z"].astype(str).str.zfill(2)
    )
    df_["verw_rivm_official"] = (
        df_["low_rivm_official"] + df_["high_rivm_official"]
    ) / 2

    return df_

def get_data_rivm():
    """laad de waardes zoals RIVM die heeft vastgesteld (11 juni 2024)

    Returns:
        _df: df
    """
    url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rivm_sterfte.csv"
    df_ = pd.read_csv(
        url,
        delimiter=";",
        low_memory=False,
    )
    df_["periodenr"] = df_["weeknr"]
    df_=df_.drop("weeknr", axis=1)
    return df_
