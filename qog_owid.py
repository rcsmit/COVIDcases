import streamlit as st
import pandas as pd

import platform
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from streamlit import caching
from sklearn.metrics import r2_score


def download_data_file(url, filename, delimiter_, fileformat):
    """Download the external datafiles
    IN :  url : the url
          filename : the filename (without extension) to export the file
          delimiter : delimiter
          fileformat : fileformat
    OUT : df_temp : the dataframe
    """

    # df_temp = None
    download = True
    with st.spinner(f"Downloading...{url}"):
        if download:  # download from the internet
            url = url
        elif fileformat == "json":
            url = INPUT_DIR + filename + ".json"
        else:
            url = INPUT_DIR + filename + ".csv"

        if fileformat == "csv":
            df_temp = pd.read_csv(url, delimiter=delimiter_, low_memory=False)
        elif fileformat == "json":
            df_temp = pd.read_json(url)
        else:
            st.error("Error in fileformat")
            st.stop()
        df_temp = df_temp.drop_duplicates()
        # df_temp = df_temp.replace({pd.np.nan: None})  Let it to test

        return df_temp


@st.cache(ttl=60 * 60 * 24, suppress_st_warning=True)
def get_data():
    """Get the data from various sources
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):

        data = [

            {
                "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/owid-covid-data_17_10_2021.csv",
                "name": "owid",
                "delimiter": ",",
                "key": "iso_code",
                "key2": "location",
            },


            {
                "url": "https://www.qogdata.pol.gu.se/data/qog_bas_cs_jan21.csv",
                "name": "qog",
                "delimiter": ",",
                "key": "ccodealp",
                "key2": "Location",

            },
             {
                "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinated_sortir_a_paris.csv",
                "name": "vaccinated",
                "delimiter": ",",
                "key": "Country",
                "key2": "Country",
            },

        ]

    type_of_join = "outer"

    # Read first datafile
    df_temp_0 = download_data_file(
        data[0]["url"], data[0]["name"], data[0]["delimiter"], "csv"
    )
    df_temp_1 = download_data_file(
        data[1]["url"], data[1]["name"], data[1]["delimiter"], "csv"
    )
    df_temp_2 = download_data_file(
        data[2]["url"], data[2]["name"], data[2]["delimiter"], "csv"
    )


    df = pd.merge(
        df_temp_0, df_temp_1, how=type_of_join, left_on=data[0]["key"], right_on= data[1]["key"]
    )
    df = pd.merge(
        df, df_temp_2, how=type_of_join, left_on=data[0]["key2"], right_on= data[2]["key"]
    )

    return df

def rename_columns(df):
    df_cog_variables = download_data_file(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/qog_variables.csv","qog_variables", ";", "csv"
        )

    for i in range(len (df_cog_variables)):
        df.rename(columns={df_cog_variables.iloc[i]["abbrevation"]:df_cog_variables.iloc[i]["description"]  }, inplace=True,)
    return df




def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,   categoryfield, hover_name, hover_data):
    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l ([type]): [description]
        what_to_show_r ([type]): [description]
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """
    df_temp = df_temp[df_temp[what_to_show_l] != None]
    df_temp = df_temp[df_temp[what_to_show_r] != None]
    #df_temp = df_temp[df_temp["continent"] != None]
    #df_temp = df_temp[df_temp["location"] != None]
    #st.write(df_temp)
    if len(df_temp) == 0:
        st.warning ("Geen data")
    else:
        correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)

        with _lock:
            fig1xy,ax = plt.subplots()
            try:

                x_ = np.array(df_temp[what_to_show_l])
                y_ = np.array(df_temp[what_to_show_r])
                #obtain m (slope) and b(intercept) of linear regression line
                idx = np.isfinite(x_) & np.isfinite(y_)
                m, b = np.polyfit(x_[idx], y_[idx], 1)
                model = np.polyfit(x_[idx], y_[idx], 1)

                predict = np.poly1d(model)
                r2 = r2_score  (y_[idx], predict(x_[idx]))
                fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data, trendline="ols", trendline_scope = 'overall', trendline_color_override = 'black')
                title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

            except:

                fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data)
                title_scatter = (f"{what_to_show_l} -  {what_to_show_r}")

            #fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r,  hover_name=hover_name, color=categoryfield)

            #
            fig1xy.update_layout(
                title=dict(
                    text=title_scatter,
                    x=0.5,
                    y=0.95,
                    font=dict(
                        family="Arial",
                        size=14,
                        color='#000000'
                    )
                ),
                xaxis_title=what_to_show_l,
                yaxis_title=what_to_show_r,
                font=dict(
                    family="Courier New, Monospace",
                    size=12,
                    color='#000000'
                )
            )

            ax.text(
                1,
                1.3,
                "Created by Rene Smit — @rcsmit",
                transform=ax.transAxes,
                fontsize="xx-small",
                va="top",
                ha="right",
            )

            st.plotly_chart(fig1xy, use_container_width=True)
def main():
    st.header ("COG OWID")
    df_getdata = get_data().copy(deep=False)
    df = rename_columns(df_getdata)
    #df = df.fillna(0)
    continent_list_ =  df["continent"].drop_duplicates().sort_values().tolist()
    continent_list = ["All"] + continent_list_
    #continent_list =  continent_list_
    continent = st.sidebar.selectbox("Continent", continent_list, index=0)
    if continent != "All":
        df = df[df["continent"] == continent]
    #df.dropna(subset=[ "Trust in Politicians"])
    columnlist = df.columns.tolist() +["Clear_cache"]

    #st.write(df["Trust in Politicians"])
    #st.write(df["people_vaccinated_per_hundred"])
    what_to_show_left = st.sidebar.selectbox("X as", columnlist, index=157)
    if  what_to_show_left == "Clear_cache":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")
        st.stop()

    what_to_show_right = st.sidebar.selectbox("Y as", columnlist, index=425)
    #st.write("For vacc.grade choose -Percentage_vaccinated_sop-")
    #try:
    make_scatterplot(df, what_to_show_left, what_to_show_right,   "continent", "location", None)

    st.subheader("Source for QoG data")
    st.write("Dahlberg, Stefan,  Aksel Sundström, Sören Holmberg, Bo Rothstein, Natalia Alvarado Pachon & Cem Mert Dalli. 2021. The Quality of Government Basic Dataset, version Jan21. University of Gothenburg: The Quality of Government Institute, http://www.qog.pol.gu.se doi:10.18157/qogbasjan21")
    st.subheader("Source for Vaccination rates")
    st.write("https://www.sortiraparis.com/news/coronavirus/articles/240384-vaccine-in-the-world-as-of-datadatestodayfrlatest-the-percentage-of-people-vacci/lang/en dd 18/10/2021")
    st.subheader("Source for Our World In Data-data")
    st.write("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")
    st.header( "Fields")
    st.write(columnlist)

if __name__ == "__main__":
    #caching.clear_cache()
    main()