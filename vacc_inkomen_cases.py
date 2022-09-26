import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import streamlit as st
import plotly.express as px
#import statsmodels
import platform
#from streamlit_plotly_events import plotly_events

import pandas as pd
from sklearn.metrics import r2_score
import datetime as dt
import datetime

def input_period():
    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    end_ = "2022-01-01"

    #today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)
    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", end_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")
    return FROM, UNTIL

def mask_data(df,from_,until_, datefield):
    """[summary]

    Args:
        df ([type]): [description]
        from_ ([type]): [description]
        until_ ([type]): [description]

    Returns:
        [type]: [description]
    """
    # masking data in sperate function for caching issues

    mask = (df[datefield].dt.date >= from_) & (df[datefield].dt.date <= until_)
    df = df.loc[mask]
    return df

@st.cache(ttl=60 * 60 * 24, allow_output_mutation=True)
def main_week_data(from_, until_):
    """Het maken van weekcijfers en gemiddelden tbv cases_hospital_decased_NL.py
    """
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    # url1 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_aantallen_gemeente_per_dag.csv"
    # #C:\Users\rcxsm\Documents\pyhton_scripts\covid19_seir_models\COVIDcases\input
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Downloads\\COVID-19_aantallen_gemeente_per_dag.csv"
    else:
        url1 = "https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv"
    datefield="Date_of_publication"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df[datefield] = pd.to_datetime(df[datefield], format="%Y-%m-%d")
    df = df[df["Municipality_code"] != None]
    df["count"] = 1

    df = mask_data(df,from_,until_, datefield)
    # print ("line27")
    # print (df)
    # from_  = dt.datetime.strptime("2021-11-16", "%Y-%m-%d").date()
    # until = dt.datetime.strptime("2021-12-7", "%Y-%m-%d").date()


    # df = df.set_index(datefield)
    # n_days = str(number_of_days) + "D"
    # df = df.last(n_days)
    print ("line 35")
    print (df)
    df = df.groupby(["Municipality_code"] ).sum().reset_index()
    #df_week = df.groupby([  pd.Grouper(key='Date_statistics', freq='W'), "Agegroup",] ).sum().reset_index()
    print ("line 40")
    print (df)

    return df

@st.cache(ttl=60 * 60 * 24, allow_output_mutation=True)
def read(inwonersgrens, from_,until_):
    # url_yorick = "https://raw.githubusercontent.com/YorickBleijenberg/COVID_data_RIVM_Netherlands/master/vaccination/2021-09-08_vac.cities.csv"
    # df_yorick = pd.read_csv(url_yorick, delimiter=';', decimal=",", encoding="ISO-8859-1")
    # Attentie: bevat - waardes en Baarle Nassau

    #url_yorick = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\vaccinatie_incidentie.csv"
    # url_yorick = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinatie_incidentie.csv"

    # Baarle Nassau is verwijderd (incidentie x*10E-15). Daarnaast worden er 3 gemeentes niet weergegeven ivm herindeingen)
    # Ameland , Noord Beveland, Rozendaal en Schiermoninkoog ook verwijderd ivm incidentie = 0 ->
    # np.log geeft -inf, waardoor correlatie niet kan worden berekend

    # https://www.cbs.nl/nl-nl/maatwerk/2021/05/inkomen-per-gemeente-en-wijk-2018
    # url_inkomen = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\inkomen_per_gemeente.csv"
    url_inkomen = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/inkomen_per_gemeente.csv"
    df_inkomen =pd.read_csv(url_inkomen, delimiter=';')


    # BRON: https://www.verkiezingsuitslagen.nl/data/gemeenten/10910
    #url_verkiezingen="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\verkiezingen2021.csv"
    url_verkiezingen = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_verkiezingen2021.csv"
    df_verkiezingen = pd.read_csv(url_verkiezingen, delimiter=',')

    url_gemeente_info = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_inwoners_oppervlakte.csv"
    df_gemeente_info = pd.read_csv(url_gemeente_info, delimiter=',')


    # C:\Users\rcxsm\Documents\pyhton_scripts\covid19_seir_models\COVIDcases\not_active_on_streamlit\preprare_gemeenten_per_dag.py
    #url_rene = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_reported_hospital_deceased.csv"
    df_rene = main_week_data(from_,until_)


    #df_rene = pd.read_csv(url_rene, delimiter=',', encoding="ISO-8859-1")

    #https://data.rivm.nl/meta/srv/dut/catalog.search#/metadata/205d0bf4-b645-4e5b-84bc-f8ec482fd3f3
    url_vaccinatie = "https://data.rivm.nl/covid-19/COVID-19_vaccinatiegraad_per_gemeente_per_week_leeftijd.csv"
    df_vaccinatie = pd.read_csv(url_vaccinatie, delimiter=';', encoding="ISO-8859-1")
    df_vaccinatie = df_vaccinatie[df_vaccinatie["Age_group"] == "18+"]

    url_migratie = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_percentage_migratieachtergrond.csv"
    df_migratie = pd.read_csv(url_migratie, delimiter=',')

    url_niet_w_migratie = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_niet_west_immigr.csv"
    df_niet_west_migratie = pd.read_csv(url_niet_w_migratie, delimiter=',')
    partijen = df_verkiezingen.columns.values.tolist()
    partijen = partijen[7:]



    df_totaal= pd.merge(
                 df_inkomen, df_verkiezingen, how="outer", left_on="Gemeentecode", right_on="ReGMioCode"
            )
    df_totaal= pd.merge(
                 df_totaal, df_gemeente_info, how="outer", left_on="RegioNaam", right_on="gemeentenaam"
            )
    #print (df_totaal.dtypes)
    df_totaal= pd.merge(
                df_totaal, df_rene, how="outer", left_on="Gemeentecode", right_on="Municipality_code",
            )
    df_totaal= pd.merge(
                df_totaal, df_vaccinatie, how="outer", left_on="Gemeentecode", right_on="Region_code",
            )
    df_totaal= pd.merge(
                 df_totaal, df_migratie, how="outer", left_on="RegioNaam", right_on="gemeente"
            )
    df_totaal= pd.merge(
                 df_totaal, df_niet_west_migratie, how="outer", left_on="RegioNaam", right_on="gemeentenaam"
            )

    df_totaal["Total_reported_per_inwoner_period"] = df_totaal["Total_reported"] / df_totaal["inwoners_2021"]
    #df_totaal["Hospital_admission_per_inwoner_period"] = df_totaal["Hospital_admission"] / df_totaal["inwoners_2021"]
    df_totaal["Deceased_per_inwoner_period"] = df_totaal["Deceased"] / df_totaal["inwoners_2021"]
    df_totaal["log_e_incidentie"] = np.log(df_totaal["Total_reported_per_inwoner_period"])
    df_totaal["log_10_incidentie"] = np.log10(df_totaal["Total_reported_per_inwoner_period"])
    #df_totaal['Vaccination_coverage_completed']
    #print(df_totaal.info(verbose=True))
    #df_totaal['volledige.vaccinatie'] = df_totaal['Vaccination_coverage_completed'].astype(str)
    df_totaal['volledige.vaccinatie'] = df_totaal['Coverage_primary_completed'].apply(lambda x: '99' if x=='>=95'  else x)
    df_totaal['volledige.vaccinatie'] = df_totaal['Coverage_primary_completed'].apply(lambda x: '0' if x=='<=5'  else x)
    df_totaal['volledige.vaccinatie'] = df_totaal['volledige.vaccinatie'].astype(float)


    df_totaal = df_totaal[df_totaal["inwoners_2021"]>= inwonersgrens]

    # uitschieters verwiijderen
    factor =3
    kolommen = ["Total_reported_per_inwoner_period",  "volledige.vaccinatie"]
    for kolom in kolommen:
        mean = df_totaal[kolom].mean()
        stdev = df_totaal[kolom].std()
        df_totaal = df_totaal[(df_totaal[kolom] > mean -(factor*stdev)) & (df_totaal[kolom] < mean +(factor*stdev)) ]
    #url_uitslag =  "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\uitslag_per_partij2021.csv"
    url_uitslag = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/landelijk_uitslag_per_partij2021.csv"
    uitslag =pd.read_csv(url_uitslag, delimiter=',')


    return df_totaal, partijen, uitslag


def make_scatterplot(df_temp, what_to_show_l, what_to_show_r, how, what):
    """Scatterplot maken
    """
    with _lock:
        fig1xy,ax = plt.subplots()

        x_ = np.array(df_temp[what_to_show_l])
        y_ = np.array(df_temp[what_to_show_r])
        #obtain m (slope) and b(intercept) of linear regression line
        idx = np.isfinite(x_) & np.isfinite(y_)
        m, b = np.polyfit(x_[idx], y_[idx], 1)
        model = np.polyfit(x_[idx], y_[idx], 1)

        predict = np.poly1d(model)
        r2 = r2_score  (y_[idx], predict(x_[idx]))


        show_cat = False
        if show_cat == True:
            # TOFIX
            cat_ = df_temp['provincie']
            cat_col = df_temp['provincie'].astype('category')
            cat_col_ = cat_col.cat.codes
            scatter = plt.scatter(x_, y_, c = cat_col_, label=cat_)
            legend1 = ax.legend(*scatter.legend_elements(), loc="best")
            ax.add_artist(legend1)
        else:
            if how== "pyplot":
                scatter = plt.scatter(x_, y_)

            elif how == "plotly":
                if what == "verkiezingen":

                    fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, size='perc_stemmen', text="partij", trendline="ols")

                else:
                    fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, size='inwoners_2021', trendline="ols",
                        hover_name="Gemeentenaam", hover_data=["provincie"])


        #add linear regression line to scatterplot


        correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)

        if how == "pyplot":
            title_scatter = (f"{what_to_show_l} -  {what_to_show_r}\nCorrelation spearman = {correlation_sp} - Correlation pearson = {correlation_p}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

            plt.plot(x_, m*x_+b, 'r')
            plt.title(title_scatter)
            plt.xlabel(what_to_show_l)
            plt.ylabel(what_to_show_r)
        elif how == "plotly":
            title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

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


        if how == "pyplot":
            st.pyplot(fig1xy)
            #plt.show()
        elif how == "plotly":
            st.plotly_chart(fig1xy, use_container_width=True)


def bewerk_df(df, partijen_selected):
    df["stemmen_op_geselecteerde_partijen_procent"] = df[partijen_selected].sum(axis=1)
    return df

def make_corr_tabel(df, partijen, uitslag):
    corr_tabel =  pd.DataFrame(
            {"partij": [], "corr_vaccinatie": [], "corr_incidentie": []}
        )
    for p in partijen:

        corr_vv = round(df[p].corr(df['volledige.vaccinatie']),2)
        corr_inc = round(df[p].corr(df['Total_reported_per_inwoner_period']),2)

        corr_tabel = corr_tabel.append(
                    {
                        "partij": p,
                        "corr_vaccinatie": corr_vv,
                        "corr_incidentie": corr_inc,


                    },
                    ignore_index=True,
                )
    corr_tabel= pd.merge(
                corr_tabel, uitslag, how="outer", left_on="partij", right_on="partij"
            )


    corr_tabel = corr_tabel[corr_tabel["perc_stemmen"] >=0.8]

    st.write(corr_tabel)
    return corr_tabel

def main():

    how  = st.sidebar.selectbox("Plotly (interactive with info on hoover) or pyplot (static - easier to copy/paste)", ["plotly", "pyplot"], index=0)
    lijst = ["gem_ink_x1000", "volledige.vaccinatie", "Total_reported_per_inwoner_period","Deceased_per_inwoner_period",
            "Total_reported", "Deceased", "inwoners_2021","inwoners_per_km2", "stemmen_op_geselecteerde_partijen_procent","perc_niet_west_migratie_achtergr",
            "perc_migratieachtergrond", "log_e_incidentie", "log_10_incidentie"]
    x  = st.sidebar.selectbox("Wat op X as", lijst, index=0)
    y = st.sidebar.selectbox("Wat op Y as", lijst, index=1)
    inwonersgrens = st.sidebar.number_input("Miniumum aantal inwoners", 0, None, value = 50_000)
    #number_of_days = st.sidebar.number_input("Aantal cases van de laatste ... dagen", 1, 1000, value = 21)
    from_, until_ = input_period()
    df, partijen,uitslag = read(inwonersgrens,from_,until_)

    if (x == "stemmen_op_geselecteerde_partijen_procent" or y == "stemmen_op_geselecteerde_partijen_procent"):
        partijen_default =  [ 'PVV (Partij voor de Vrijheid)', 'Forum voor Democratie']
        partijen_selected = st.sidebar.multiselect(
                "Welke politieke partijen", partijen, partijen_default)
        df = bewerk_df(df, partijen_selected)

    make_scatterplot(df,  x, y , how, None)



    st.subheader("Correlaties partijen - vacc.graad")
    corr_tabel = make_corr_tabel(df, partijen, uitslag)
    make_scatterplot(corr_tabel,  "corr_vaccinatie","corr_incidentie", how, "verkiezingen")


    st.write("Incidentiecijfers zijn totaal tussen 29 september en 20 oktober 2021, per inwoner. Vaccinatiegraad dd 13 otkober 2021. ")
    st.write("Er kunnen gemeentes misssen ivm herindelingen of incidentie = 0.")
    #st.write("3 gemeentes worden niet weergegeven ivm herindelingen. Baarle Nassau is verwijderd (incidentie x*10E-15). ).")
    #st.write(" Ameland , Noord Beveland, Rozendaal en Schiermoninkoog ook verwijderd ivm incidentie = 0")
if __name__ == "__main__":
    main()