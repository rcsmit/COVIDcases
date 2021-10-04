import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import streamlit as st
import plotly.express as px
#import statsmodels

#from streamlit_plotly_events import plotly_events

import pandas as pd
from sklearn.metrics import r2_score

def read():
    # url_yorick = "https://raw.githubusercontent.com/YorickBleijenberg/COVID_data_RIVM_Netherlands/master/vaccination/2021-09-08_vac.cities.csv"
    # df_yorick = pd.read_csv(url_yorick, delimiter=';', decimal=",", encoding="ISO-8859-1")
    # Attentie: bevat - waardes en Baarle Nassau

    #url_yorick = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\vaccinatie_incidentie.csv"
    url_yorick = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinatie_incidentie.csv"
    df_yorick = pd.read_csv(url_yorick, delimiter=';', encoding="ISO-8859-1")
    # Baarle Nassau is verwijderd (incidentie x*10E-15). Daarnaast worden er 3 gemeentes niet weergegeven ivm herindeingen)
    # Ameland , Noord Beveland, Rozendaal en Schiermoninkoog ook verwijderd ivm incidentie = 0 ->
    # np.log geeft -inf, waardoor correlatie niet kan worden berekend

    # https://www.cbs.nl/nl-nl/maatwerk/2021/05/inkomen-per-gemeente-en-wijk-2018
    # url_inkomen = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\inkomen_per_gemeente.csv"
    url_inkomen = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/inkomen_per_gemeente.csv"
    df_inkomen =pd.read_csv(url_inkomen, delimiter=';')


    # BRON: https://www.verkiezingsuitslagen.nl/data/gemeenten/10910
    #url_verkiezingen="C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\verkiezingen2021.csv"
    url_verkiezingen = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/verkiezingen2021.csv"
    df_verkiezingen = pd.read_csv(url_verkiezingen, delimiter=',')

    partijen = df_verkiezingen.columns.values.tolist()
    partijen = partijen[7:]

    df_totaal= pd.merge(
                df_yorick, df_inkomen, how="inner", left_on="Municipality_code", right_on="Gemeentecode"
            )

    df_totaal= pd.merge(
                df_totaal, df_verkiezingen, how="inner", left_on="Municipality_code", right_on="ReGMioCode"
            )

    df_totaal["log_e_incidentie"] = np.log(df_totaal["incidentie"])
    df_totaal["log_10_incidentie"] = np.log10(df_totaal["incidentie"])
    df_totaal['volledige.vaccinatie'] = df_totaal['volledige.vaccinatie'].astype(float)

    # uitschieters verwiijderen
    factor =2
    kolommen = ["incidentie",  "volledige.vaccinatie"]
    for kolom in kolommen:
        mean = df_totaal[kolom].mean()
        stdev = df_totaal[kolom].std()
        df_totaal = df_totaal[(df_totaal[kolom] > mean -(factor*stdev)) & (df_totaal[kolom] < mean +(factor*stdev)) ]
    #url_uitslag =  "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\uitslag_per_partij2021.csv"
    url_uitslag = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/uitslag_per_partij2021.csv"
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
                    fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, size='inwoners', trendline="ols",
                        hover_name="Gemeente_Naam", hover_data=["provincie"])


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
            st.plotly_chart(fig1xy)


def bewerk_df(df, partijen_selected):
    df["stemmen_op_geselecteerde_partijen_procent"] = df[partijen_selected].sum(axis=1)
    return df

def make_corr_tabel(df, partijen, uitslag):
    corr_tabel =  pd.DataFrame(
            {"partij": [], "corr_vaccinatie": [], "corr_incidentie": []}
        )
    for p in partijen:

        corr_vv = round(df[p].corr(df['volledige.vaccinatie']),2)
        corr_inc = round(df[p].corr(df['incidentie']),2)

        corr_tabel = corr_tabel.append(
                    {
                        "partij": p,
                        "corr_vaccinatie": corr_vv,
                        "corr_incidentie": corr_inc,


                    },
                    ignore_index=True,
                )
    corr_tabel= pd.merge(
                corr_tabel, uitslag, how="inner", left_on="partij", right_on="partij"
            )


    corr_tabel = corr_tabel[corr_tabel["perc_stemmen"] >=0.8]

    st.write(corr_tabel)
    return corr_tabel

def main():
    df, partijen,uitslag = read()
    how  = st.sidebar.selectbox("Plotly (interactive with info on hoover) or pyplot (static - easier to copy/paste)", ["plotly", "pyplot"], index=0)

    partijen_default =  [ 'PVV (Partij voor de Vrijheid)', 'Forum voor Democratie']

    partijen_selected = st.sidebar.multiselect(
            "What to show left-axis (multiple possible)", partijen, partijen_default)
    df = bewerk_df(df, partijen_selected)

    st.subheader("Naar inkomen")
    make_scatterplot(df,  "gem_ink_x1000", "volledige.vaccinatie", how, None)
    make_scatterplot(df,  "gem_ink_x1000","log_e_incidentie", how, None)
    make_scatterplot(df,  "gem_ink_x1000","stemmen_op_geselecteerde_partijen_procent", how, None)

    st.subheader("Naar vaccinatiegraad")
    make_scatterplot(df,  "volledige.vaccinatie", "incidentie", how, None)

    make_scatterplot(df,  "volledige.vaccinatie", "log_e_incidentie", how, None)
    make_scatterplot(df,  "volledige.vaccinatie", "log_10_incidentie" , how, None)


    st.subheader("Naar inwoners")
    make_scatterplot(df,  "inwoners", "incidentie" , how, None)
    make_scatterplot(df,  "inwoners", "volledige.vaccinatie", how, None)
    make_scatterplot(df,  "inwoners","gem_ink_x1000", how, None)

    st.subheader("Naar geselecteerde partijen")
    make_scatterplot(df,  "stemmen_op_geselecteerde_partijen_procent", "volledige.vaccinatie", how, None)
    make_scatterplot(df,  "stemmen_op_geselecteerde_partijen_procent", "incidentie", how, None)
    make_scatterplot(df,  "stemmen_op_geselecteerde_partijen_procent","gem_ink_x1000", how, None)
    make_scatterplot(df,  "stemmen_op_geselecteerde_partijen_procent","inwoners", how, None)

    st.subheader("Correlaties partijen - vacc.graad")
    corr_tabel = make_corr_tabel(df, partijen, uitslag)
    make_scatterplot(corr_tabel,  "corr_vaccinatie","corr_incidentie", how, "verkiezingen")


    st.write("Cijfers dd. 8 september 2021. Datafile met vaccinatiegraad en incidentie samengesteld door Yorick Bleijenberg / @YorickB.  3 gemeentes worden niet weergegeven ivm herindelingen. Baarle Nassau is verwijderd (incidentie x*10E-15). ). Ameland , Noord Beveland, Rozendaal en Schiermoninkoog ook verwijderd ivm incidentie = 0")
if __name__ == "__main__":
    main()