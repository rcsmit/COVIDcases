import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import streamlit as st

import pandas as pd
from sklearn.metrics import r2_score

def read():
    # url_yorick = "https://raw.githubusercontent.com/YorickBleijenberg/COVID_data_RIVM_Netherlands/master/vaccination/2021-09-08_vac.cities.csv"
    # df_yorick = pd.read_csv(url_yorick, delimiter=';', decimal=",", encoding="ISO-8859-1")
    # Attentie: bevat - waardes en Baarle Nassau

    #url_yorick = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\vaccinatie_incidentie.csv"
    url_yorick = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/vaccinatie_incidentie.csv"
    df_yorick = pd.read_csv(url_yorick, delimiter=';', encoding="ISO-8859-1")
    # Baarle Nassau is verwijderd (incidentie x*10E-15). Daarnaast worden er 3 gemeentes niet weergegeven ivm herindeingen)
    # Ameland , Noord Beveland, Rozendaal en Schiermoninkoog ook verwijderd ivm incidentie = 0 ->
    # np.log geeft -inf, waardoor correlatie niet kan worden berekend

    # https://www.cbs.nl/nl-nl/maatwerk/2021/05/inkomen-per-gemeente-en-wijk-2018
    # url_inkomen = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\inkomen_per_gemeente.csv"
    url_inkomen = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/inkomen_per_gemeente.csv"
    df_inkomen =pd.read_csv(url_inkomen, delimiter=';')

    df_totaal= pd.merge(
                df_yorick, df_inkomen, how="inner", left_on="Municipality_code", right_on="Gemeentecode"
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

    return df_totaal
def make_scatterplot(df_temp, what_to_show_l, what_to_show_r):
    """Scatterplot maken

    """
    with _lock:

        fig1xy,ax = plt.subplots()

        x_ = np.array(df_temp[what_to_show_l])
        y_ = np.array(df_temp[what_to_show_r])

        show_cat = False
        if show_cat == True:
            cat_ = df_temp['provincie']
            cat_col = df_temp['provincie'].astype('category')
            cat_col_ = cat_col.cat.codes
            scatter = plt.scatter(x_, y_, c = cat_col_, label=cat_)
            legend1 = ax.legend(*scatter.legend_elements(), loc="best")
            ax.add_artist(legend1)
        else:
            scatter = plt.scatter(x_, y_)

        plt.xlabel(what_to_show_l)
        plt.ylabel(what_to_show_r)

        #obtain m (slope) and b(intercept) of linear regression line
        idx = np.isfinite(x_) & np.isfinite(y_)
        m, b = np.polyfit(x_[idx], y_[idx], 1)
        model = np.polyfit(x_[idx], y_[idx], 1)

        predict = np.poly1d(model)
        r2 = r2_score  (y_[idx], predict(x_[idx]))

        #add linear regression line to scatterplot
        plt.plot(x_, m*x_+b, 'r')

        correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)
        title_scatter = (f"{what_to_show_l} -  {what_to_show_r}\nCorrelation spearman = {correlation_sp} - Correlation pearson = {correlation_p}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")
        plt.title(title_scatter)

        ax.text(
            1,
            1.3,
            "Created by Rene Smit — @rcsmit",
            transform=ax.transAxes,
            fontsize="xx-small",
            va="top",
            ha="right",
        )
        #plt.show()
        st.pyplot(fig1xy)

def main():
    df = read()

    make_scatterplot(df,  "gem_ink_x1000","log_e_incidentie" )
    make_scatterplot(df,  "gem_ink_x1000", "volledige.vaccinatie" )

    make_scatterplot(df,  "volledige.vaccinatie", "incidentie", )
    make_scatterplot(df,  "volledige.vaccinatie", "log_e_incidentie" )
    make_scatterplot(df,  "volledige.vaccinatie", "log_10_incidentie" )

if __name__ == "__main__":
    main()