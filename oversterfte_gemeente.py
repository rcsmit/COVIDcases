# Is er een verband tussen vaccinatiegraad en oversterfte

# geinspireerd door
# https://www.linkedin.com/feed/update/urn:li:activity:6896954127893508096/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A6896954127893508096%2C6944757704380809216)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(6944757704380809216%2Curn%3Ali%3Aactivity%3A6896954127893508096)&dashReplyUrn=urn%3Ali%3Afsd_comment%3A(6944773760147193856%2Curn%3Ali%3Aactivity%3A6896954127893508096)&replyUrn=urn%3Ali%3Acomment%3A(activity%3A6896954127893508096%2C6944773760147193856)

# - van de site van CBS pak je overlijdens per gemeente 2015-2019, 
# bepaal sterftetrend per gemeente, 
# trek door naar 2021,
# en schaal het totaal op tot totale verwachte sterfte levert verwachte sterfte per gemeente.
# Dit kun je, rekening houdend met seizoenspatroon per leeftijdsgroep en bevolkingsopbouw per gemeente 
#       verdelen over de weken.
# - van de site van CBS pak je de sterfte per week per gemeente 2021.
# - het verschil met het vorige is de oversterfte per week per gemeente.
# - RIVM heeft per wijk per week de vaccinatiegraad gepubliceerd. 
# De wijken kun je optellen naar gemeenten, en verschil met vaccinatiegraad van de vorige week is h
# et aantal gezette vaccinaties.
# - dan regressieanalyse tussen oversterfte per week per gemeente en 
# gezette prikken per gemeente in de drie voorafgaande weken.
# Als je dat doet, dan zie je geen verband, dat houdt in: geen verhoogde sterfte na vaccinatie.
# Hierbij geldt: geen invloed van doodsoorzaken, want je gaat uit van totale sterfte.


# overlijdens per week: https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1655808656887
# overlijdens per gemeente per maand : https://opendata.cbs.nl/#/CBS/nl/dataset/37230ned/table?ts=1655808862774


import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
import streamlit as st
import plotly.express as px

import platform

import pandas as pd
from sklearn.metrics import r2_score


from sklearn.linear_model import LinearRegression
def gemeente_info():
    url_gemeente_info = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/gemeente_inwoners_oppervlakte.csv"
    df_gemeente_info = pd.read_csv(url_gemeente_info, delimiter=',')
    return df_gemeente_info
def oversterfte_2021():
    if platform.processor() != "":
        url_overlijdens_gemeente_jaar = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_gemeentes_jaar.csv"
    else:
        url_overlijdens_gemeente_jaar = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_gemeentes_jaar.csv"
        
    df = pd.read_csv(url_overlijdens_gemeente_jaar, delimiter=";", low_memory=False)
    #pd.to_numeric(df)
    
    df["2020prediction"]=None
    df["2021prediction"]=None
    
    no_columns =  len(df.columns)
    # https://stackoverflow.com/questions/65523812/estimate-a-linear-trend-in-every-row-across-multiple-columns-in-order-to-project
    X = np.array([[1],[2],[3],[4],[5]]) # dummy x values for each of the columns you are going to use as input data
    
   
    for i in range(len(df)):
        try:
            #print (f"{i} -  {df.iloc[i,0]} ")

            # predict 2020
            Y = df.iloc[i,-9:-4].values.reshape(-1, 1) # work on row i
            
            linear_regressor = LinearRegression()  # create object for the class
            linear_regressor.fit(X, Y)  # perform linear regression

            prediction = linear_regressor.predict(np.array([[6]]))
            
            prediction_=prediction[0]
        
            df.iloc[i,no_columns-2] = prediction_[0]

            #predict 2021
            Y2 = df.iloc[i,-8:-3].values.reshape(-1, 1) # work on row i
        
            linear_regressor = LinearRegression()  # create object for the class
            linear_regressor.fit(X, Y2)  # perform linear regression

            prediction = linear_regressor.predict(np.array([[6]]))
            prediction_=prediction[0]
        
            df.iloc[i,no_columns-1] = prediction_[0]

            laatste_5_jaar = df.iloc[i,-9:-4]
            
            # mean = laatste_5_jaar.mean()
            # std = laatste_5_jaar.std()
            # upper = mean + (2*std)
        except:
            df.iloc[i,no_columns-1] = 9999999  # using None doeesn't work
 
    df = df.drop(df[df["2021prediction"]==9999999].index)
    df = df.drop(df[df["Regio's"]=="Urk"].index)
    df = df.drop(df[df["Regio's"]=="Staphorst"].index)
 
    #df["2021prediction"] = df["2021prediction"].to_numeric(errors='coerce')
    df["oversterfte_abs2020"] =  df["2020"] -df["2020prediction"] 
    df["oversterfte_abs2021"] =  df["2021*"]  - df["2021prediction"] 
    df["oversterfte_proc2020"] =  (df["2020"]-df["2020prediction"]) /df["2020prediction"] *100
    df["oversterfte_proc2021"] =  (df["2021*"] - df["2021prediction"] ) /df["2021prediction"] *100
    # df["oversterfte_methode2_2020"] = ( df["2020"] - ( df[["2015", "2016", "2017", "2018", "2019"]].mean()+2* df[["2015", "2016", "2017", "2018", "2019"]].std()))/df[["2015", "2016", "2017", "2018", "2019"]].mean()
    # df["test"]= df[["2015", "2016", "2017", "2018", "2019"]].mean()

    print (df)
    return df

def vaccinatiegraad():
    
    if platform.processor() != "":
        url_vaccinatie = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\COVID-19_vaccinatiegraad_per_gemeente_per_week_leeftijd (2).csv"
    
       # url_overlijdens_gemeente_jaar = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_gemeentes_jaar.csv"
    else:
        # url_vaccinatie = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/COVID-19_vaccinatiegraad_per_gemeente_per_week_leeftijd.csv"
    
        url_vaccinatie = "https://data.rivm.nl/covid-19/COVID-19_vaccinatiegraad_per_gemeente_per_week_leeftijd.csv"
    
    df_vaccinatie = pd.read_csv(url_vaccinatie, delimiter=';', encoding="ISO-8859-1")
    df_vaccinatie = df_vaccinatie[df_vaccinatie["Age_group"] == "18+"]
    return df_vaccinatie



def make_scatterplot(df_temp, what_to_show_l, what_to_show_r, how, what):
    """Scatterplot maken
    """
    #with _lock:
    if 1==1:
        fig1xy,ax = plt.subplots()
        try:
            x_ = np.array(df_temp[what_to_show_l].astype('float64') )
            y_ = np.array(df_temp[what_to_show_r].astype('float64') )
            #obtain m (slope) and b(intercept) of linear regression line
            idx = np.isfinite(x_) & np.isfinite(y_)
            m, b = np.polyfit(x_[idx], y_[idx], 1)
            model = np.polyfit(x_[idx], y_[idx], 1)

            predict = np.poly1d(model)
            r2 = r2_score  (y_[idx], predict(x_[idx]))
        except:
            r2= None

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
            
                fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r,size='inwoners_2021', hover_name="Regio's", trendline="ols",)


        #add linear regression line to scatterplot


        correlation_sp = round(df_temp[what_to_show_l].astype('float64') .corr(df_temp[what_to_show_r].astype('float64') , method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].astype('float64').corr(df_temp[what_to_show_r].astype('float64') , method='pearson'), 3)
       
        if how == "pyplot":
            title_scatter = None # (f"{what_to_show_l} -  {what_to_show_r}\nCorrelation spearman = {correlation_sp} - Correlation pearson = {correlation_p}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

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

def main():
    st.header("Oversterfte vs. Vaccinatiegraad per gemeente ")
   
    inwonersgrens = st.sidebar.number_input("Miniumum aantal inwoners", 0, None, value = 100_000)
    how  = st.sidebar.selectbox("Plotly (interactive with info on hoover) or pyplot (static - easier to copy/paste)", ["plotly", "pyplot"], index=0)
   
    df_totaal = read_data(inwonersgrens)
    df_totaal["Coverage_primary_completed"] = df_totaal["Coverage_primary_completed"].replace("<=5", 0).astype(int)

    x = "Coverage_primary_completed" # was "Vaccination_coverage_completed"

    y1 = "oversterfte_proc2020"
    y2 = "oversterfte_proc2021"
    
    make_scatterplot(df_totaal,  x, y1 , how, None)
    make_scatterplot(df_totaal,  x, y2 , how, None)
    
def read_data(inwonersgrens):
    df_vaccinatie = vaccinatiegraad()
    df_oversterfte = oversterfte_2021()
    df_gemeente_info = gemeente_info()

    df_totaal= pd.merge(
                df_vaccinatie, df_oversterfte, how="inner", left_on="Region_name", right_on="Regio's",
            )

    df_totaal= pd.merge(
                 df_totaal, df_gemeente_info, how="inner", left_on="Region_name", right_on="gemeentenaam"
            )
    df_totaal = df_totaal[df_totaal["inwoners_2021"]>= inwonersgrens]
    a= df_totaal["inwoners_2021"].sum()
    b =  df_gemeente_info["inwoners_2021"].sum()
    st.sidebar.write(f"Totaal inwoners inbegrepen {a} =  {round(a/b*100,2)}%")
    st.sidebar.write("Er is een grens voor inwoneraaantal. Er worden gemeentes weggelaten wegens ontbrekende gegevens, herindlelingen of naamswijzigingen. Er wordt gewerkt met de laatst bekende volledige vaccinatiegraad per gemeente.")

    st.write (df_totaal)
    return df_totaal
if __name__ == "__main__":
    print ("============================================================================")
    main()