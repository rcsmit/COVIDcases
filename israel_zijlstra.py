#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
#from scipy.stats import weibull_min
import pandas as pd
from statistics import mean
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
import streamlit as st
import random
from itertools import cycle
#from streamlit import caching
import time
# partly derived from https://stackoverflow.com/a/37036082/4173718
import pandas as pd
import numpy as np
#import openpyxl
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from scipy.stats import fisher_exact

# na 21/8 zelfde waardes voor de vaccinaties voor 90+ aangehoude
@st.cache(ttl=60 * 60 * 24)
def read():
    sheet_id = "104fiQWDNmLP73CBEo5Lu-fXeTOYiTx_7PZb0N-b3joE"
    sheet_name = "mastersheet"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    url_data = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/israel_zijlstra/israel_zijlstra_mastersheet.csv"
    url_populatie = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/israel_zijlstra/populatie_grootte.csv"
    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\schoonmaaktijden.csv",
    df_data = pd.read_csv(url_data, delimiter=',', error_bad_lines=False)
    df_populatie = pd.read_csv(url_populatie, delimiter=',')
    df = pd.merge(df_data, df_populatie, how="outer", on="Age_group" )
    #df = df[:-1]  #remove last row which appears to be a Nan

    df["einddag_week"] = pd.to_datetime(df["einddag_week"], format="%d-%m-%Y")
    # for col in df.select_dtypes(include=['object']).columns:
    #     try:
    #         df[col] = pd.to_numeric(df[col])
    #         df[col] = df[col].fillna(0)
    #     except:
    #         pass
    return df


def line_chart (df, what_to_show):
    """Make a linechart from an unpivoted table, with different lines (agegroups)

    Args:
        df ([type]): [description]
        what_to_show ([type]): [description]
    """
    # fig = go.Figure()
    try:
        fig = px.line(df, x="einddag_week", y=what_to_show, color='Age_group')
    except:
        fig = px.line(df, x="einddag_week", y=what_to_show)
    fig.update_layout(
        title=what_to_show,
        xaxis_title="Einddag vd week",
        yaxis_title=what_to_show,
    )
    st.plotly_chart(fig, use_container_width=True)

def line_chart_pivot (df_, field, title):
    """Makes a linechart from a pivoted table, each column in a differnt line. Smooths the lines too.

    Args:
        df ([type]): [description]
        title ([type]): [description]
    """
    df = make_pivot(df_, field)
    fig = go.Figure()

    columns = df.columns.tolist()
    columnlist = columns[1:]
    # st.write(columnlist)
    for col in columnlist:
        col_sma = col +"_sma"

        df[col_sma] =  df[col].rolling(window = 3, center = False).mean()
        fig.add_trace(go.Scatter(x=df["einddag_week"], y= df[col_sma], mode='lines', name=col ))

    fig.update_layout(
        title=dict(
                text=title+ " (SMA 3)",
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title="VE"    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander (f"dataframe pivottable {title}"):
        df_temp = df.astype(str).copy(deep = True)
        st.write (df_temp)

def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,  show_cat, categoryfield, hover_name, hover_data):
    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l ([type]): [description]
        what_to_show_r ([type]): [description]
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """
    # with _lock:
    if 1==1:
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
        except:
            m,b,model,predict,r2 =None,None,None,None,None

        try:

            fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data, trendline="ols", trendline_scope = 'overall', trendline_color_override = 'black')
        except:
            # avoid exog contains inf or nans
            fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data)

        #add linear regression line to scatterplot


        correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)


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


        st.plotly_chart(fig1xy, use_container_width=True)
def make_calculations(df):

    df["unvaxxed_new"] = df["populatie_grootte"] - ( df["sec_cumm"]  - df["third_cumm"] ) - df["third_cumm"] #NB Unvaxxed is vanaf 2e vaccin
    df["healthy_vax"] =   df["sec_cumm"] - df["positive_above_20_days_after_2nd_dose"]
    df["healthy_nonvax"] =  df["unvaxxed_new"] - df["Sum_positive_without_vaccination"]

    df["unboostered_new"] = df["populatie_grootte"] -  df["third_cumm"]
    df["perc_sec_dose"] = round(((df["sec_cumm"] -  df["third_cumm"]) / df["populatie_grootte"])*100,1)
    df["perc_boostered"] = round((df["third_cumm"] / df["populatie_grootte"])*100,1)
    df["ziek_V_2_per_100k"] = (df["positive_above_20_days_after_2nd_dose"] / ( df["sec_cumm"] - df["third_cumm"])*100_000)
    df["ziek_V_3_per_100k"] = (df["positive_above_20_days_after_3rd_dose"] /  df["third_cumm"] *100_000)
    df["ziek_N_per_100k"] = (df["Sum_positive_without_vaccination"] /  df["unvaxxed_new"]  *100_000)
    df["VE_2_N"] = (1 - ( df["ziek_V_2_per_100k"]/ df["ziek_N_per_100k"]))*100
    df["HR_2_N"] = ( ( df["ziek_V_2_per_100k"]/ df["ziek_N_per_100k"]))*100 # zelfde als RR of IRR
    df["VE_3_N"] = (1 - ( df["ziek_V_3_per_100k"]/ df["ziek_N_per_100k"]))*100
    df["VE_3_N_2_N"]= (1 - ( df["ziek_V_3_per_100k"]/ df["ziek_V_2_per_100k"]))*100


    # after second dose
    # https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/

    df["p_inf_vacc"] = df["positive_above_20_days_after_2nd_dose"] /  df["sec_cumm"]
    df["p_inf_non_vacc"] = df["Sum_positive_without_vaccination"] / df["unvaxxed_new"]





            # df["fisher_oddsratio"], df["fisher_pvalue"] = fisher_exact(([df["positive_above_20_days_after_2nd_dose"], df["healthy_vax"]),
            #                                              (df["Sum_positive_without_vaccination"],      df["healthy_nonvax"])])


    # ODDS RATIO = IRR
    # VE = -100 * OR + 100

    df["odds_ratio_V_2_N"] = (  df["p_inf_vacc"]/(1-  df["p_inf_vacc"])) /  (   df["p_inf_non_vacc"] / (1-   df["p_inf_non_vacc"]))
    # https://wikistatistiek.amc.nl/index.php/Logistische_regressie
    df["odds_ratio_amc"] = ( df["positive_above_20_days_after_2nd_dose"]*   df["healthy_nonvax"] ) / ( df["healthy_vax"] *  df["Sum_positive_without_vaccination"])

    df["IRR"] =  df["odds_ratio_V_2_N"] / ((1-df["p_inf_non_vacc"]) + (df["p_inf_non_vacc"] *  df["odds_ratio_V_2_N"] ))



    df = calculate_fisher(df)
    df = calculate_ci(df)
    #st.write(df)
    return df

def calculate_ci(df):
    import math

    # https://stats.stackexchange.com/questions/297837/how-are-p-value-and-odds-ratio-confidence-interval-in-fisher-test-are-related
    #  p<0.05 should be true only when the 95% CI does not include 1. All these results apply for other α levels as well.
    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
    # https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/

    # 90%	1.64	1.28
    # 95%	1.96	1.65
    # 99%	2.58	2.33
    Za2 = 2.58
    for i in range(0,len(df)):

        a = df.iloc[i]["positive_above_20_days_after_2nd_dose"]
        b = df.iloc[i]["Sum_positive_without_vaccination"]
        c = df.iloc[i]["healthy_vax"]

        d = df.iloc[i]["healthy_nonvax"]

        df.at[i,"a"] =a
        df.at[i,"b"] = b
        df.at[i,"c"] =c
        df.at[i,"d"] = d

        # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
        #https://twitter.com/MrOoijer/status/1445074609506852878
        rel_risk = (a/(a+c))/(b/(b+d)) # relative risk !!
        # SE_theta = ((1/a) - (1/(a+c))  + (1/b)  - (1/(b+d)))**2
        yyy = 1/a  + 1/c
        # df.at[i,"CI_low_theta"] =   theta * np.exp(  Za2 * SE_theta)
        # df.at[i,"or_theta"] = theta
        # df.at[i,"CI_high_theta"]=  theta - np.exp(  Za2 * SE_theta)
        df.at[i,"CI_rel_risk_low"] = np.exp(np.log(rel_risk) -Za2 * math.sqrt(yyy))
        df.at[i,"rel_risk"] = rel_risk
        df.at[i,"CI_rel_risk_high"]= np.exp(np.log(rel_risk) +Za2 * math.sqrt(yyy))


        #  https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/
        # Explained in Martin Bland, An Introduction to Medical Statistics, appendix 13C
        or_ = (a*d) / (b*c)
        xxx = 1/a + 1/b + 1/c + 1/d
        df.at[i,"CI_OR_low"] = np.exp(np.log(or_) -Za2 * math.sqrt(xxx))
        df.at[i,"or_fisher_2"] = or_
        df.at[i,"CI_OR_high"]= np.exp(np.log(or_) +Za2 * math.sqrt(xxx))



    return df

def plot_line_with_ci(dataframe,title, lower, line, upper):
    """
    Interactive plotting for volatility

    input:
    dataframe: Dataframe with upperbound, lowerbound, moving average, close.
    filename: Plot is saved as html file. Assign a name for the file.

    output:
    Interactive plotly plot and html file
    """
    fig = go.Figure()
    upper_bound = go.Scatter(
        name='Upper Bound',
        x=dataframe["einddag_week"],
        y=dataframe[upper] ,
        mode='lines',
        line=dict(width=0.5,
                 color="rgb(255, 188, 0)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty')

    trace1 = go.Scatter(
        name=line,
        x=dataframe["einddag_week"],
        y=dataframe[line],
        mode='lines',
        line=dict(color='rgba(68, 68, 68, 0.8)'),
    	)


    lower_bound = go.Scatter(
        name='Lower Bound',
        x=dataframe["einddag_week"],
        y=dataframe[lower],
        mode='lines',
        line=dict(width=0.5,
                 color="rgb(255, 188, 0)"),
        fillcolor='rgba(68, 68, 68, 0.1)',
       )

    data = [lower_bound, upper_bound, trace1 ]

    layout = go.Layout(
        yaxis=dict(title=title),
        title=title,)
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

def calculate_fisher_from_R(df):
    pass

def calculate_fisher(df):
    """Calculate odds- and p-value of each row with statpy

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """

    # The calculated odds ratio is different from the one R uses. The scipy
    # implementation returns the (more common) "unconditional Maximum
    # Likelihood Estimate", while R uses the "conditional Maximum Likelihood
    # Estimate".
    for i in range(len(df)):
        a =  df.iloc[i]["positive_above_20_days_after_2nd_dose"]
        b =   df.iloc[i]["healthy_vax"]
        c = df.iloc[i]["Sum_positive_without_vaccination"]
        d =   df.iloc[i]["healthy_nonvax"]

        odds, p =  fisher_exact([[a,b],[c,d]])
        df.at[i, "fischer_odds"]= odds
        df.at[i, "fischer_p_val"]= p
    return df

def toelichting(df):
    st.write ("Gemaakt nav https://twitter.com/DennisZeilstra/status/1442121747361374217")
    st.write("    unvaxxed_new = populatie_grootte -  sec_cumm")
    st.write("    unboostered_new = populatie_grootte -  third_cumm")
    st.write("    perc_sec_dose = round((sec_cumm / populatie_grootte)*100,1)")
    st.write("    perc_boostered = round((third_cumm / populatie_grootte)*100,1)")
    st.write("    ziek_V_2_per_100k = positive_above_20_days_after_2nd_dose /  sec_cumm *100_000")
    st.write("    ziek_V_3_per_100k = positive_above_20_days_after_3rd_dose /  third_cumm *100_000")
    st.write("    ziek_N_per_100k = Sum_positive_without_vaccination / unvaxxed_new *100_000")
    st.write("    VE_2_N = (1 - ( ziek_V_2_per_100k/ ziek_N_per_100k))*100")
    st.write("    VE_3_N = (1 - ( ziek_V_3_per_100k/ ziek_N_per_100k))*100")
    st.write("    VE_3_N_2_N= (1 - ( ziek_V_3_per_100k/ ziek_V_2_per_100k))*100")
    st.write("    healthy_vax =   sec_cumm - positive_above_20_days_after_2nd_dose")
    st.write("    healthy_nonvax =  unvaxxed_new - Sum_positive_without_vaccination")
    st.write("")
    st.write("    after second dose")
    st.write("    https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/")
    st.write("")
    st.write("    p_inf_vacc = positive_above_20_days_after_2nd_dose /  sec_cumm")
    st.write("    p_inf_non_vacc = Sum_positive_without_vaccination / unvaxxed_new")
    st.write("")
    st.write("    ODDS RATIO = IRR")
    st.write("    VE = -100 * OR + 100")
    st.write("")
    st.write("    odds_ratio = (  p_inf_vacc/(1-  p_inf_vacc)) /  (   p_inf_non_vacc / (1-   p_inf_non_vacc))")
    st.write("    https://wikistatistiek.amc.nl/index.php/Logistische_regressie")
    st.write("    odds_ratio_amc = ( positive_above_20_days_after_2nd_dose*   healthy_nonvax ) / ( healthy_vax *  Sum_positive_without_vaccination)")
    st.write("    IRR =  odds_ratio / ((1-p_inf_non_vacc) + (p_inf_non_vacc *  odds_ratio )) Incidence Rate Ratio")
    st.write(df)

def group_table(df, valuefield):

    df = df[df["Age_group"] != "0-19"]

    df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
    return df_grouped

def make_pivot(df, valuefield):
    df_pivot = (
    pd.pivot_table(
        df,
        values=valuefield,
        index=["einddag_week"],
        columns=["Age_group"],
        aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    return df_pivot

def main():
    df_ = read()
    df_ = df_.fillna(0)
    # df_["fischer_odds"] = None
    # df_["fischer_p_val"] = None

    df = make_calculations(df_)

    #st.write(df)
    df_grouped = group_table(df_, "einddag_week").copy(deep = True)
    df_grouped = make_calculations(df_grouped)



    #st.write(df_grouped)

    st.subheader ("VE vs non vax - smoothed")

    line_chart_pivot (df,  "VE_2_N", "VE (2 vaccins / N)")
    line_chart_pivot ( df,"odds_ratio_V_2_N", "Odds Ratio (2 vaccins / N)")
    line_chart (df, "fischer_p_val")
    st.subheader("All ages together (excl. 0-19)")

    line_chart (df_grouped,  "VE_2_N")
    line_chart (df_grouped,  "IRR")
    line_chart ( df_grouped,"odds_ratio_V_2_N")
    plot_line_with_ci(df_grouped, "Odds Ratio", "CI_OR_low", "or_fisher_2", "CI_OR_high")
    plot_line_with_ci(df_grouped, "Relative Risk",  "CI_rel_risk_low", "rel_risk", "CI_rel_risk_high")

    line_chart ( df_grouped,"fischer_p_val")
    with st.expander ("Non smoothed", expanded = False):
        st.subheader ("VE and odds ratio vs non vax")
        line_chart (df, "VE_2_N")
        line_chart (df, "odds_ratio_V_2_N")


    st.subheader ("Percentage vaccinated - 2nd and 3rd")
    line_chart (df, "perc_sec_dose")
    line_chart (df, "perc_boostered")

    st.subheader("Are boostershots succesfull?")

    df_boostered = df[df["perc_boostered"] >0 ]
    make_scatterplot(df_boostered, "perc_boostered","VE_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])
    make_scatterplot(df_boostered, "perc_boostered","odds_ratio_V_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])

    st.subheader ("Booster vs nonvax and booster vs doublevaxx")
    line_chart_pivot ( df,  "VE_3_N", "VE (3 vaccins / N)")
    line_chart_pivot ( df,"VE_3_N_2_N", "VE (3 vaccins / 2vaccins")

    st.subheader("Relation between VE, odds_ratio and IRR")
    make_scatterplot(df, "odds_ratio_V_2_N","VE_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])
    make_scatterplot(df, "odds_ratio_V_2_N","HR_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])

    make_scatterplot(df, "odds_ratio_V_2_N","IRR",True, "Age_group", None, ["Age_group", "einddag_week"])
    make_scatterplot(df, "odds_ratio_V_2_N","odds_ratio_amc",True, "Age_group", None, ["Age_group", "einddag_week"])

    toelichting(df)

if __name__ == "__main__":
    #caching.clear_cache()
    #st.set_page_config(layout="wide")
    main()