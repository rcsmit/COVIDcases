import streamlit as st
from math import e
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import datetime as dt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import fisher_exact

#import random

# na 21/8 zelfde waardes voor de vaccinaties voor 90+ aangehoude
#@st.cache(ttl=60 * 60 * 24)
def read():
    """Read the data from a Google sheet
    original data from
    https://www.rivm.nl/coronavirus-covid-19/grafieken
    https://www.rivm.nl/covid-19-vaccinatie/cijfers-vaccinatieprogramma
    Returns:
        [type]: [description]
    """
    sheet_id = "12pLaItlz1Lw1BM-f1Zu66rq6nnXcw0qSOO64o3xuWco"
    sheet_name = "DATA2" # sheet copied as values
    url_data = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    url_pop = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/pop_size_age_NL2.csv"

    url_voll_vax="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/datum_50_pct_voll_gevaxx.csv"

    df_data = pd.read_csv(url_data, delimiter=',', error_bad_lines=False)

    df_data["einddag_week_"] = pd.to_datetime(df_data["einddag_week"], format="%Y-%m-%d" )


    df_pop = pd.read_csv(url_pop, delimiter=',', error_bad_lines=False)

    df_voll_vax = pd.read_csv(url_voll_vax, delimiter=',', error_bad_lines=False)

    df_voll_vax["datum_50_pct_voll_gevaxx"] = pd.to_datetime(df_voll_vax["datum_50_pct_voll_gevaxx"], format="%m/%d/%Y" )
    df  = pd.merge(
                df_data, df_pop, how="outer", on="Agegroup"
            )
    # corr_vax = st.sidebar.number_input("correction vax", value=1.0)
    # corr_unvax = st.sidebar.number_input("correction non vax", value=1.0)
    # df_data["TOTAAL sick vax"]  = df_data["TOTAAL sick vax"] * corr_vax
    # df_data["TOTAAL sick non vax"]  =df_data["TOTAAL sick non vax"] * corr_unvax
    df  = pd.merge(
                df, df_voll_vax, how="outer", on="Agegroup"
            )
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
            df[col] = df[col].fillna(0)
        except:
            print (f"omzetten {col} niet gelukt ")

    # print (df.dtypes)
    return df

def perc_gevacc():
    """Make a linegraph with the cumm-% of vaccinated people per age group
    """
    sheet_id = "12pLaItlz1Lw1BM-f1Zu66rq6nnXcw0qSOO64o3xuWco"
    sheet_name = "VACC__GRAAD" # sheet copied as values
    url_data = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df_data = pd.read_csv(url_data, delimiter=',', error_bad_lines=False)
    # print (df_data.dtypes)
    line_chart_pivot (df_data, None, "% Volledig gevaccineerd", False)

def line_chart (df, what_to_show):
    """Make a linechart from an unpivoted table, with different lines (agegroups)

    Args:
        df (dataframe): dataframe
        what_to_show ([type]): [description]
    """
    # fig = go.Figure()
    try:
        fig = px.line(df, x="einddag_week", y=what_to_show, color='Agegroup')
    except:
        fig = px.line(df, x="einddag_week", y=what_to_show)
    fig.update_layout(
        title=what_to_show,
        xaxis_title="Einddag vd week",
        yaxis_title=what_to_show,
    )
    st.plotly_chart(fig, use_container_width=True)

def line_chart_pivot (df_, field, title,sma):
    """Makes a linechart from a pivoted table, each column in a differnt line. Smooths the lines too.

    Args:
        df (dataframe): dataframe
        title ([type]): [description]
        sma(boolean) : show smooth averages?
    """
    if field is not None:
        df = make_pivot(df_, field)
    else:
        df = df_
    fig = go.Figure()

    columns = df.columns.tolist()
    columnlist = columns[1:]
    # st.write(columnlist)
    for col in columnlist:
        if sma:
            col_sma = col +"_sma"
            df[col_sma] =  df[col].rolling(window = 3, center = False).mean()
            fig.add_trace(go.Scatter(x=df["einddag_week"], y= df[col_sma], mode='lines', name=col ))
            title = title+ " (SMA 3)"
        else:
            fig.add_trace(go.Scatter(x=df["einddag_week"], y= df[col], mode='lines', name=col ))


    fig.update_layout(
        title=dict(
                text=title,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title=title    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander (f"dataframe pivottable {title}"):
        df_temp = df.astype(str).copy(deep = True)
        st.write (df_temp)

def line_chart_VE_as_index (df):
    """Makes a linechart from a pivoted table, each column in a differnt line. Smooths the lines too.

    Args:
        df (dataframe): dataframe
        title ([type]): [description]
        sma(boolean) : show smooth averages?
    """

    fig = go.Figure()

    columnlist = df.columns.tolist()
    title = "VE as index"
    # st.write(columnlist)
    for col in columnlist:
        fig.add_trace(go.Scatter( y= df[col], mode='lines', name=col ))

    fig.update_layout(
        title=dict(
                text=title,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),
        xaxis_title="Einddag vd week",
        yaxis_title=title    )
    st.plotly_chart(fig, use_container_width=True)

def find_slope_sklearn(df_temp, what_to_show_l, what_to_show_r, intercept_100):
    """Find slope of regression line - DOESNT WORK

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis
        intercept_100(boolean)) : intercept on (0,100) ie. VE starts at 100% ?
    Returns:
        [type]: [description]
    """
    x = np.array(df_temp[what_to_show_l]).reshape((-1, 1))
    y = np.array(df_temp[what_to_show_r])
    #obtain m (slope) and b(intercept) of linear regression line
    if intercept_100 :
        fit_intercept_=False
        i = 100
    else:
        fit_intercept_=True
        i = 0
    model = LinearRegression(fit_intercept=fit_intercept_)
    model.fit(x, y - i)
    m = model.coef_[0]
    b = model.intercept_+ i
    r_sq = model.score(x, y- i)
    return m,b,r_sq

def find_slope(df_temp, what_to_show_l, what_to_show_r):
    """Find slope and interception of regression line with polyfit. Not used

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis

    Returns:
        [type]: [description]
    """
    x_ = np.array(df_temp[what_to_show_l])
    y_ = np.array(df_temp[what_to_show_r])
    #obtain m (slope) and b(intercept) of linear regression line
    idx = np.isfinite(x_) & np.isfinite(y_)
    m, b = np.polyfit(x_[idx], y_[idx], 1)
    return m,b

def create_trendline(l,m,b, complete):
    """creates a dataframe with the values for a trendline. Apperentlu plotlyexpress needs a dataframe to plot sthg

    Args:
        l (int) : length
        m (float): slope
        b (float): intercept
        complete (boolean) : Show trendline until VE =0 or only the given dataset

    Returns:
        df: dataframe
    """
    t = []
    if complete == True:
        x_ = int(-100/m)+1
    else:
        x_ = int(l)

    for i in range (x_):
        j = m*i +b
        t.append([i,j])
    df_trendline = pd.DataFrame(t, columns = ['x', 'y'])
    return df_trendline
def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,  show_cat, categoryfield, hover_name, hover_data, intercept_100, complete, day_zero, a):
    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """
    with _lock:
        fig1xy,ax = plt.subplots()

        # if intercept_100 == False:
        #     try:

        #         x_ = np.array(df_temp[what_to_show_l])
        #         y_ = np.array(df_temp[what_to_show_r])
        #         #obtain m (slope) and b(intercept) of linear regression line
        #         idx = np.isfinite(x_) & np.isfinite(y_)
        #         m, b = np.polyfit(x_[idx], y_[idx], 1)
        #         model = np.polyfit(x_[idx], y_[idx], 1)

        #         predict = np.poly1d(model)
        #         r2 = r2_score  (y_[idx], predict(x_[idx]))
        #     except:
        #         m,b,model,predict,r2 =None,None,None,None,None

        #     try:

        #         fig3 = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data,
        #                             trendline="ols", trendline_scope = 'overall', trendline_color_override = 'black')
        #     except:
        #         # avoid exog contains inf or nans
        #         fig3 = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data)

        # else:


        #try:
        m,b,r2 = find_slope_sklearn(df_temp, what_to_show_l, what_to_show_r, intercept_100)

        l = 2 + df_temp["days_bweteen_50_pct_and_einddag"].max()
        fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data)
        df_trendline = create_trendline(l,m,b, complete)

        fig2 = px.line(df_trendline, x="x", y="y")
        fig2.update_traces(line=dict(color = 'rgba(50,50,50,0.8)'))
        #add linear regression line to scatterplot

        fig3 = go.Figure(data=fig1xy.data + fig2.data)
        correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)
        d_end = round(-100/m)

        d_half = round (-50/m)
        day_end = (day_zero+  pd.Timedelta(d_end, unit='D')   ).date()
        day_half = (day_zero+  pd.Timedelta(d_half, unit='D')   ).date()
        st.subheader(f"{a}")
        if a != "All" and a!= "All except 10-19 and 80+":
            st.write(f"Day zero = {day_zero.date()} | VE = 50% : {day_half} | VE = 0% :  {day_end}")
        else:
            st.write(f"Day zero = {day_zero.date()} | VE = 50% : {d_half} days | VE = 0% :  {d_end} days")

        title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

        fig3.update_layout(
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

        st.plotly_chart(fig3, use_container_width=True)
def make_calculations(df):
    # Vaccinatiestatus (nog) onbekend	(Nog) niet gevaccineerd	Niet volledig gevaccineerd	Volledig gevaccineerd
    # SICK_UNVAX	SICK_VAX	vacc_graad	VE	aantal mensen	anatal mensen non vax	aantal mensen vax	sick non vax	sick vax
    df["unvaxxed_new"] = (df["pop_size"]*(1-(df["vacc_graad"]/100))) #NB Unvaxxed is vanaf 2e vaccin
    df["vaxxed_new"] = df["pop_size"]* (df["vacc_graad"]/100) #NB Unvaxxed is vanaf 2e vaccin


    corr_vax = st.sidebar.number_input("correction vax", value=1.0)
    corr_unvax = st.sidebar.number_input("correction nonvax", value=1.0)
    df["SICK_VAX"]  = df["SICK_VAX"] * corr_vax
    df["SICK_UNVAX"]  =df["SICK_UNVAX"] * corr_unvax

    df["TOTAAL sick vax"]  = df["TOTAAL sick vax"] * corr_vax
    df["TOTAAL sick non vax"]  =df["TOTAAL sick non vax"] * corr_unvax
    df["TOTAAL sick vax no kids"]  = df["TOTAAL sick vax no kids"] * corr_vax
    df["TOTAAL sick non vax no kids"]  =df["TOTAAL sick non vax no kids"] * corr_unvax

    df["VE"] =(1-( (df["SICK_VAX"] /df["vaxxed_new"] ) / (df["SICK_UNVAX"]/ df["unvaxxed_new"])))*100

    df["VE TOTAAL"] =(1-( (df["TOTAAL sick vax"] /df["TOTAAL aantal mensen vax"] ) / (df["TOTAAL sick non vax"]/ df["TOTAAL aantal mensen non vax"])))*100
    df["VE ZONDER KIDS"] =(1-( (df["TOTAAL sick vax no kids"] /df["TOTAAL aantal mensen vax zonder kids"] ) / (df["TOTAAL sick non vax no kids"]/ df["TOTAAL aantal mensen non vax zonder kids"])))*100

    df["ODDS"] = (df["SICK_UNVAX"]/ df["unvaxxed_new"]) / (df["SICK_VAX"] /df["vaxxed_new"] )

    df["ODDS TOTAAL"] = (df["TOTAAL sick non vax"]/ df["TOTAAL aantal mensen non vax"]) / (df["TOTAAL sick vax"] /df["TOTAAL aantal mensen vax"] )
    df["ODDS ZONDER KIDS"] =  (df["TOTAAL sick non vax no kids"]/ df["TOTAAL aantal mensen non vax zonder kids"]) /  (df["TOTAAL sick vax no kids"] /df["TOTAAL aantal mensen vax zonder kids"] )

    dont_use = True
    if dont_use == True:
        df["unvaxxed_new"] = (df["pop_size"]*(1-(df["vacc_graad"]/100))) #NB Unvaxxed is vanaf 2e vaccin
        df["vaxxed_new"] = df["pop_size"]* (df["vacc_graad"]/100) #NB Unvaxxed is vanaf 2e vaccin

        df["healthy_vax"] =   df["vaxxed_new"]  - df["SICK_VAX"]
        df["healthy_nonvax"] =  df["unvaxxed_new"] - df["SICK_UNVAX"]

        # after second dose
        # https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/

        df["p_inf_vacc"] = df["SICK_VAX"] /  df["vaxxed_new"]
        df["p_inf_non_vacc"] = df["SICK_UNVAX"] / df["unvaxxed_new"]

        # df["fisher_oddsratio"], df["fisher_pvalue"] = fisher_exact(([df["SICK_VAX"], df["healthy_vax"]),
        #                                              (df["SICK_UNVAX"],      df["healthy_nonvax"])])

        # ODDS RATIO = IRR
        # VE = -100 * OR + 100
        df["VE_2_N"] = (1 - (   df["p_inf_vacc"]/df["p_inf_non_vacc"]))*100
        df["odds_ratio_V_2_N"] = (  df["p_inf_vacc"]/(1-  df["p_inf_vacc"])) /  (   df["p_inf_non_vacc"] / (1-   df["p_inf_non_vacc"]))
        # https://wikistatistiek.amc.nl/index.php/Logistische_regressie
        df["odds_ratio_amc"] = ( df["SICK_VAX"]*   df["healthy_nonvax"] ) / ( df["healthy_vax"] *  df["SICK_UNVAX"])

        df["IRR"] =  df["odds_ratio_V_2_N"] / ((1-df["p_inf_non_vacc"]) + (df["p_inf_non_vacc"] *  df["odds_ratio_V_2_N"] ))



        #df = calculate_fisher(df)
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

        a = df.iloc[i]["SICK_VAX"]
        b = df.iloc[i]["SICK_UNVAX"]
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
        df (dataframe): dataframe

    Returns:
        [type]: [description]
    """

    # The calculated odds ratio is different from the one R uses. The scipy
    # implementation returns the (more common) "unconditional Maximum
    # Likelihood Estimate", while R uses the "conditional Maximum Likelihood
    # Estimate".
    for i in range(len(df)):

        a = df.iloc[i]["SICK_VAX"]
        b = df.iloc[i]["SICK_UNVAX"]
        c = df.iloc[i]["healthy_vax"]

        d = df.iloc[i]["healthy_nonvax"]

        odds, p =  fisher_exact([[a,b],[c,d]])
        df.at[i, "fischer_odds"]= odds
        df.at[i, "fischer_p_val"]= p
    return df

def toelichting(df):
    st.write ("Gemaakt nav https://twitter.com/DennisZeilstra/status/1442121747361374217")
    # st.write("    unvaxxed_new = populatie_grootte -  sec_cumm")
    # st.write("    unboostered_new = populatie_grootte -  third_cumm")
    # st.write("    perc_sec_dose = round((sec_cumm / populatie_grootte)*100,1)")
    # st.write("    perc_boostered = round((third_cumm / populatie_grootte)*100,1)")
    # st.write("    ziek_V_2_per_100k = SICK_VAX /  sec_cumm *100_000")
    # st.write("    ziek_V_3_per_100k = positive_above_20_days_after_3rd_dose /  third_cumm *100_000")
    # st.write("    ziek_N_per_100k = SICK_UNVAX / unvaxxed_new *100_000")
    # st.write("    VE_2_N = (1 - ( ziek_V_2_per_100k/ ziek_N_per_100k))*100")
    # st.write("    VE_3_N = (1 - ( ziek_V_3_per_100k/ ziek_N_per_100k))*100")
    # st.write("    VE_3_N_2_N= (1 - ( ziek_V_3_per_100k/ ziek_V_2_per_100k))*100")
    # st.write("    healthy_vax =   sec_cumm - SICK_VAX")
    # st.write("    healthy_nonvax =  unvaxxed_new - SICK_UNVAX")
    st.write("")
    st.write("    after second dose")
    st.write("    https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/")
    st.write("")
    st.write("    p_inf_vacc = SICK_VAX /  sec_cumm")
    st.write("    p_inf_non_vacc = SICK_UNVAX / unvaxxed_new")
    st.write("")
    st.write("    ODDS RATIO = IRR")
    st.write("    VE = -100 * OR + 100")
    st.write("")
    st.write("    odds_ratio = (  p_inf_vacc/(1-  p_inf_vacc)) /  (   p_inf_non_vacc / (1-   p_inf_non_vacc))")
    st.write("    https://wikistatistiek.amc.nl/index.php/Logistische_regressie")
    st.write("    odds_ratio_amc = ( SICK_VAX*   healthy_nonvax ) / ( healthy_vax *  SICK_UNVAX)")
    st.write("    IRR =  odds_ratio / ((1-p_inf_non_vacc) + (p_inf_non_vacc *  odds_ratio )) Incidence Rate Ratio")
    st.write("    https://www.rivm.nl/covid-19-vaccinatie/cijfers-vaccinatieprogramma")
    st.write("    https://www.rivm.nl/coronavirus-covid-19/grafieken")
    st.write("    https://docs.google.com/spreadsheets/d/12pLaItlz1Lw1BM-f1Zu66rq6nnXcw0qSOO64o3xuWco/edit#gid=1548858810")
    #st.write(df)

def group_table(df, valuefield):
    """Group the table by -valuefield-

    Args:
        df (dataframe): dataframe
        valuefield ([type]): [description]

    Returns:
        [type]: [description]
    """

    df = df[df["Agegroup"] != "0-19"]

    df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
    return df_grouped

def make_pivot(df, valuefield):
    """Pivot the table with valuefield as field

    Args:
        df (dataframe): dataframe
        valuefield ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = df[df["Agegroup"] != "0-9"]
    df = df[df["Agegroup"] != "10-19"]
    df_pivot = (
    pd.pivot_table(
        df,
        values=valuefield,
        index=["einddag_week"],
        columns=["Agegroup"],
        aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    return df_pivot

def normeren(df):

    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""

    what_to_norm_ = list(df.columns.values)
    what_to_norm = what_to_norm_[1:]

    # print(df.dtypes)
    how_to_norm_ = ["index", "rel"]
    how_to_norm_ = ["index"]

    normed_columns = []
    for how_to_norm in how_to_norm_:

        for column in what_to_norm:
            maxvalue = (df[column].max())
            firstvalue = df[column].iloc[0]

            for i in range(len(df)):
                if how_to_norm == "max":
                    name = f"{column}_normed"
                    df.loc[i, name] = df.loc[i, column] / maxvalue
                elif how_to_norm == "index":
                    name = f"{column}_indexed"
                    df.loc[i, name] = df.loc[i, column] / firstvalue * 100
                elif how_to_norm == "rel":
                    name = f"{column}_relative"
                    df.loc[i, name] = (df.loc[i, column] - firstvalue) / firstvalue * 100
            normed_columns.append(name)
            # print(f"{name} generated")
    return df, normed_columns

def show_tabel_waningtime(df, agegroups, intercept_100):
    """Show a table with the waningtime for the agegroups
    Args:
        df (dataframe): dataframe
        agegroups (list): list with the agegroups to show
        intercept_100 (boolean): intercept at (0,100)?
    """
    table, table_bar=[],[]

    for a in agegroups:
        df_ =  df[df["Agegroup"] == a]
        m,b,r2 = find_slope_sklearn(df_, "days_bweteen_50_pct_and_einddag", "VE",intercept_100)
        #m,b = find_slope_sklearn(df_, "days_bweteen_50_pct_and_einddag", "VE")
        table.append([a,round(m,2),round (b,2),round(100/(m*-1),1)])
        table_bar.append([a,round(100/(m*-1),1)])
        #st.write(f"{a} - {round(m,2)} - {round (b,2)} - waning in {round(100/(m*-1),1)} days")
    df_result = pd.DataFrame(table, columns = ['Agegroup', 'slope', 'intercept', 'waning_in_x_days'])
    df_result_bar = pd.DataFrame(table_bar, columns = ['Agegroup', 'waning_in_x_days'])
    st.write(df_result)

def VE_door_tijd(df):
    """Calculate the VE in time per agegroup
    Args:
        df (dataframe): dataframe
    """
    df =df.sort_values(by=['einddag_week'])
    df = df[df["datum_50_pct_voll_gevaxx"] !=0]
    df["datum_50_pct_voll_gevaxx"] = pd.to_datetime(df["datum_50_pct_voll_gevaxx"], format="%m/%d/%Y" )
    df['days_bweteen_50_pct_and_einddag'] = (df['einddag_week_'] - df['datum_50_pct_voll_gevaxx']).dt.days #//np.timedelta64(1,'D')    #curr_time
    df = df[df["days_bweteen_50_pct_and_einddag"] >=0]
    agegroups =  df["Agegroup"].drop_duplicates().sort_values().tolist()
    days_zero =  df["datum_50_pct_voll_gevaxx"].drop_duplicates().sort_values(ascending=False).tolist()
    intercept_100 = st.sidebar.selectbox("Intercept = 100", [True, False], index=0)
    complete = st.sidebar.selectbox("Trendline until y=0", [True, False], index=0)
    what_to_list = ["All", "All except 10-19 and 80+", "All seperately"] + agegroups
    what_to_show = st.sidebar.selectbox("What to show", what_to_list, index=0)
    categoryfield =  st.sidebar.selectbox("Categoryfield", ["Agegroup","einddag_week"], index=0)

    if what_to_show == "All":
        day_zero = days_zero[-1]
        make_scatterplot(df, "days_bweteen_50_pct_and_einddag", "VE" , None, categoryfield, "Agegroup", ["Agegroup", "datum_50_pct_voll_gevaxx", "VE"], intercept_100, complete, day_zero, what_to_show)
        #make_scatterplot(df, "einddag_week", "VE" , None, categoryfield, "Agegroup", ["Agegroup", "datum_50_pct_voll_gevaxx", "VE"], intercept_100, complete, day_zero, what_to_show)
        show_tabel_waningtime(df,agegroups, intercept_100)
    elif what_to_show == "All except 10-19 and 80+":
        day_zero = days_zero[-1]
        #80+ verwijderen (is uitschieter)
        df = df[df["Agegroup"] != "80+"]
        df = df[df["Agegroup"] != "10-19"]
        make_scatterplot(df, "days_bweteen_50_pct_and_einddag", "VE" , None, categoryfield, "Agegroup", ["Agegroup", "datum_50_pct_voll_gevaxx", "VE"], intercept_100, complete, day_zero, what_to_show)

    elif what_to_show == "All seperately":
        for a,day_zero in zip(agegroups,days_zero):
            df_ = df[df["Agegroup"] ==  a ]
            make_scatterplot(df_, "days_bweteen_50_pct_and_einddag", "VE" , None, "einddag_week", "Agegroup", ["Agegroup", "datum_50_pct_voll_gevaxx", "VE"], intercept_100, complete, day_zero, a)
    else:
        day_zero = days_zero[agegroups.index(what_to_show)]
        df = df[df["Agegroup"] ==  what_to_show ]
        make_scatterplot(df, "days_bweteen_50_pct_and_einddag", "VE" , None, "einddag_week", "Agegroup", ["Agegroup", "datum_50_pct_voll_gevaxx", "VE"], intercept_100, complete, day_zero, what_to_show)


def show_VE_totaal(df):
    """Calculate VE for the total population, with and without kids in time

    Args:
        df (dataframe): dataframe
    """


    def VE_low (df):
        # Taylor series
        # https://apps.who.int/iris/bitstream/handle/10665/264550/PMC2491112.pdf
        a_b = df["TOTAAL aantal mensen vax"]
        c_d = df["TOTAAL aantal mensen non vax"]
        a = df["TOTAAL sick vax"]
        c= df["TOTAAL sick non vax"]
        Za2 = 1.96

        ARV = a / (a_b)
        ARU = c/ (c_d)
        RR_taylor = ARV/ARU
        zzz = ((1-ARV)/a)+((1-ARU)/c)
        zzz = 1/a + 1/c
        ci_low_taylor =( 1 - round( np.exp(np.log(RR_taylor) +Za2 * math.sqrt(zzz)),4))*100
        return ci_low_taylor

    def VE_high (row):
        # Taylor series
        # https://apps.who.int/iris/bitstream/handle/10665/264550/PMC2491112.pdf
        a_b = row["TOTAAL aantal mensen vax"]
        c_d = row["TOTAAL aantal mensen non vax"]
        a = row["TOTAAL sick vax"]
        c= row["TOTAAL sick non vax"]
        Za2 = 1.96

        ARV = a / (a_b)
        ARU = c / (c_d)
        RR_taylor =  ARV/ARU

        zzz = ((1-ARV)/a)+((1-ARU)/c)
        zzz = 1/a + 1/c
        ci_high_taylor = (1- round( np.exp(np.log(RR_taylor) -Za2 * math.sqrt(zzz)),4))*100

        return ci_high_taylor
    df_VE_totaal = df[df["Agegroup"] == "80+"]

    df_VE_totaal['VE_totaal_low'] = df_VE_totaal.apply ( VE_low, axis=1)

    df_VE_totaal['VE_totaal_high'] = df_VE_totaal.apply (VE_high, axis=1)
    #df_VE_totaal = df_VE_totaal[["einddag_week_", "VE TOTAAL", "VE_totaal_low", "VE_totaal_high", "VE ZONDER KIDS" ]]
    df_VE_totaal_ = df_VE_totaal[["einddag_week_", "VE TOTAAL",  "VE ZONDER KIDS" ]]
    df_odds_totaal_ = df_VE_totaal[["einddag_week_", "ODDS TOTAAL",  "ODDS ZONDER KIDS" ]]

    linechart_columns_as_lines(df_VE_totaal_,"VE",[0,100])
    st.subheader("... times the change that a vaccinated persion gets sick than an unvaccineted")
    linechart_columns_as_lines(df_odds_totaal_,"inverse odds", None)

def linechart_columns_as_lines(df_VE_totaal, yaxtitle, range_):
    fig = go.Figure()

    columns = df_VE_totaal.columns.tolist()
    columnlist = columns[1:]

    for col in columnlist:
        fig.add_trace(go.Scatter(x=df_VE_totaal["einddag_week_"], y= df_VE_totaal[col], mode='lines', name=col ))


    fig.update_layout(
        yaxis={"range":range_},
        title=dict(
                text=f"{yaxtitle} Gehele bevolking door de tijd heen",
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title=yaxtitle    )
    st.plotly_chart(fig, use_container_width=True)

def referentie_andere_studies():
    sheet_id = "12pLaItlz1Lw1BM-f1Zu66rq6nnXcw0qSOO64o3xuWco"
    sheet_name = "OTHER_STUDIES"
    url_data = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df_data = pd.read_csv(url_data, delimiter=',')
    st.subheader("Other studies")

    #print (df_data)
    df_data = df_data[["NAME", "14", "30", "45", "60", "90", "120", "150", "180", "210" ]]
    df_data = df_data.transpose()# .reset_index()
    df_data.columns = df_data.iloc[0]
    print (df_data)
    df_data = df_data.iloc[1: , :]
    #df_data.drop(["NAME"], axis=0)
    print (df_data)
    df_data = df_data.reset_index()

    print (df_data)
    df_data['index'].astype(str).astype(int)
    print (df_data)
    print (df_data.dtypes)

    quit()
    fig = go.Figure()

    columnlist = df_data.columns.tolist()
    title = "VE as index"
    # st.write(columnlist)
    for col in columnlist:
            fig.add_trace(go.Scatter(x=df_data["NAME"], y= df_data[col], mode='lines', name=col ))


    fig.update_layout(
        title=dict(
                text=title,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title=title    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    df_ = read()
    df_ = df_.fillna(0)
    df = make_calculations(df_)

    st.header("VE in NL")
    st.write("Article : https://rcsmit.medium.com/calculation-of-waning-of-vaccin-effectiveness-in-the-netherlands-82957de37ea3")
    st.write("Attention: very rough estimation due the high number of unknown vaccin statusses. Assumed is that they have the same ratio as the known numbers.")

    st.write("Vaccination% of 11-17 is linked to the cases10-19, 18-30 to 20-29. ")
    show_VE_totaal(df)
    VE_door_tijd(df)

    line_chart (df, "VE_2_N")
    line_chart (df, "odds_ratio_V_2_N")

    df_pivot = make_pivot(df, "VE_2_N").copy(deep = True)
    df_pivot, normed_columns = normeren(df_pivot)
    df_pivot = df_pivot[normed_columns]
    line_chart_VE_as_index (df_pivot)
    # st.write(df_pivot)

    #line_chart_pivot (df_pivot,  "VE_2_N", "VE (2 vaccins / N)", None)
    # line_chart_pivot ( df,"odds_ratio_V_2_N", "Odds Ratio (2 vaccins / N)")
    # line_chart (df, "fischer_p_val")
    # st.subheader("All ages together (excl. 0-19)")

    # group table, all age groups in one total
    #df_grouped = group_table(df_, "einddag_week").copy(deep = True)
    #df_grouped = make_calculations(df_grouped)

    # line_chart (df_grouped,  "VE_2_N")
    # line_chart (df_grouped,  "IRR")
    # line_chart ( df_grouped,"odds_ratio_V_2_N")
    # plot_line_with_ci(df_grouped, "Odds Ratio", "CI_OR_low", "or_fisher_2", "CI_OR_high")
    # plot_line_with_ci(df_grouped, "Relative Risk",  "CI_rel_risk_low", "rel_risk", "CI_rel_risk_high")

    # line_chart ( df_grouped,"fischer_p_val")

    perc_gevacc()
    st.write ("eg. 30-39 = 30-34 and 30-39.1 = 35-39")

    toelichting(df)

if __name__ == "__main__":
    #caching.clear_cache()
    #st.set_page_config(layout="wide")
    main()
    #referentie_andere_studies()