# from mortality_yearly_per_capita import get_bevolking
import streamlit as st
from typing import List, Tuple
import pandas as pd
# import platform
import random
import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np
from covid_dashboard_rcsmit import find_lag_time
import plotly.express as px

from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats
# from oversterfte_compleet import  get_sterftedata, get_data_for_series_wrapper,make_df_quantile #, layout_annotations_fig
# for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from statsmodels.tools.tools import add_constant
# from oversterfte_eurostats_maand import get_data_eurostat
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from utils import  get_sterfte, get_rioolwater,get_rioolwater_oud, get_vaccinaties, get_oversterfte, get_ziekenhuis_ic, get_maandelijkse_overlijdens, from_week_to_month


try:
    st.set_page_config(layout="wide")
except:
    pass
# WAAROM TWEE KEER add_custom_age_group_deaths ??? TODO


def compare_rioolwater(rioolwater):
    st.subheader("compare the rioolwater given by RIVM (x)  and calculated from the file with various meetpunten (y)")
    rioolwater_oud =  get_rioolwater_oud()


    # compare the rioolwater given by RIVM and calculated from the file with various meetpunten
   
    rw = pd.merge(rioolwater,rioolwater_oud, on=["jaar", "week"])

    rw["YearWeekISO"] = rw["jaar"].astype(str) +"-W" +rw["week"].astype(str)
    line_plot_2_axis(rw,"YearWeekISO","RNA_flow_per_100000_x","RNA_flow_per_100000_y","TOTAL")
    rw = from_week_to_month(rw,"mean")
    line_plot_2_axis(rw,"YearMonth","RNA_flow_per_100000_x","RNA_flow_per_100000_y","TOTAL")

def multiple_linear_regression(df: pd.DataFrame, x_values: List[str], y_value_: str, age_sex: str, normalize:bool):
    """
    Perform multiple linear regression and display results.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_values (List[str]): List of independent variable column names.
        y_value (str): Dependent variable column name.
    Returns:
        Datadict :  C_RNA':model.params["RNA_flow_per_100000"],
                    'C_vacc':model.params["TotalDoses"],
                    'P_const': model.pvalues["const"],
                    'P_RNA':model.pvalues["RNA_flow_per_100000"],
                    'P_vacc':model.pvalues["TotalDoses"],
                    'R-squared': [model.rsquared], #* len(model.params),
                    'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
                    'F-statistic': [model.fvalue], # * len(model.params),
                    'F-statistic P-value':
    """
    st.subheader("Multiple Lineair Regression")
    standard=  False#  st.sidebar.checkbox("Standardizing dataframe", True)
    intercept=  True# st.sidebar.checkbox("Intercept", False)
    only_complete = False # st.sidebar.checkbox("Only complete rows", False)
    if only_complete:
        df=df.dropna()
    else:
        df = df.dropna(subset=x_values)
        df = df.dropna(subset=y_value_)

    x = df[x_values]
    y = df[y_value_]

    if normalize:

        # Normalize each feature in x to the range [0, 1]
        x_normalized = (x - x.min()) / (x.max() - x.min())

        # If intercept is required, add a constant term
        if intercept:
            x_normalized = sm.add_constant(x_normalized)  # adding a constant
    
        # Fit the OLS model using the normalized data
        model = sm.OLS(y, x_normalized).fit()
    else:
        if intercept:
            x= sm.add_constant(x) # adding a constant

        model = sm.OLS(y, x).fit()
    #predictions = model.predict(x)
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    #col1,col2=st.columns(2)
    # with col1:
    #     st.write("**Model**")
    #     print_model = model.summary()
    #     st.write(print_model)
    # with col2:
    robust_model = model.get_robustcov_results(cov_type='HC0')  # You can use 'HC0', 'HC1', 'HC2', or 'HC3'
    st.write("**robust model**")
    st.write(robust_model.summary())
    col1,col2,col3=st.columns([2,1,1])
   
    with col1:
        st.write("**Correlation matrix**")
        correlation_matrix = x.corr()
        st.write(correlation_matrix)
    with col2:
        # Calculate VIF for each variable

        # VIF = 1: No correlation between the variable and others.
        # 1 < VIF < 5: Moderate correlation, likely acceptable.
        # VIF > 5: High correlation, indicating possible multicollinearity.
        # VIF > 10: Strong multicollinearity, action is needed.
        st.write("**VIF**")
        vif = pd.DataFrame()
        vif["Variable"] = x.columns
        vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        st.write(vif)
    with col3:
        U, s, Vt = np.linalg.svd(x)
        st.write("**spread of singular values**")
        st.write(s)  # Look at the spread of singular values
    data_dict = {
        'Y value':f"{y_value_}_{age_sex}",
        # 'Coefficients': model.params,
        # 'P-values': model.pvalues,
        # 'T-values': model.tvalues,
        # 'Residuals': model.resid,
        'C_RNA':model.params["RNA_flow_per_100000"],
        'C_vacc':model.params["TotalDoses"],
        'P_const': model.pvalues["const"],
        'P_RNA':model.pvalues["RNA_flow_per_100000"],
        'P_vacc':model.pvalues["TotalDoses"],
        'R-squared': model.rsquared, #* len(model.params),
        'Adjusted R-squared': model.rsquared_adj, # * len(model.params),
        'F-statistic': model.fvalue, # * len(model.params),
        'F-statistic P-value': model.f_pvalue, # * len(model.params)
    } 
    
    return data_dict


def make_scatterplot(df: pd.DataFrame, x: str, y: str, age_sex: str):
    """
    Create and display a scatterplot.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y (str): Y-axis column name.
        age_sex (str): Age and sex group.
    """
    df[x]=df[x].astype(float)
    df[y]=df[y].astype(float)
    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    r_squared = r_value ** 2
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(df[x], df[y])[0, 1]

    title_ = f"{age_sex} - {x} vs {y} [n = {len(df)}]"
    r_sq_corr = f'R2 = {r_squared:.2f} / Corr coeff = {correlation_coefficient:.2f}'
    try:
        fig = px.scatter(df, x=x, y=y,  hover_data=['jaar','week'],   title=f'{title_} ||<br> {r_sq_corr}')
    except:
        fig = px.scatter(df, x=x, y=y,    title=f'{title_} ||<br> {r_sq_corr}')
    fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    key=str(int(random.randint(1,500000)))
    r = random.randint(1,50)
    st.plotly_chart(fig, key=key)

def line_plot_2_axis(df: pd.DataFrame, x: str, y1: str, y2: str, age_sex: str, period=None):
    """
    Create and display a line plot with two y-axes.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y1 (str): First y-axis column name.
        y2 (str): Second y-axis column name.
        age_sex (str): Age and sex group. (for the title)
    """
    
    # Create a figure
    fig = go.Figure()
    
    # Add OBS_VALUE as the first line on the left y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y1],
            mode='lines',
            name=y1,
            line=dict(color='blue')
        )
    )
    if y1=="base_value":

        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y2],
                mode='lines',
                name=y2,
                line=dict(color='red'),
              
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y2],
                mode='lines',
                name=y2,
                line=dict(color='red'),
                yaxis='y2'
            )
        )

    title = f"{age_sex} - {x} vs<br>{y1} and {y2}"
    if period is not None:
        title+= f"<br>{period}"
    # Update layout to include two y-axes
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y1,
        yaxis2=dict(
            title=y2,
            overlaying='y',
            side='right'
        ),
        showlegend=False
        
    )
    #legend=dict(x=0.5, y=1, orientation='h')
    # Show the figure
    if 1==2:
        
        try:
            if 2020 in df["jaar"].values:
                fig.add_vrect(
                    x0="2020-W13",
                    x1="2020-W18",
                    annotation_text="Eerste golf",
                    annotation_position="top left",
                    fillcolor="pink",
                    opacity=0.25,
                    line_width=0,
                    )
                fig.add_vrect(
                    x0="2020-W39",
                    x1="2021-W03",
                    annotation_text="Tweede golf",
                    annotation_position="top left",
                    fillcolor="pink",
                    opacity=0.25,
                    line_width=0,
                )


                # hittegolven
                fig.add_vrect(
                    x0="2020-W33",
                    x1="2020-W34",
                    annotation_text=" ",
                    annotation_position="top left",
                    fillcolor="yellow",
                    opacity=0.35,
                    line_width=0,
                )
                fig.add_vrect(
                    x0="2020-W01",
                    x1="2020-W52",
                    fillcolor="grey",
                    opacity=0.1,
                    line_width=0,
                )
            if 2021 in df["jaar"].values:
                fig.add_vrect(
                    x0="2021-W33",
                    x1="2021-W52",
                    annotation_text="Derde golf",
                    annotation_position="top left",
                    fillcolor="pink",
                    opacity=0.25,
                    line_width=0,
                )
                fig.add_vrect(
                    x0="2021-W01",
                    x1="2021-W02",
                    fillcolor="grey",
                    opacity=0.35,
                    line_width=0,
                )
        
            if 2022 in df["jaar"].values:
            
                fig.add_vrect(
                    x0="2022-W32",
                    x1="2022-W33",
                    annotation_text=" ",
                    annotation_position="top left",
                    fillcolor="yellow",
                    opacity=0.35,
                    line_width=0,
                )
                fig.add_vrect(
                    x0="2022-W01",
                    x1="2022-W52",
                    fillcolor="grey",
                    opacity=0.1,
                    line_width=0,
                )
            if 2023 in df["jaar"].values:
            
                fig.add_vrect(
                    x0="2023-W23",
                    x1="2023-W24",
                    annotation_text=" ",
                    annotation_position="top left",
                    fillcolor="yellow",
                    opacity=0.35,
                    line_width=0,
                )
                fig.add_vrect(
                    x0="2023-W36",
                    x1="2023-W37",
                    annotation_text="Geel = Hitte golf",
                    annotation_position="top left",
                    fillcolor="yellow",
                    opacity=0.35,
                    line_width=0,
                )
                
            if 2024 in df["jaar"].values:
                # geen hittegolf in 2024
                fig.add_vrect(
                    x0="2024-W01",
                    x1="2024-W39",
                    fillcolor="grey",
                    opacity=0.1,
                    line_width=0,
                )
            
        except Exception as e:
            print ("Error in annotations {e}")
    key=str(int(random.random()*100000))
    st.plotly_chart(fig, key=key)

def analyse_maandelijkse_overlijdens(oorzaak, age_sex, df_result, time_period, seizoen, maand, normalize):
    """_summary_

    Args:
        oorzaak (_type_): _description_
        age_sex (_type_): _description_
        df_result (_type_): _description_
        time_period (_type_): _description_
        seizoen (bool): _description_
        maand (bool): _description_

    Returns:
        _type_: _description_
    """    
    
    df_result_month = from_week_to_month(df_result,"sum")
    #df_result_month = df_result_month[df_result_month["jaar"] != 2024]
    df_hartvaat = get_maandelijkse_overlijdens(oorzaak)
    
    df_month = pd.merge(df_result_month, df_hartvaat, on="YearMonth", how="inner") 
    df_month["maand"] = (df_month["YearMonth"].str[5:]).astype(int)
    
    data_dict,_,_,_,_ = perform_analyse(age_sex, df_month, time_period, "RNA_flow_per_100000","TotalDoses",f"OBS_VALUE_{oorzaak}", seizoen, maand, normalize)
    return data_dict
def perform_analyse(age_sex, df, time_period,x1,x2,y, seizoen, maand, normalize, only_graph=False):
    """_summary_

    Args:
        age_sex (_type_): _description_
        df (_type_): _description_
        time_period (_type_): _description_
        x1 (_type_): _description_
        x2 (_type_): _description_
        y (_type_): _description_
        seizoen (_type_): _description_
        maand (_type_): _description_
        normalize
        only graph
    Returns:
        tuple: data_dict
        
                    C_RNA':model.params["RNA_flow_per_100000"],
                    'C_vacc':model.params["TotalDoses"],
                    'P_const': model.pvalues["const"],
                    'P_RNA':model.pvalues["RNA_flow_per_100000"],
                    'P_vacc':model.pvalues["TotalDoses"],
                    'R-squared': [model.rsquared], #* len(model.params),
                    'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
                    'F-statistic': [model.fvalue], # * len(model.params),
                    'F-statistic P-value':
                max_lag,max_corr,max_lag_sma,max_corr_sma
    """    
    # Voeg een sinus- en cosinusfunctie toe om seizoensinvloeden te modelleren
    try:           
        df['sin_time'] = np.sin(2 * np.pi * df['maand']/ 12)
        df['cos_time'] = np.cos(2 * np.pi * df['maand'] / 12)
        m= True
    except:
       
        df['sin_time'] = np.sin(2 * np.pi * df['week']/ 52)
        df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
        m=False
        
    x_values = [x1,x2] # + 
    if seizoen:
        x_values += ['sin_time', 'cos_time']
    if maand:
        if m:
            x_values += ['maand']
        else:
            x_values += ['week']
    y_value_ = y
    if only_graph:
        line_plot_2_axis(df, time_period,y_value_, x1,age_sex)
        max_lag,max_corr,max_lag_sma,max_corr_sma=None,None,None,None
    else:
        #col1,col2=st.columns(2)
        col1,col2,col3=st.columns(3)
        with col1:
            line_plot_2_axis(df, time_period,y_value_, x1,age_sex)
            make_scatterplot(df, y_value_, x1,age_sex)
    
        with col2:
            line_plot_2_axis(df, time_period,y_value_, x2,age_sex)
            make_scatterplot(df, y_value_, x2,age_sex)
        with col3:
            line_plot_2_axis(df, time_period,x2, x1,age_sex)
            make_scatterplot(df, x1, x2,age_sex)
        max_lag,max_corr,max_lag_sma,max_corr_sma = find_lag_time(df, x1, y_value_, -14, 14)
    
    try:
        data_dict = multiple_linear_regression(df,x_values,y_value_, age_sex, normalize)
    except Exception as e:
        data_dict = None
        st.write(f"error {e}")

        
    return data_dict,max_lag,max_corr,max_lag_sma,max_corr_sma

def main():
    st.subheader("Relatie sterfte/rioolwater/vaccins")
    st.info("Inspired by https://www.linkedin.com/posts/annelaning_vaccinatie-corona-prevalentie-activity-7214244468269481986-KutC/")
    
    opdeling = [[0, 120], [15, 17], [18, 24], [25, 49], [50, 59], [60, 69], [70, 79], [80, 120]]
    col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="center")
    
    with col1:
        fixed_periods = st.checkbox("Fixed periods", True)
    
    if not fixed_periods:
        with col2:
            start_week = st.number_input("Startweek", 1, 52, 1)
        with col3:
            start_jaar = st.number_input("Startjaar", 2020, 2024, 2020)
        with col4:
            eind_week = st.number_input("Eindweek", 1, 52, 52)
        with col5:
            eind_jaar = st.number_input("Eindjaar", 2020, 2024, 2024)
        
        pseudo_start_week = start_jaar * 52 + start_week
        pseudo_eind_week = eind_jaar * 52 + eind_week
        
        if pseudo_start_week >= pseudo_eind_week:
            st.error("Eind kan niet voor start")
            st.stop()
    
    if not fixed_periods:
        col1, col2, col3, col4,col5,col6 = st.columns(6, vertical_alignment="center")
        with col1:
            y_value = st.selectbox("Y value/left ax", ["OBS_VALUE", "oversterfte", "p_score", "Hospital_admission",  "IC_admission"], 0, help="Alleen bij leeftijdscategorieen")
        with col2:
            normalize = st.checkbox("Normaliseer X values", True, help="Normalizeren omdat de vaccindosissen een hoog getal kunnen zijn")
        with col3:
            seizoen = st.checkbox("Seizoensinvloeden meenemen", True)
        with col4:
            maand = st.checkbox("Maand-/week invloeden meenemene")
        with col5:
            shift_weeks = st.slider(f"Shift {y_value}", -52,52,0)
        with col6:
            window = st.slider(f"SMA window {y_value}", 1,52,1)
    #else:
     
    df = get_sterfte(opdeling)
    df_rioolwater = get_rioolwater()
    df_vaccinaties = get_vaccinaties()
    df_oversterfte = get_oversterfte(opdeling)
    df_ziekenhuis_ic = get_ziekenhuis_ic()

    df_oversterfte["age_sex"] = df_oversterfte["age_sex"].replace("Y0-120_T", "TOTAL_T")
    
    df_merged = (
        pd.merge(df, df_rioolwater,  on=["jaar", "week"], how="left")
        .merge(df_ziekenhuis_ic, on=["jaar", "week"], how="left")
        .merge(df_vaccinaties, on=["jaar", "week", "age_sex"], how="left")
        .fillna(0).infer_objects(copy=False)
        .merge(df_oversterfte, on=["jaar", "week", "age_sex"], how="left")
    )
    
    df_merged["pseudoweek"] = df_merged["jaar"] * 52 + df_merged["week"]
          
    if fixed_periods:
        periods = [
            [1, 2020, 26, 2021],
            [27, 2021, 26, 2022],
            [27, 2022, 26, 2023],
            [27, 2023, 52, 2024],
            # [1,2022,52,2023],
            # [1,2020,52,2024]
        ]
        results = []
        col=[None,None,None,None]
        options = ["OBS_VALUE",  "oversterfte", "p_score", "Hospital_admission",  "IC_admission", "RNA_flow_per_100000","TotalDoses"]
        secondary_ax = st.selectbox("Right axis", options,5)
        age_sex = "TOTAL_T"
        #with st.expander("results"):
        if 1==1:
            for n,what in enumerate(options):
                if what != secondary_ax:    
                    st.header(what)
                    df_filtered = make_df_filtered(df_merged, age_sex, what, 0, 1, 1, 2020, 52, 2024)
                  
                    line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", what, secondary_ax, age_sex, )
            
                    m=0

                    col[0],col[1],col[2],col[3] = st.columns(4)
                    for start_wk, start_yr, end_wk, end_yr in periods:          
                        
                        df_filtered = make_df_filtered(df_merged, age_sex,what, 0, 1, start_wk, start_yr, end_wk, end_yr)
                        with col[m]:
                            period=f"{start_wk}-{start_yr} - {end_wk}-{end_yr}"
                            
                            corr= df_filtered[what].corr(df_filtered[secondary_ax])
                            max_lag,max_corr,max_lag_sma,max_corr_sma = find_lag_time(df_filtered, what, secondary_ax,-14,14,verbose=False)
                            line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", what, secondary_ax, age_sex, period=period )
                        
                        
                        m+=1
                        
                        result = {
                            "period": period,
                            "Primary ax": what,
                            "Secondary ax": secondary_ax,
                            "corr_coeff": round(corr,4),
                            "max_lag_days": max_lag,
                            "max_corr": max_corr,
                            "max_lag_days_sma_(7)": max_lag_sma,
                            "max_corr_sma_(7)": max_corr_sma
                        }
                        
                        # Append the result dictionary to the results list
                        results.append(result)

        # Convert the results list to a dataframe
        df_results = pd.DataFrame(results)

        # Display the resulting dataframe

        st.subheader("Results")
        st.write(df_results)
        
    else:
        # not a loop of fixed periods. Just one period
        df_period = df_merged[(df_merged["pseudoweek"] >= pseudo_start_week) & (df_merged["pseudoweek"] <= pseudo_eind_week)]
        df_period = df_period[df_period["week"] != 53]
        age_sex = "TOTAL_T"
        df_filtered = df_period[df_period["age_sex"] == age_sex]
        df_filtered[y_value] = df_filtered[y_value].rolling(window=window, center=True).mean()
        df_filtered[y_value] = df_filtered[y_value].shift(shift_weeks)
                
        with st.expander("Rioolwater"):
            compare_rioolwater(df_rioolwater)
        
        with st.expander("OBS VALUE - oversterfte - Pvalue"):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "OBS_VALUE", "oversterfte", age_sex)
                make_scatterplot(df_filtered, "OBS_VALUE", "oversterfte", age_sex)
            with col2:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "OBS_VALUE", "p_score", age_sex)
                make_scatterplot(df_filtered, "OBS_VALUE", "p_score", "")
            with col3:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "base_value", "OBS_VALUE", age_sex)
                make_scatterplot(df_filtered, "base_value", "OBS_VALUE", age_sex)
        
        # Analyze based on age groups and causes
        age_sex_list = ["TOTAL_T"] if y_value in ["oversterfte", "p_score"] else df["age_sex"].unique().tolist()
        
        results = []
        df_complete=pd.DataFrame()
        for age_sex in age_sex_list:
            df_result = df_period[df_period["age_sex"] == age_sex].copy()
            df_result["TotalDoses"].fillna(0)
            
            if age_sex == "TOTAL_T":
                for oorzaak in ["hart_vaat_ziektes", "covid", "ademhalingsorganen", "accidentele_val", "wegverkeersongevallen", "nieuwvormingen"]:
                    if df_result["TotalDoses"].sum() != 0:
                        with st.expander(oorzaak):
                            st.subheader(f"TOTAL overlijdens {oorzaak} vs rioolwater en vaccinaties")
                            
                            df_iteration = analyse_maandelijkse_overlijdens(oorzaak, age_sex, df_result, "YearMonth", seizoen, maand, normalize)
                            results.append(df_iteration)
            
            time_period = "YearMonth" if maand else "TIME_PERIOD_x"
            #st.write(df_result)
            if df_result["TotalDoses"].sum() != 0:
                with st.expander(f"{age_sex} - Alle overlijdensoorzaken"):
                    st.subheader(f"{age_sex} - Alle overlijdensoorzaken")
                    st.write(df_result)
                    #df_result.to_csv(f"{age_sex}")
                    data_dict,_,_,_,_ = perform_analyse(age_sex, df_result, time_period, "RNA_flow_per_100000", "TotalDoses", y_value, seizoen, maand, normalize)
                    
                    xx=data_dict["Y value"]
                    y_value_x =  f"{xx}_{age_sex}"
                    # df_iteration = pd.DataFrame({
                    #     "Y value":y_value_x,
                    
                    #     #'P_const': data['P_const'],
                    #     'coef_RNA':data_dict['C_RNA'],
                    #     'coef_vacc':data_dict['C_vacc'],

                    #     'p_RNA':data_dict['P_RNA'],
                    #     'p_vacc':data_dict['P_vacc'],

                    #     #"R-squared": data["R-squared"],
                    #     "Adj. R2": data_dict["Adjusted R-squared"],
                    #     "F-stat.": data_dict["F-statistic"],
                    #     "p_F-stat.": data_dict["F-statistic P-value"]
                    # })

                    
                    # Append the DataFrame to the results list
                    results.append(data_dict)
      
        # Convert the results list to a dataframe
        df_results = pd.DataFrame(results)

        # Display the resulting dataframe

        st.subheader("Results")
        st.write(df_results)

        # st.write(df_complete)
        make_scatterplot(df_results, "F-statistic P-value", "Adjusted R-squared", "")
    
    st.subheader("Data sources")
    st.info("https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea")
    st.info("https://www.rivm.nl/corona/actueel/weekcijfers")

def make_df_filtered(df_merged,age_sex, y_value, shift_weeks, window, start_wk, start_yr, end_wk, end_yr):
    """ Filter on period and age sex
        move and smooth y_value

    Args:
        df_merged (_type_): _description_
        age_sex (_type_): _description_
        y_value (_type_): _description_
        shift_weeks (_type_): _description_
        window (_type_): _description_
        start_wk (_type_): _description_
        start_yr (_type_): _description_
        end_wk (_type_): _description_
        end_yr (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    pseudo_start_week = start_yr * 52 + start_wk
    pseudo_eind_week = end_yr * 52 + end_wk
    df_period = df_merged[(df_merged["pseudoweek"] >= pseudo_start_week) & (df_merged["pseudoweek"] <= pseudo_eind_week)]
    df_period = df_period[df_period["week"] != 53]
                    
    
    df_filtered = df_period[df_period["age_sex"] == age_sex].copy(deep=True)
    df_filtered[y_value] = df_filtered[y_value].rolling(window=window, center=True).mean()
    df_filtered[y_value] = df_filtered[y_value].shift(shift_weeks)
    return df_filtered

if __name__ == "__main__":
    import os
    import datetime
    
    os.system('cls')
    print(f"-------xx-------{datetime.datetime.now()}-------------------------")
    main()
  