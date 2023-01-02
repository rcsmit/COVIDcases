# Data van https://www.cbs.nl/nl-nl/maatwerk/2022/51/overledenen-per-week-provincie-en-gemeente-week-50-2022
# Data van https://coronadashboard.rijksoverheid.nl/landelijk/rioolwater
# TODO : via cbsodb binnen halen

import pandas as pd
import platform
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def get_data():
    """Get the data

    Returns:
        df: dataframe with the data
    """    
    if platform.processor() != "":
        url_data = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overleden_rioolwater.csv"
    else:
        url_data = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overleden_rioolwater.csv"
        
    df =  pd.read_csv(url_data, delimiter=',', low_memory=False)

    geslachten = ["m_v_", "m_", "v_"]
    for g in geslachten:
        df[g + "0_64_real"] =df[g+"0_49_real"] + df[g+"50_64_real"] 
        df[g + "80_999_real"] =df[g+"80_89_real"] + df[g+"90_999_real"] 
    return df

def calculate_values (df,cat):
    """Bereken de verschillende waardes
    Args:
        df (df): de dataframe
        cat (str): een vd leeftijdscategorieen  ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", "m_0_999", "v_0_999"]

    Returns:
        df: dataframe
    """  
    df = df.fillna(0)

    df[cat+"year_minus_high95"] = df[cat+"_real"] - df[cat+"_high"]
    df[cat+"year_minus_exp"] = df[cat+"_real"]- df[cat+"_exp"]
    df[cat+"year_minus_exp_sma"] = df[cat+"year_minus_exp"].rolling(window=3, center=False).mean()
    df[cat+"p_score"] = ( df[cat+"_real"]- df[cat+"_exp"]) /   df[cat+"_exp"]
    df[cat+"p_score"] = df[cat+"p_score"].rolling(window=6, center=True).mean()

    for i in range( len (df)):
        if df.loc[i,cat+"_real" ] >  df.loc[i,cat+"_high"] :
            df.loc[i,cat+"over_onder_sterfte" ] =  df.loc[i,cat+"_real" ]  -  df.loc[i,cat+"_exp"] 
            df.loc[i,cat+"meer_minder_sterfte" ] =  df.loc[i,cat+"_real" ]  -  df.loc[i,cat+"_high"] 
        elif df.loc[i,cat+"_real" ] <  df.loc[i,cat+"_low"]:
            df.loc[i,cat+"over_onder_sterfte" ] =     df.loc[i,cat+"_real" ] - df.loc[i,cat+"_exp"]
            df.loc[i,cat+"meer_minder_sterfte" ] =     df.loc[i,cat+"_real" ] - df.loc[i,cat+"_low"]
    return df

def make_plot(df_,cat, what_,jaar,rw, show_interval):
        """_summary_

        Args:
            df_ (df): _description_
            cat (str): een vd leeftijdscategorieen  ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", "m_0_999", "v_0_999"]
            what_ (str):  ["over_onder_sterfte", "meer_minder_sterfte", "year_minus_exp", "year_minus_exp_sma"]
            jaar (int): 2020,2021 of 2021
            rw (str): ["value_rivm_official" ,  "value_rivm_official_sma", None]
            show_interval (bool) : to show or not to show the intervals and the real sterfte
        """    
        df = df_[df_["year_number"] == jaar].copy(deep=True) 
        what = cat+what_
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace( go.Scatter(x=df['weeknr'],
                                y=df[what],
                                #line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                                line=dict(width=2,color='rgba(205, 61,62, 1)'),
                                mode='lines',
                                name=what,
                            ))
    
        if what == "p_score":
        # the p-score is already plotted
            pass
        elif what == "year_minus_exp": 
            show_avg = False
            if show_avg:   
            
                fig.add_trace(go.Scatter(
                        name="expected",
                        x=df["weeknr"],
                        y=df[cat+"_exp"],
                        mode='lines',
                        line=dict(width=1,color='rgba(205, 61,62, 1)'),
                        ))
       
        if show_interval==True: 
            fig.add_trace( go.Scatter(
                    name='low',
                    x=df["weeknr"],
                    y=df[cat+"_low"],
                    mode='lines',
                    line=dict(width=0.5,
                            color="rgba(255, 188, 0, 0.5)"),
                    fillcolor='rgba(68, 68, 68, 0.2)',
                ))
            fig.add_trace(go.Scatter(
                    name='high',
                    x=df["weeknr"],
                    y=df[cat+"_high"],
                    mode='lines',
                    line=dict(width=0.5,
                            color="rgba(255, 188, 0, 0.5)"), fill='tonexty'
                    ))
            
            fig.add_trace( go.Scatter(
                        name="Verwachte Sterfte",
                        x=df["weeknr"],
                        y=df[cat+"_exp"],
                        mode='lines',
                        line=dict(width=.5,color='rgba(204, 63, 61, .8)'),
                        )) 
            
            fig.add_trace( go.Scatter(
                        name="Sterfte",
                        x=df["weeknr"],
                        y=df[cat+"_real"],
                        mode='lines',
                        line=dict(width=1,color='rgba(204, 63, 61, 1)'),
                        )) 

        
        if rw != None:
           
            fig.add_trace(  go.Scatter(
                    name='rioolwater',
                    x=df["weeknr"],
                    y=df[rw],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=True) 
        
            corr = df[rw].corr(df[what])
            
            st.write(f"Correlation = {round(corr,3)}")  
    
        #data.append(booster)  
                
        title = what
        layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                                title=title,)
                
        fig.add_hline(y=0)
        # fig.add_vrect(x0="2020_33", x1="2020_34", 
        #       annotation_text="Hitte golf", annotation_position="top left",
        #       fillcolor="orange", opacity=0.5, line_width=0)

        fig.add_vrect(x0="2022_32", x1="2022_33", 
              annotation_text="Hitte golf", annotation_position="top right",
              fillcolor="orange", opacity=0.25, line_width=0)
        fig.update_yaxes(rangemode='tozero')

        st.plotly_chart(fig, use_container_width=True)

def main():
    st.header("Rioolwaarde vs Oversterfte")
    st.write("Replicatie van https://twitter.com/rubenivangaalen/status/1606178444328325120")

    st.write("Data van https://www.cbs.nl/nl-nl/maatwerk/2022/51/overledenen-per-week-provincie-en-gemeente-week-50-2022")
    st.write("Data van https://coronadashboard.rijksoverheid.nl/landelijk/rioolwater")
    st.write("Year minus exp = werkelijke waarde minus verwachte waarde")
    st.write("Over-/ondersterfte = werkelijke waarde minus verwachte waarde indien de waarde buiten de 95% grenzen vallen ")
    st.write("Meer-/mindersterfte = werkelijke waarde minus 95% boven-/ondergrens ")
    df = get_data()
    
    cat_ = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", "m_0_999", "v_0_999"]
    what = st.sidebar.selectbox("What", ["over_onder_sterfte", "meer_minder_sterfte", "year_minus_exp", "year_minus_exp_sma"], index = 3)
    jaar = st.sidebar.selectbox("Jaar", [2020,2021,2022], index = 2)
    rw =  st.sidebar.selectbox("Rioolwaterwaarde",  ["value_rivm_official" ,  "value_rivm_official_sma", None], index = 1)
    show_interval = st.sidebar.selectbox("Show sterfte en CI-interval", [True, False], index = 1)
    for cat in cat_:
        st.subheader(f"Grafiek van {cat}")
        df = calculate_values (df,cat)
        make_plot(df, cat, what, jaar, rw, show_interval)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()