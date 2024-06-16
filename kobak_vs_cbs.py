import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import matplotlib as mpl

from matplotlib.patches import Polygon
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels.api as sm

import cbsodata
import streamlit as st
import plotly.graph_objects as go
import math


def get_kobak():
    """Load the csv with the baselines as calculated by Ariel Karlinsky and Dmitry Kobak
    https://elifesciences.org/articles/69336#s4
    https://github.com/dkobak/excess-mortality/


    One line is deleted: Netherlands, 2020, 53, 3087.2
    since all other years have 52 weeks

    Returns:
        _type_: _description_
    """
  
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/kobak_baselines.csv"
    #file = r"C:\\Users\\rcxsm\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\kobak_baselines.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
  
    return df_
  
# @st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata(seriename="m_v_0_999"):
    """Get and manipulate data of the deaths

    Args:
        seriename (str, optional): _description_. Defaults to "m_v_0_999".
    """

        
    def manipulate_data_df(data):
        """Filters out week 0 and 53 and makes a category column (eg. "M_V_0_999")

        """    
        data = data[data['week'].notna()]
        data = data[~data['week'].isin([0, 53])] #filter out week 2020-53
        data["weeknr"] = data["jaar"].astype(str) +"_" + data["week"].astype(str).str.zfill(2)
        data["week_int"]=data['week'].astype(int)
        
        data['Geslacht'] = data['Geslacht'].replace(['Totaal mannen en vrouwen'],'m_v_')
        data['Geslacht'] = data['Geslacht'].replace(['Mannen'],'m_')
        data['Geslacht'] = data['Geslacht'].replace(['Vrouwen'],'v_')
        data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['Totaal leeftijd'],'0_999')
        data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['0 tot 65 jaar'],'0_64')
        data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['65 tot 80 jaar'],'65_79')
        data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['80 jaar of ouder'],'80_999')
        data['categorie'] = data['Geslacht']+data['LeeftijdOp31December']
        return data

    data_ruw = pd.DataFrame(cbsodata.get_data('70895ned'))

    # Filter rows where Geslacht is 'Totaal mannen en vrouwen' and LeeftijdOp31December is 'Totaal leeftijd'
    data_ruw = data_ruw[(data_ruw['Geslacht'] == 'Totaal mannen en vrouwen') & (data_ruw['LeeftijdOp31December'] == 'Totaal leeftijd')]
    data_ruw[['jaar','week']] = data_ruw.Perioden.str.split(" week ",expand=True,)
    data_compleet = data_ruw[~data_ruw['Perioden'].str.contains('dag')]
    data_inclompleet = data_ruw[data_ruw['Perioden'].str.contains('dag')]
   
    # Function to extract the year, week, and days
    def extract_period_info(period):
        import re
        pattern = r"(\d{4}) week (\d{1,2}) \((\d+) dag(?:en)?\)"
        match = re.match(pattern, period)
        if match:
            year, week, days = match.groups()
            return int(year), int(week), int(days)
        return None, None, None

    # Apply the function to the "perioden" column and create new columns
    data_inclompleet[['year', 'week', 'days']] = data_inclompleet['Perioden'].apply(lambda x: pd.Series(extract_period_info(x))) 

   
    def adjust_overledenen(df):
        """ # Adjust "Overledenen_1" based on the week number
            # if week = 0, overledenen_l : add to week 52 of the year before
            # if week = 53: overleden_l : add to week 1 to the year after
        """        
        for index, row in df.iterrows():
            if row['week'] == 0:
                previous_year = row['year'] - 1
                df.loc[(df['year'] == previous_year) & (df['week'] == 52), 'Overledenen_1'] += row['Overledenen_1']
            elif row['week'] == 53:
                next_year = row['year'] + 1
                df.loc[(df['year'] == next_year) & (df['week'] == 1), 'Overledenen_1'] += row['Overledenen_1']
        # Filter out the rows where week is 0 or 53 after adjusting
        df = df[~df['week'].isin([0, 53])]
        return df

    data_adjusted = adjust_overledenen(data_inclompleet)
    
    # Combine the adjusted rows with the remaining rows
    data = pd.concat([data_compleet, data_adjusted])

    data = manipulate_data_df(data)

    df_ = data.pivot(index=['weeknr', "jaar", "week"], columns='categorie', values = 'Overledenen_1').reset_index()
    df_["week"] = df_["week"].astype(int)
    df_["jaar"] = df_["jaar"].astype(int)


    if seriename == "m_v_0_999":
        df = df_[["jaar","weeknr","week", seriename]].copy(deep=True)
    else:
        df = df_[["jaar","week","weeknr","m_v_0_999", seriename]].copy(deep=True)

    df = df[ (df["jaar"] > 2014)] 
    df = df.sort_values(by=['jaar','weeknr']).reset_index()
    
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    som_2015_2019 = 0
    for y in range (2015,2020):
        df_year = df[(df["jaar"] == y)]
        som = df_year["m_v_0_999"].sum()
        som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
        factors = {
            2014: 1,
            2015: 1,
            2016: 1,
            2017: 1,
            2018: 1,
            2019: 1,
            2020: 153402 / (som_2015_2019/5),
            2021: 154887 / (som_2015_2019/5),
            2022: 155494 / (som_2015_2019/5),
            2023: 156666 / (som_2015_2019/5),  # or 169333 / som if you decide to use the updated factor
            2024: 157846 / (som_2015_2019/5)
        }
    avg_overledenen_2015_2019 = (som_2015_2019/5)
    
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, 2025):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        df[new_column_name] = df[seriename] * factor

    
    return df

def make_df_quantile(series_name, df_data):
    """_Makes df quantile
    make_df_quantile -> make_df_quantile -> make_row_quantile

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        df : merged df
        df_corona: df with baseline
        df_quantiles : df with quantiles
    """    
   

    def make_df_quantile_year(series_name, df_data, year):

        """ Calculate the quantiles for a certain year
            make_df_quantile -> make_df_quantile -> make_row_quantile


        Returns:
            _type_: _description_
        """    

            
        def make_row_df_quantile(series_name, year, df_to_use, w_):
            """ Calculate the percentiles of a certain week
                make_df_quantile -> make_df_quantile -> make_row_quantile

            Args:
                series_name (_type_): _description_
                year (_type_): _description_
                df_to_use (_type_): _description_
                w_ (_type_): _description_

            Returns:
                _type_: _description_
            """    
            if w_ == 53:
                w = 52
            else:
                w = w_
            
            df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)
            
            
            column_to_use = series_name +  "_factor_" + str(year)
            data = df_to_use_[column_to_use ] #.tolist()

            try:           
                q05 = np.percentile(data, 5)
                q25 = np.percentile(data, 25)
                q50 = np.percentile(data, 50)
                q75 = np.percentile(data, 75)
                q95 = np.percentile(data, 95)
            except:
                q05, q25,q50,q75,q95 = 0,0,0,0,0
                        
            avg = round(data.mean(),0)
            
            sd = round(data.std(),0)
            low05 = round(avg - (2*sd),0)
            high95 = round(avg +(2*sd),0)
            
            df_quantile_ =  pd.DataFrame(
                        [ {
                                "week_": w_,
                                "jaar":year,
                                "q05": q05,
                                "q25": q25,
                                "q50": q50,
                                "avg_": avg,
                                "q75": q75,
                                "q95": q95,
                                "low05":low05,
                                "high95":high95,
                            
                                }]
                        )
                    
            return df_quantile_
        df_to_use = df_data[(df_data["jaar"] >= 2015 ) & (df_data["jaar"] <2020)].copy(deep=True)
    
        
        df_quantile =None
    
            
        week_list = df_to_use['weeknr'].unique().tolist()
        for w in range(1,53):
            df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w)
            df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)     
        return df_quantile
    df_corona = df_data[df_data["jaar"].between(2015, 2025)]

    # List to store individual quantile DataFrames
    df_quantiles = []

    # Loop through the years 2014 to 2024
    for year in range(2015, 2025):
        df_quantile_year = make_df_quantile_year(series_name, df_data, year)
        df_quantiles.append(df_quantile_year)

    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)    
    
    df_quantile["weeknr"]= df_quantile["jaar"].astype(str) +"_" + df_quantile['week_'].astype(str).str.zfill(2)
    
    df = pd.merge(df_corona, df_quantile, on="weeknr")
    return df,df_corona, df_quantile

def predict(X,  verbose=False, excess_begin=None):   
    """Function to predict the baseline with linear regression
       Source: https://github.com/dkobak/excess-mortality/blob/main/all-countries.ipynb

    Args:
        X (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        excess_begin (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    
    def get_excess_begin( datapoints_per_year = 53):    
        if datapoints_per_year > 12:
            beg = 9  # week 10
        
        elif datapoints_per_year > 4 and datapoints_per_year <= 12:
            beg = 2  # March
            
        elif datapoints_per_year == 4:
            beg = 0 
            
        return beg

    # Fit regression model on pre-2020 data 
    ind = (X[:,0] < 2020) & (X[:,1]<53)
    m = np.max(X[ind,1])
    onehot = np.zeros((np.sum(ind), m))
    for i,k in enumerate(X[ind,1]):
        onehot[i,k-1] = 1
    predictors = np.concatenate((X[ind,:1], onehot), axis=1)
    reg = LinearRegression(fit_intercept=False).fit(predictors, X[ind,2])
        
    if verbose:
        est = sm.OLS(X[ind,2], predictors).fit()
        print(est.summary())
    
    # Compute 2020 baseline
    ind2 = X[:,0] == 2020
    predictors2020 = np.concatenate((np.ones((m,1))*2020, np.eye(m)), axis=1)
    baseline = reg.predict(predictors2020)
            
    # Week 53 usually does not have enough data, so we'll use 
    # the same baseline value as for week 52
    if np.max(X[:,1])==53:
        baseline = np.concatenate((baseline, [baseline[-1]]))
        
    # Compute 2021 baseline
    predictors2021 = np.concatenate((np.ones((m,1))*2021, np.eye(m)), axis=1)
    baseline2021 = reg.predict(predictors2021)
    
    # Compute 2022 baseline
    predictors2022 = np.concatenate((np.ones((m,1))*2022, np.eye(m)), axis=1)
    baseline2022 = reg.predict(predictors2022)
    
    # Compute 2023 baseline
    predictors2023 = np.concatenate((np.ones((m,1))*2023, np.eye(m)), axis=1)
    baseline2023 = reg.predict(predictors2023)

    # Excess mortality
    ind2 = X[:,0] == 2020
    diff2020 = X[ind2,2] - baseline[X[ind2,1]-1]
    ind3 = X[:,0] == 2021
    diff2021 = X[ind3,2] - baseline2021[X[ind3,1]-1]
    ind4 = X[:,0] == 2022
    diff2022 = X[ind4,2] - baseline2022[X[ind4,1]-1]
    ind5 = X[:,0] == 2023
    diff2023 = X[ind5,2] - baseline2023[X[ind5,1]-1]
    if excess_begin is None:
        excess_begin = get_excess_begin( baseline.size)
    total_excess = np.sum(diff2020[excess_begin:]) + np.sum(diff2021) + np.sum(diff2022) + np.sum(diff2023)
    # Manual fit for uncertainty computation
    if np.unique(X[ind,0]).size > 1:
        y = X[ind,2][:,np.newaxis]
        beta = np.linalg.pinv(predictors.T @ predictors) @ predictors.T @ y
        yhat = predictors @ beta
        sigma2 = np.sum((y-yhat)**2) / (y.size-predictors.shape[1])
        
        S = np.linalg.pinv(predictors.T @ predictors)
        w = np.zeros((m, 1))
        w[X[(X[:,0] == 2020) & (X[:,1] < 53),1]-1] = 1
        if np.sum((X[:,0] == 2020) & (X[:,1] == 53)) > 0:
            w[52-1] += 1
        w[:excess_begin] = 0
        
        p = 0
        for i,ww in enumerate(w):
            p += predictors2020[i] * ww

        w2021 = np.zeros((m, 1))
        w2021[X[ind3,1]-1] = 1
        for i,ww in enumerate(w2021):
            p += predictors2021[i] * ww
            
        w2022 = np.zeros((m, 1))
        w2022[X[ind4,1]-1] = 1
        for i,ww in enumerate(w2022):
            p += predictors2022[i] * ww

        w2023 = np.zeros((m, 1))
        w2023[X[ind5,1]-1] = 1
        for i,ww in enumerate(w2023):
            p += predictors2023[i] * ww
            
        p = p[:,np.newaxis]
                        
        predictive_var = sigma2 * (np.sum(w)+np.sum(w2021)+np.sum(w2022)+np.sum(w2023)) + sigma2 * p.T @ S @ p
        total_excess_std = np.sqrt(predictive_var)[0][0]
    else:
        total_excess_std = np.nan
       
    return (baseline, baseline2021, baseline2022, baseline2023), total_excess, excess_begin, total_excess_std


def show_plot(df, df_covid, df_kobak_github):
    """_summary_

    Args:
        df (df): df with the calculated values of the Kobak baseline
        df_covid (df): df with the calcualted values with the CBS method
        df_kobak_github (df): df with the values of the Kobak baseline from their Github repo
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df['baseline_kobak'],
        mode='lines',
        name=f'kobak_baseline'
    ))

    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df_kobak_github['baseline_kobak'],
        mode='lines',
        name=f'kobak_baseline GITHUB'
    ))

    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df_covid['avg_'],
        mode='lines',
        name=f'CBS_method_baseline'
    ))

    fig.update_layout(
        title=f'Koak vs CBS',
        xaxis_title='Tijd',
        yaxis_title='Aantal'
    )

    st.plotly_chart(fig)

def do_kobak_vs_cbs():

    """Main function

    Results :
        df_kobak_calculated (df): df with the calculated values of the Kobak baseline
        df_covid (df): df with the calcualted values with the CBS method
        df_kobak_github (df): df with the values of the Kobak baseline from their Github repo
    """    
    st.subheader("Kobak vs CBS")
    df_deaths = get_sterftedata()
    df,_,_ = make_df_quantile("m_v_0_999", df_deaths)

    df_covid=df[(df["jaar_x"]>=2020 )& (df["jaar_x"] <=2023)]
 
    X = df[['jaar_x','week','m_v_0_999']].values
    X = X[~np.isnan(X[:,2]),:]
    X = X.astype(int)
          
    baselines, total_excess, excess_begin, total_excess_std = predict(X)
    list1 = baselines[0].tolist()
    list2 = baselines[1].tolist()
    list3 = baselines[2].tolist()
    list4 = baselines[3].tolist()

    # Combine the lists
    combined_list = list1 + list2 + list3 + list4
    
    # Generate a date range from the start of 2020 to the end of 2023
    date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='W-MON')

    # Extract year and week number
    df_kobak_calculated = pd.DataFrame({
        'year': date_range.year,
        'week': date_range.isocalendar().week
    })
    df_kobak_calculated['baseline_kobak'] = combined_list
    df_kobak_calculated["weeknr"] = df_kobak_calculated["year"].astype(str) +"_" + df_kobak_calculated["week"].astype(str).str.zfill(2)
    
    df_kobak_github = get_kobak()


    show_plot(df_kobak_calculated, df_covid, df_kobak_github)

def main():
    do_kobak_vs_cbs()

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()

   