import pandas as pd
import cbsodata
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from oversterfte_helpers import *

import statsmodels.api as sm

# TO SUPRESS
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
pd.options.mode.chained_assignment = None

# replicating https://www.rivm.nl/monitoring-sterftecijfers-nederland

# Doorgetrokken lijn: gemelde sterfte tot en met 2 juni 2024.

# Band: het aantal overlijdens dat het RIVM verwacht.
# Dit is gebaseerd op cijfers uit voorgaande jaren.
# Als de lijn hoger ligt dan de band, overleden er meer mensen dan verwacht. 
# De band geeft de verwachte sterfte weer tussen een bovengrens en een ondergrens.
# De bovengrens is de verwachte sterfte plus twee standaarddeviaties ten opzichte 
# van de verwachte sterfte. De ondergrens is de verwachte sterfte min twee standaarddeviaties
# ten opzichte van de verwachte sterfte. Dit betekent dat 95% van de cijfers van de afgelopen 
# vijf jaar (met uitzondering van de pieken)2 in de band zat.

# De gestippelde lijn geeft schattingen van de sterftecijfers voor de 6 meest recente weken.
# Deze cijfers kunnen nog veranderen. Gemeenten geven hun sterfgevallen door aan het CBS.
# Daar zit meestal enkele dagen vertraging in. Dat zorgt voor een vertekend beeld. 
# Om dat tegen te gaan, zijn de al gemelde sterftecijfers voor de laatste 6 weken opgehoogd.
# Voor deze ophoging kijkt het RIVM naar het patroon van de vertragingen in de meldingen 
# van de sterfgevallen in de afgelopen weken.
# Het RIVM berekent elk jaar in de eerste week van juli de verwachte sterfte 
# voor het komende jaar. Hiervoor gebruiken we de sterftecijfers van de afgelopen 
# vijf jaar. Om vertekening van de verwachte sterfte te voorkomen, tellen we 
# eerdere pieken niet mee. Deze pieken vallen vaak samen met koude- en hittegolven of 
# uitbraken van infectieziekten. Het gaat hierbij om de 25% hoogste sterftecijfers 
# van de afgelopen vijf jaar en de 20% hoogste sterftecijfers in juli en augustus.
# De berekening maakt gebruik van een lineair regressiemodel met een lineaire tijdstrend
# en sinus/cosinus-termen om mogelijke seizoensschommelingen te beschrijven.
# Als de sterfte hoger is dan 2 standaarddeviaties boven de verwachte sterfte, 
# noemen we de sterfte licht verhoogd. Bij 3 standaarddeviaties noemen we de sterfte
# verhoogd. Bij 4 of meer standaarddeviaties noemen we de sterfte sterk verhoogd.
def filter_rivm(df, series_name, y):
    """ Hiervoor gebruiken we de sterftecijfers van de afgelopen 
        vijf jaar. Om vertekening van de verwachte sterfte te voorkomen, tellen we 
        eerdere pieken niet mee. Deze pieken vallen vaak samen met koude- en hittegolven of 
        uitbraken van infectieziekten. Het gaat hierbij om de 25% hoogste sterftecijfers 
        van de afgelopen vijf jaar en de 20% hoogste sterftecijfers in juli en augustus.

        Resultaat vd functie is een df van de 5 jaar voór jaar y (y=2020: 2015-2019) met
        de gefilterde waardes
    
    """
    # Selecteer de gegevens van de afgelopen vijf jaar
    recent_years = df['boekjaar'].max() - 6
    
    df = df[(df['boekjaar'] >= recent_years) & (df['boekjaar'] < y) ]
    
    # Bereken de drempelwaarde voor de 25% hoogste sterftecijfers van de afgelopen vijf jaar
    threshold_25 = df[series_name].quantile(0.75)

    # Filter de data voor juli en augustus (weken 27-35)
    summer_data = df[df['week'].between(27, 35)]
    threshold_20 = summer_data[series_name].quantile(0.80)
    # st.write(f"drempelwaarde voor de 25% hoogste sterftecijfers : {threshold_25=} /  drempelwaarde voor 20% hoogste sterftecijfers in juli en augustus {threshold_20=}")
    set_to_none = False
    if set_to_none :
        # de 'ongeldige waardes' worden vervangen door None
        df.loc[df['jaar'] >= recent_years, series_name] = df.loc[df['jaar'] >= recent_years, series_name].apply(lambda x: np.nan if x > threshold_25 else x)
        df.loc[df['week'].between(27, 35), series_name] = df.loc[df['week'].between(27, 35), series_name].apply(lambda x: np.nan if x > threshold_20 else x)
    else:
        # verwijder de rijen met de 'ongeldige waardes'
        df = df[~((df['jaar'] >= recent_years) & (df[series_name] > threshold_25))]
        df = df[~((df['week'].between(27, 35)) & (df[series_name] > threshold_20))]

    return df
def add_columns(df):
    """voeg columns tijd, sin en cos toe. de sinus/cosinus-termen zijn om mogelijke 
    seizoensschommelingen te beschrijven 
    """  
    # Maak een tijdsvariabele
    df['tijd'] = df['boekjaar'] + (df['boekweek'] - 1) / 52

    # Voeg sinus- en cosinustermen toe voor seizoensgebondenheid (met een periode van 1 jaar)
    
    df.loc[:,'sin'] = np.sin(2 * np.pi * df['boekweek'] / 52)
    df.loc[:,'cos'] = np.cos(2 * np.pi * df['boekweek'] / 52)
    return df

def filter_period(df_new, start_year, start_week,end_year,end_week, add):
    """Filter de dataframe en dus de grafiek in tijd (x-as)  

    """    
  
    condition1 = (df_new[f'jaar{add}'] == start_year) & (df_new[f'week{add}'] >= start_week)
    condition2 = (df_new[f'jaar{add}'] >= start_year) & (df_new[f'jaar{add}'] <= end_year)
    condition3 = (df_new[f'jaar{add}'] == end_year) & (df_new[f'week{add}'] <= end_week)

    # Rijen selecteren die aan een van deze voorwaarden voldoen
    df_new = df_new[condition1 | condition2 | condition3]
    return df_new

def get_data_rivm():
    """laad de waardes zoals RIVM die heeft vastgesteld (11 juni 2024)

    Returns:
        _df: df
    """    
    url="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rivm_sterfte.csv"
    df_ = pd.read_csv(
        url,
        delimiter=";",
        
        low_memory=False,
    )
    return df_

def do_lin_regression( df_filtered,df_volledig, series_naam,y):
    """ lineair regressiemodel met een lineaire tijdstrend
        en sinus/cosinus-termen om mogelijke seizoensschommelingen te beschrijven.
    Args:
        df_filtered : df zonder de uitschieters
        df_volledig : volledige df
        series_naam (_type_): welke serie
        y : jaar

    Returns:
        df : (volledige) df met de CI's
    """    
    df_volledig = add_columns(df_volledig)
    df_filtered = add_columns(df_filtered)

    # Over een tijdvak [j-6-tot j-1] wordt per week wordt de standard deviatie berekend.
    # Hier wordt dan het gemiddelde van genomen

    weekly_std = df_filtered.groupby('boekweek')[series_naam].std().reset_index()
    weekly_std.columns = ['week', 'std_dev']
    sd =  weekly_std['std_dev'].mean()
    #st.write(f"Standard deviatie = {sd}")

    X = df_filtered[['tijd', 'sin', 'cos']]
    X = sm.add_constant(X)  # Voegt een constante term toe aan het model
    y = df_filtered[f'{series_naam}']
    
    model = sm.OLS(y, X).fit()

    X2 = df_volledig[['tijd', 'sin', 'cos']]
    X2 = sm.add_constant(X2) 

    df_volledig.loc[:, 'voorspeld'] = model.predict(X2)
    ci_model = False
    if ci_model:
        # Geeft CI van de voorspelde waarde weer. Niet de CI van de meetwaardes
        voorspellings_interval = model.get_prediction(X2).conf_int(alpha=0.05)
        df_volledig.loc[:,'lower_ci'] = voorspellings_interval[:, 0]
        df_volledig.loc[:,'upper_ci'] = voorspellings_interval[:, 1]
    else:
        df_volledig.loc[:, 'lower_ci'] = df_volledig['voorspeld'] - 2 * sd
        df_volledig.loc[:, 'upper_ci'] = df_volledig['voorspeld'] + 2 * sd
    df_new = pd.merge(df_filtered, df_volledig, on="weeknr", how="outer")
    
    df_new = df_new.sort_values(by=['jaar_y', 'week_y']).reset_index(drop=True)

    return df_new

def plot_graph_rivm(df_, series_naam, rivm):
    """plot the graph

    Args:
        df_ (str): _description_
        series_naam (str): _description_
        rivm (bool): show the official values from the RIVM graph
                        https://www.rivm.nl/monitoring-sterftecijfers-nederland
    """    
    df_rivm = get_data_rivm()
   
    df = pd.merge(df_, df_rivm, on="weeknr", how="outer")
    df = df.sort_values(by=['weeknr']) #.reset_index()
    
    # Maak een interactieve plot met Plotly
    fig = go.Figure()

    # Voeg de werkelijke data toe
    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df[f'{series_naam}_y'],
        mode='lines',
        name='Werkelijke data cbs'
    ))

    # Voeg de voorspelde lijn toe
    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df['voorspeld'],
        mode='lines',
        name='Voorspeld model'
    ))
    if rivm==True:
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(go.Scatter(
            x=df['weeknr'],
            y=df['verw_waarde_rivm'],
            mode='lines',
            name='Voorspeld RIVM'
        ))
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(go.Scatter(
            x=df['weeknr'],
            y=df['ondergrens_verwachting_rivm'],
            mode='lines',
            name='onder RIVM'
        )) # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(go.Scatter(
            x=df['weeknr'],
            y=df['bovengrens_verwachting_rivm'],
            mode='lines',
            name='boven RIVM'
        ))

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df['upper_ci'],
        mode='lines',
        fill=None,
        line_color='lightgrey',
        name='Bovenste CI'
    ))

    fig.add_trace(go.Scatter(
        x=df['weeknr'],
        y=df['lower_ci'],
        mode='lines',
        fill='tonexty',  # Vul het gebied tussen de lijnen
        line_color='lightgrey',
        name='Onderste CI'
    ))

    # Titel en labels toevoegen
    fig.update_layout(
        title='Voorspelling van Overledenen met 95% Betrouwbaarheidsinterval RIVM',
        xaxis_title='Tijd',
        yaxis_title='Aantal Overledenen'
    )

    st.plotly_chart(fig)

def sterfte_rivm(df, series_naam):
    """Verwachte sterfte/baseline  uitrekenen volgens RIVM methode

   _
    """

    # adding week 52, because its not in the data
    # based on the rivm-data, we assume that the numbers are quit the same
    df = duplicate_row(df, "2021_51", "2021_52")
    df = duplicate_row(df, "2022_51", "2022_52")
    df["boekjaar"] = df["jaar"].shift(26)
    df["boekweek"] = df["week"].shift(26)

    df_compleet = pd.DataFrame()
    for y in [2019,2020, 2021,2022,2023]:
        # we filteren 5 jaar voor jaar y (y=2020: 2015 t/m 2020 )
        recent_years = y - 5
        df_ = df[(df['boekjaar'] >= recent_years) & (df['boekjaar'] <= y)]

        df_volledig = df_[['weeknr', "jaar","week","boekjaar", "boekweek", series_naam]]
        df_filtered=filter_rivm(df_, series_naam, y)
        
        df_do_lin_regression = do_lin_regression(df_filtered, df_volledig,  series_naam,y)
        df_do_lin_regression = df_do_lin_regression[(df_do_lin_regression['boekjaar_y'] == y)]
        df_compleet =  pd.concat([df_compleet,df_do_lin_regression])
    return df_compleet
    
def main():
    rivm=st.sidebar.selectbox("Show RIVM values", [False,True])
    df = get_sterftedata()
    series_naam = "m_v_0_999"
    df_compleet = sterfte_rivm(df, series_naam)
    plot_graph_rivm(df_compleet, series_naam, rivm)

if __name__ == "__main__":
    import datetime
    os.system('cls')
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
    show_difference_()