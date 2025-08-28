import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import median_abs_deviation

# replicating https://www.rivm.nl/monitoring-sterftecijfers-nederland

# Imitating RIVM overstefte grafieken
# overlijdens per week: https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1655808656887


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
    """ Filter de 25% hoogste sterftecijfers
         van de afgelopen vijf jaar en de 20% hoogste sterftecijfers in juli en augustus.

    Resultaat vd functie is een df van de 5 jaar voór jaar y (y=2020: 2015-2019) met
    de gefilterde waardes
    """
    # Selecteer de gegevens van de afgelopen vijf jaar
    df_ = df[(df["boekjaar"] >= y-5) & (df["boekjaar"] < y)]

    # Bereken de drempelwaarde voor de 25% hoogste sterftecijfers van de afgelopen vijf jaar
    threshold_25 = df_[series_name].quantile(0.75)
    #threshold_25 = df_[series_name].quantile(1)

    # Filter de data voor juli en augustus (weken 27-35) per jaar
    summer_thresholds = df[df["week"].between(27, 35)].groupby("boekjaar")[series_name].quantile(0.80)

    # Apply the thresholds
    set_to_none = False
    if set_to_none:
        # de 'ongeldige waardes' worden vervangen door None
        df.loc[df["jaar"] >= y-5, series_name] = df.loc[
            df["jaar"] >= y-5, series_name
        ].apply(lambda x: np.nan if x > threshold_25 else x)
        
        for year, threshold in summer_thresholds.items():
            df.loc[(df["week"].between(27, 35)) & (df["boekjaar"] == year), series_name] = df.loc[
                (df["week"].between(27, 35)) & (df["boekjaar"] == year), series_name
            ].apply(lambda x: np.nan if x > threshold else x)
    else:
        # Create mask for threshold_25 condition
        mask_threshold = ((df["jaar"] >= y-5) & (df[series_name] > threshold_25))
        df_filtered_out_threshold = df[mask_threshold].copy()
        # st.write("df76")
        # st.write(df_filtered_out_threshold)
        # Create empty DataFrame for summer threshold filtered rows
        df_filtered_out_summer = pd.DataFrame()

        # Filter out rows and store filtered rows
        for year, threshold in summer_thresholds.items():
            summer_mask = ((df["week"].between(27, 35)) & (df["boekjaar"] == year) & (df[series_name] >= threshold))
            df_filtered_out_summer = pd.concat([df_filtered_out_summer, df[summer_mask]])

        # Combine all filtered out rows
        df_filtered_out = pd.concat([df_filtered_out_threshold, df_filtered_out_summer])
        # if y==2025:
        #     st.write("df 88df_filtered_out")

        #     st.write(df_filtered_out)

        # Apply filters to main DataFrame
        df = df[~mask_threshold]
        for year, threshold in summer_thresholds.items():
            df = df[~((df["week"].between(27, 35)) & (df["boekjaar"] == year) & (df[series_name] >= threshold))]


        # Return both DataFrames
    pivot_df = create_pivot_df(df, df_filtered_out, series_name)
    
     
    return df, pivot_df

def create_pivot_df(df_filtered, df_filtered_out, series_name):
    """Plot the filtered and filtered out values for a given series
    """

   
    # Create a combined table structure
    # First, prepare both dataframes with consistent columns
    df_filtered = df_filtered.copy()
    df_filtered_out = df_filtered_out.copy()
    
    # Add a status column to each dataframe
    df_filtered['status'] = 'Included'
    df_filtered_out['status'] = 'Filtered Out'
    
  
    # Create display labels with zero-padded week numbers
    df_filtered['date'] = df_filtered.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
    df_filtered_out['date'] = df_filtered_out.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
    df_filtered['periodenr'] = df_filtered.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
    df_filtered_out['periodenr'] = df_filtered_out.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
      # Create display labels with zero-padded week numbers
    df_filtered['label'] = df_filtered.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
    df_filtered_out['label'] = df_filtered_out.apply(
        lambda row: f"{int(row['jaar'])}_{row['week']:02d}", axis=1
    )
     
    # Combine into one dataframe
    columns_to_keep = ['date','periodenr', 'label', 'boekjaar', 'week', series_name, 'status']
    combined_df = pd.concat([
        df_filtered[columns_to_keep],
        df_filtered_out[columns_to_keep]
    ]).sort_values('date')

    
    # Create the pivot table
    pivot_df = pd.pivot_table(
        combined_df,
        values=series_name,
        index='date',
        columns='status',
        aggfunc='mean'  # Using mean, but could be any function that makes sense for your data
    )
    

    # Reset the index to make it more readable
    pivot_df = pivot_df.reset_index()
    return pivot_df    
  


def add_columns_lin_regression(df):
    """voeg columns tijd, sin en cos toe. de sinus/cosinus-termen zijn om mogelijke
    seizoensschommelingen te beschrijven
    """
    # Maak een tijdsvariabele
    df["tijd"] = df["boekjaar"] + (df["boekweek"] - 1) / 52

    # Voeg sinus- en cosinustermen toe voor seizoensgebondenheid (met een periode van 1 jaar)

    df.loc[:, "sin"] = np.sin(2 * np.pi * df["boekweek"] / 52)
    df.loc[:, "cos"] = np.cos(2 * np.pi * df["boekweek"] / 52)
    return df

def do_lin_regression_rivm(df_filtered, df_volledig, series_naam, y):
    """lineair regressiemodel met een lineaire tijdstrend
        en sinus/cosinus-termen om mogelijke seizoensschommelingen te beschrijven.
    Args:
        df_filtered : df zonder de uitschieters
        df_volledig : volledige df
        series_naam (_type_): welke serie
        y : jaar

    Returns:
        df : (volledige) df met de CI's
    """
    df_volledig = add_columns_lin_regression(df_volledig)
    df_filtered = add_columns_lin_regression(df_filtered)

    # Over een tijdvak [j-6-tot j-1] wordt per week wordt de standard deviatie berekend.
    # Hier wordt dan het gemiddelde van genomen
    #df_filtered=df_filtered[df_filtered["boekjaar"]!= y]
    #df_filtered=df_filtered[df_filtered["boekjaar"]== y]

    

    
    # Option 3: Use median absolute deviation (more robust to outliers)
    # NO IDEA WHY THIS FITS BETTER
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    sd_mad = median_abs_deviation(df_filtered[series_naam], scale=1.4826)

    
    weekly_std = df_filtered.groupby("boekweek")[series_naam].std().reset_index()
    weekly_std.columns = ["week", "std_dev"]
    sd = weekly_std["std_dev"].mean()

    #sd = df_filtered[series_naam].std()
    # st.write(f"Standard deviatie = {sd}")

    X = df_filtered[["tijd", "sin", "cos"]]
    X = sm.add_constant(X)  # Voegt een constante term toe aan het model
    y = df_filtered[f"{series_naam}"]

    model = sm.OLS(y, X).fit()

    X2 = df_volledig[["tijd", "sin", "cos"]]
    X2 = sm.add_constant(X2)

    df_volledig.loc[:, "voorspeld"] = model.predict(X2)
    ci_model = False
    if ci_model:
        # Geeft CI van de voorspelde waarde weer. Niet de CI van de meetwaardes
        voorspellings_interval = model.get_prediction(X2).conf_int(alpha=0.05)
        df_volledig.loc[:, "lower_ci"] = voorspellings_interval[:, 0]
        df_volledig.loc[:, "upper_ci"] = voorspellings_interval[:, 1]
    else:
        df_volledig.loc[:, "lower_ci"] = df_volledig["voorspeld"] - 2 * sd
        df_volledig.loc[:, "upper_ci"] = df_volledig["voorspeld"] + 2 * sd
        df_volledig.loc[:, "lower_ci_mad"] = df_volledig["voorspeld"] - 2 * sd_mad
        df_volledig.loc[:, "upper_ci_mad"] = df_volledig["voorspeld"] + 2 * sd_mad

    df_new = pd.merge(df_filtered, df_volledig, on="periodenr", how="outer")

    df_new = df_new.sort_values(by=["jaar_y", "week_y"]).reset_index(drop=True)

    return df_new


def verwachte_sterfte_rivm(df, series_naam):
    """Verwachte sterfte/baseline  uitrekenen volgens RIVM methode

    _
    """

    # adding week 52, because its not in the data
    # based on the rivm-data, we assume that the numbers are quit the same

    # df["boekjaar"] = df["jaar"].shift(26)
    # df["boekweek"] = df["week"].shift(26)
   

    
    df = df.copy()

    # maak ISO-maandagdatum per jaar/week
    df["iso_date_mon"] = df.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["jaar"]), int(r["week"]), 1), axis=1
    )

    # schuif 26 weken vooruit
    shifted = df["iso_date_mon"] - pd.to_timedelta(26, "W")
    iso = shifted.dt.isocalendar()

    df["boekjaar"] = iso.year.astype(int)
    df["boekweek"] = iso.week.astype(int)
    df.drop(columns=["iso_date_mon"], inplace=True)
    df_compleet = pd.DataFrame()
    df_compleet_pivot = pd.DataFrame()
    for y in [2019, 2020, 2021, 2022, 2023,2024,2025,2026,2027]:
        # we filteren 5 jaar voor jaar y (y=2020: 2015 t/m 2020 )
        recent_years = y - 5
        df_ = df[(df["boekjaar"] >= recent_years) & (df["boekjaar"] <= y)]
       
        
        df_volledig = df_[
            ["periodenr", "jaar", "week", "boekjaar", "boekweek", series_naam]
        ]
        # if y == 2026:
        #     st.write("df277 alles ok")
        #     st.write(df_volledig)
        df_filtered, pivot_df = filter_rivm(df_, series_naam, y)
        # if y == 2026:
        #     st.write("df281 not ok")
        #     st.write(df_filtered)

        #     st.write(pivot_df)        

        df_do_lin_regression_rivm = do_lin_regression_rivm(
            df_filtered, df_volledig, series_naam, y
        )
        df_do_lin_regression_rivm = df_do_lin_regression_rivm[
            (df_do_lin_regression_rivm["boekjaar_y"] == y)
        ]
        # if y==2025:
        #     st.write("df_do_lin_regression_rivm 2025")
        #     st.write(df_do_lin_regression_rivm)
        df_compleet = pd.concat([df_compleet, df_do_lin_regression_rivm])
        df_compleet_pivot = pd.concat([df_compleet_pivot, pivot_df])
    #plot_filtered_values(df_compleet_pivot, series_naam)
    # st.write("271")
    # st.write(df_compleet)
    # st.write(df_compleet_pivot)
    return df_compleet, df_compleet_pivot