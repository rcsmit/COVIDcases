import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

def voorspel_overlijdens(df, serienaam, startjaar=2015, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023, 2024,2025], model_type="linear"):
    """Predict mortality  using the specified regression model.

    Args:
        df (pd.DataFrame): DataFrame containing mortality data.
        startjaar (int, optional): Start year for regression. Defaults to 2000.
        eindjaar (int, optional): End year for regression. Defaults to 2019.
        voorspeljaren (list, optional): Years to predict. Defaults to [2020, 2021, 2022, 2023].
        model_type (str, optional): Regression model type ("linear" or "quadratic"). Defaults to "linear".

    Returns:
        pd.DataFrame: DataFrame with predicted mortality rates.
    """
    
    models = {
        "linear": {
            "func": lambda x, a, b: a * x + b,
            "p0": [1, 1],
            "equation": "a*x + b",
            "params": ["a", "b"]
        },
        "quadratic": {
            "func": lambda x, a, b, c: a * x**2 + b * x + c,
            "p0": [1, 1, 1],
            "equation": "a*x^2 + b*x + c",
            "params": ["a", "b", "c"]
        }
    }
    
    

    # Select the model
    model = models[model_type]
    func = model["func"]
    p0 = model["p0"]
    
    df_jaar = df.groupby(["jaar"])[serienaam].sum().reset_index()

    noemer = df_jaar[df_jaar["jaar"].between(2015, 2019)][serienaam].mean()

    voorspelde_overlijdens = []
        # Filter data for regression
    data = df_jaar[(df_jaar["jaar"] >= startjaar) & (df_jaar["jaar"] <= eindjaar)]
    # if len(data) < len(p0):  # Ensure enough data points for the model
    #     continue  # Skip if there is insufficient data

    # Fit the selected regression model
    try:
        popt, _ = curve_fit(func, data["jaar"], data[serienaam], p0=p0)
        params = dict(zip(model["params"], popt))

        # Predict mortality rates for the prediction years
        for jaar in voorspeljaren:
            voorspelde_overlijdens_ = func(jaar, *popt)
            voorspelde_overlijdens.append({"jaar":jaar, "prediction":voorspelde_overlijdens_})
        
    except RuntimeError:
        pass
        # # Skip if the curve fitting fails
        # continue
    

    return voorspelde_overlijdens,params, noemer


def get_data_for_series_wrapper(df_, seriename, vanaf_jaar):

    df = df_[["jaar", "week", "periodenr", seriename]].copy(deep=True)

    df = df[(df["jaar"] > vanaf_jaar)]
    df = df.sort_values(by=["jaar", "week"]).reset_index()



    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    # som_2015_2019 = 0
    df = get_data_for_series(df, seriename, vanaf_jaar)
    return df
def get_data_for_series(df, seriename, vanaf_jaar):
    voorspelde_overlijdens,params, noemer = voorspel_overlijdens(df, seriename, startjaar=2000, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023, 2024,2025], model_type="linear")
    with st.expander("Average and expected", expanded=False):
        df_voorsp = pd.DataFrame(voorspelde_overlijdens)
        st.write(f"Average deaths 2015-2019 : {noemer}")
        st.write("Expected deaths (extrapolation from lin. regression 2015-2019)")
        st.write(df_voorsp)

    for y in range(2015, 2020):
        df_year = df[(df["jaar"] == y)]
        # som = df_year["m_v_0_999"].sum()
        # som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
        

        # 2.5.1 Verwachte sterfte en oversterfte
        # De oversterfte is het verschil tussen het waargenomen aantal overledenen en een verwacht 
        # aantal overledenen in dezelfde periode. Het verwachte aantal overledenen wanneer er geen 
        # COVID-19-epidemie zou zijn geweest, wordt geschat op basis van de waargenomen sterfte in 
        # 2015–2019 in twee stappen. Eerst wordt voor elk jaar de sterfte per week bepaald.
        # Vervolgens wordt per week de gemiddelde sterfte in die week en de zes omliggende weken bepaald. 
        # Deze gemiddelde sterfte per week levert een benadering van de verwachte wekelijkse sterfte. 
        # Er is dan nog geen rekening gehouden met de ontwikkeling van de bevolkingssamenstelling. 

        # Daarom is de sterfte per week nog herschaald naar de verwachte totale sterfte voor het jaar. 
        # Het verwachte aantal overledenen in het hele jaar wordt bepaald op basis van de prognoses 
        # die het CBS jaarlijks maakt. Deze prognoses geven de meest waarschijnlijke toekomstige 
        # ontwikkelingen van de bevolking en de sterfte. De prognoses houden rekening met het feit 
        # dat de bevolking continu verandert door immigratie en vergrijzing. Het CBS gebruikt voor 
        # de prognose van de leeftijds- en geslachtsspecifieke sterftekansen een extrapolatiemodel 
        # (L. Stoeldraijer, van Duin et al., 2013
        # https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf
        # ): er wordt van uitgegaan dat de toekomstige trends 
        # een voortzetting zijn van de trends uit het verleden. In het model wordt niet alleen 
        # uitgegaan van de trends in Nederland, maar ook van de meer stabiele trends in andere 
        # West-Europese landen. Tijdelijke versnellingen en vertragingen die voorkomen in de 
        # Nederlandse trends hebben zo een minder groot effect op de toekomstverwachtingen. 
        # Het model houdt ook rekening met het effect van rookgedrag op de sterfte, wat voor 
        # Nederland met name belangrijk is om de verschillen tussen mannen en vrouwen in sterftetrends 
        # goed te beschrijven.
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022?onepage=true
        # 
        # Op basis van de geprognosticeerde leeftijds- en geslachtsspecifieke sterftekansen en de verwachte
        #  bevolkingsopbouw in dat jaar, wordt het verwachte aantal overledenen naar leeftijd en geslacht 
        # berekend voor een bepaald jaar. Voor 2020 is de verwachte sterfte 153 402, 
        # voor 2021 is deze 154 887 en voor 2022 is dit 155 493. 
        # Het aantal voor 2020 is ontleend aan de Kernprognose 2019-2060 
        # (L. Stoeldraijer, van Duin, C., Huisman, C., 2019), het aantal voor 2021 aan de
        #  Bevolkingsprognose 2020-2070 exclusief de aanname van extra sterfgevallen door de 
        # COVID-19-epidemie; (L. Stoeldraijer, de Regt et al., 2020) en het aantal voor 2022 
        # aan de Kernprognose 2021-2070 (exclusief de aanname van extra sterfgevallen door de coronapandemie)
        #  (L. Stoeldraijer, van Duin et al., 2021). 
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022/2-data-en-methode
        

           
      
        # Het verwachte aantal overledenen wanneer er geen COVID-19-epidemie was geweest, is
        # geschat op basis van de waargenomen sterfte in 2015–2019. Eerst wordt voor elk jaar de
        # sterfte per week bepaald. Vervolgens wordt per week de gemiddelde sterfte in die week
        # en de zes omliggende weken bepaald. Deze gemiddelde sterfte per week levert een
        # benadering van de verwachte wekelijkse sterfte. Er is dan nog geen rekening gehouden
        # met de trendmatige vergrijzing van de bevolking. Daarom is de sterfte per week nog
        # herschaald naar de verwachte totale sterfte voor het jaar. Voor 2020 is de verwachte sterfte
        # 153 402 en voor 2021 is deze 154 887. Het aantal voor 2020 is ontleend aan de
        # Kernprognose 2019–2060 en het aantal voor 2021 aan de Bevolkingsprognose 2020–2070
        # CBS en RIVM | Sterfte en oversterfte in 2020 en 2021 | Juni 2022 15
        # (exclusief de aanname van extra sterfgevallen door de corona-epidemie). De marges rond
        # de verwachte sterfte zijn geschat op basis van de waargenomen spreiding in de sterfte per
        # week in de periode 2015–2019. Dit 95%-interval geeft de bandbreedte weer van de
        # gewoonlijk fluctuaties in de sterfte. 95 procent van de sterfte die in eerdere jaren is
        # waargenomen, valt in dit interval. Er wordt van oversterfte gesproken wanneer de sterfte
        # boven de bovengrens van dit interval ligt.
        # https://www.cbs.nl/en-gb/news/2024/06/fewer-deaths-in-2023/excess-mortality-and-expected-mortality
        # geen waarde voor 2024, zie https://twitter.com/Cbscommunicatie/status/1800505651833270551
        # huidige waarde 2024 is geexptrapoleerd 2022-2023
         # huidige waarde 2025 is geexptrapoleerd vanuit 2022-2023
        if seriename =="m_v_0_64":
            noemer = 21664

            
            factors = {
                # lineaire regressie sterfte 2015-2019
                # https://chatgpt.com/share/68029f4d-5388-8004-8ba5-bf9b603c84fc
                # Sum of OBS_VALUE_	Column Labels			
                # year	Y0-49_T	Y50-64_T	Total
                # 2015	5643	16480		22123
                # 2016	5670	16504		22174
                # 2017	5643	15992		21635
                # 2018	5442	16060		21502
                # 2019	5334	15554		20888
                # 2020	5670	16316		21986
                # 2021	5525	17063		22588
                # 2022	5582	16292		21874
                # 2023	5440	15775		21215

                2014: 1,
                2015: 1,
                2016: 1,
                2017: 1,
                2018: 1,
                2019: 1,
                2020   :       20721.8 / noemer,
                2021   :       20407.6 / noemer,
                2022   :       20093.4 / noemer,
                2023   :       19779.2 / noemer,
                2024   :       19465.0 / noemer,
                2025: 19150/ noemer,
                2026:18837/ noemer,
                2027:18522/ noemer,
                2028:18208/ noemer,
                2029:17894/ noemer,
                2030:17579/ noemer
            }
        elif (seriename =="m_v_0_999") or (seriename=="TOTAL_T"):
            noemer = 149832 # average deaths per year 2015-2019     
            factors = {
                2014: 1,
                2015: 1,
                2016: 1,
                2017: 1,
                2018: 1,
                2019: 1,
                2020: 153402 / noemer,
                2021: 154887 / noemer,
                2022: 155494 / noemer,
                2023: 156666 / noemer,  # or 169333 / som if you decide to use the updated factor
                2024: 157846 / noemer,
                2025: (((156666/155494)**1) * 157846)/ noemer,
                2026: (((156666/155494)**2) * 157846)/ noemer,
                2027: (((156666/155494)**3) * 157846)/ noemer,
                2028: (((156666/155494)**4) * 157846)/ noemer,
                2029: (((156666/155494)**5) * 157846)/ noemer,
            }

            # 2015	16,9	0,5
            # 2016	17	0,6
            # 2017	17,1	0,6
            # 2018	17,2	0,6
            # 2019	17,3	0,6
            # 2020	17,4	0,7
            # 2021	17,5	0,4
            # 2022	17,6	0,7
            # 2023	17,8	1,3
            # 2024	17,9	0,7
        else:
            factors = {
                2014: 1,
                2015: 1,
                2016: 1,
                2017: 1,
                2018: 1,
                2019: 1,
                2020: voorspelde_overlijdens[0]["prediction"] / noemer,
                2021: voorspelde_overlijdens[1]["prediction"] / noemer,
                2022: voorspelde_overlijdens[2]["prediction"] / noemer,
                2023: voorspelde_overlijdens[3]["prediction"] / noemer,  # or 169333 / som if you decide to use the updated factor
                2024: voorspelde_overlijdens[4]["prediction"] / noemer,
                2025: voorspelde_overlijdens[5]["prediction"]/ noemer,
                2026: voorspelde_overlijdens[6]["prediction"]/ noemer,
                # 2026: (((156666/155494)**2) * 157846)/ noemer,
                # 2027: (((156666/155494)**3) * 157846)/ noemer,
                # 2028: (((156666/155494)**4) * 157846)/ noemer,
                # 2029: (((156666/155494)**5) * 157846)/ noemer,
            }
    # avg_overledenen_2015_2019 = (som_2015_2019/5)
    # st.write(avg_overledenen_2015_2019)
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, datetime.now().year+2):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        # factor=1
        df[new_column_name] = df[seriename] * factor

    return df


@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata(data_ruw, vanaf_jaar, seriename="m_v_0_999", ):
    """Manipulate data of the deaths with CBSOdata so we can use it

    
    Args:
        seriename (str, optional): _description_. Defaults to "m_v_0_999".
    """
 
    def manipulate_data_df(data):
        """Filters out week 0 and 53 and makes a category column (eg. "M_V_0_999")"""

        # data = data[~data['week'].isin([0, 53])] #filter out week 2020-53
        data["periodenr"] = (
            data["jaar"].astype(str) + "_" + data["week"].astype(str).str.zfill(2)
        )

        data["Geslacht"] = data["Geslacht"].replace(
            ["Totaal mannen en vrouwen"], "m_v_"
        )
        data["Geslacht"] = data["Geslacht"].replace(["Mannen"], "m_")
        data["Geslacht"] = data["Geslacht"].replace(["Vrouwen"], "v_")
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["Totaal leeftijd"], "0_999"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["0 tot 65 jaar"], "0_64"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["65 tot 80 jaar"], "65_79"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["80 jaar of ouder"], "80_999"
        )
        data["categorie"] = data["Geslacht"] + data["LeeftijdOp31December"]

        return data

    def extract_period_info(period):
        """Function to extract the year, week, and days

        Args:
            period (string): e.g. 2024 week 12 (3 dagen)

        Returns:
            year, week, days: number of yy,ww and dd
        """
        #
        import re

        #pattern = r"(\d{4}) week (\d{1,2}) \((\d+) dag(?:en)?\)"
        pattern = r"(\d{4}) week (\d{1,2})(?: \((\d+) dag(?:en)?\))?"

        match = re.match(pattern, period)
        if match:
            year, week, days = match.groups()
            if days==None: 
                days = 7
            return int(year), int(week), int(days)
        return None, None, None

    def adjust_overledenen(df):
        """# Adjust "Overledenen_1" based on the week number
        # if week = 0, overledenen_l : add to week 52 of the year before
        # if week = 53: overleden_l : add to week 1 to the year after

        TODO: integrate chagnes from calculate_baselines.py
        """
         # Extract the number after 'week' in the column
        
        for index, row in df.iterrows():

           
            if row["week"] == 0:
                previous_year = row["year"] - 1
                df.loc[
                    (df["year"] == previous_year) & (df["week"] == 52), "Overledenen_1"
                ] += row["Overledenen_1"]
            elif row["week"] == 53:
                next_year = row["year"] + 1
                df.loc[
                    (df["year"] == next_year) & (df["week"] == 1), "Overledenen_1"
                ] += row["Overledenen_1"]
        # Filter out the rows where week is 0 or 53 after adjusting
        df = df[~df["week"].isin([0, 53])]
        return df

   
    
    def extend_dataframe_to_future(df_, seriename):
        """
        Extends a DataFrame with time series data to include future dates up to 2026 week 52.
        
        Parameters:
        -----------
        df_ : pandas.DataFrame
            Original DataFrame with 'jaar', 'periodenr', 'week' and series data columns
        seriename : str
            Name of the series column to extend
            
        Returns:
        --------
        pandas.DataFrame
            Extended DataFrame with future dates added
        """
        import pandas as pd
        import numpy as np
        
        # Create a clean copy with only the columns we need
        df = df_[["jaar", "periodenr", "week", seriename]].copy()
        
        # Sort the dataframe by year and period
        df = df.sort_values(by=["jaar", "periodenr"])
        
        # Get existing year-week combinations
        existing = set(zip(df['jaar'], df['week']))
        
        # Define future period parameters
        target_year = 2026
        target_week = 52
        
        # Create future records
        future_records = []
        
        for year in range(2025, target_year + 1):
            max_week = 52 if year < target_year else target_week
            
            for week in range(1, max_week + 1):
                if (year, week) not in existing:
                    future_records.append({
                        'jaar': year,
                        'week': week,
                        'periodenr': f"{year}_{week:02d}",
                        seriename: np.nan
                    })
        
        # Only create a new DataFrame if we have future records to add
        if future_records:
            # Create future DataFrame
            df_future = pd.DataFrame(future_records)
            
            # Ensure consistent dtypes between original and future DataFrames
            dtypes = {
                'jaar': 'int32',
                'periodenr': 'object',
                'week': 'int32',
                seriename: 'float64'
            }
            
            df = df.astype(dtypes)
            df_future = df_future.astype(dtypes)
            
            # Concatenate the DataFrames
            df_extended = pd.concat([df, df_future], ignore_index=True)
            
            # Sort the extended DataFrame
            df_extended = df_extended.sort_values(by=["jaar", "week"]).reset_index(drop=True)
            
            return df_extended
        else:
            # If no future records needed, just return the original (sorted)
            return df.reset_index(drop=True)

    
    # Filter rows where Geslacht is 'Totaal mannen en vrouwen' and LeeftijdOp31December is 'Totaal leeftijd'
    # data_ruw = data_ruw[(data_ruw['Geslacht'] == 'Totaal mannen en vrouwen') & (data_ruw['LeeftijdOp31December'] == 'Totaal leeftijd')]
   
    # week 2024-53 mist aanduiding 2 dagen
    data_ruw['Perioden'] = data_ruw['Perioden'].replace('2024 week 53', '2024 week 53 (2 dagen)')
    data_ruw[["jaar", "week"]] = data_ruw.Perioden.str.split(
        " week ",
        expand=True,
    
    )
    # data_ruw["week"].fillna("_", inplace=True)
    
    data_ruw["week"].fillna("_")
    #data_ruw.fillna({col: "week"}, inplace=True)

    data_ruw["week_number"] = data_ruw["week"].str.extract(r"week (\d+)")

    data_ruw["jaar"] = data_ruw["jaar"].astype(int)
    data_ruw = data_ruw[(data_ruw["jaar"] > 2013)]
    data_ruw = manipulate_data_df(data_ruw)
    data_ruw = data_ruw[data_ruw["categorie"] == seriename]
    data_ruw[["year", "week", "days"]] = data_ruw["Perioden"].apply(
        lambda x: pd.Series(extract_period_info(x))
    )
    data_compleet = data_ruw[~data_ruw["Perioden"].str.contains("dag")]
    data_incompleet = data_ruw[data_ruw["Perioden"].str.contains("dag")]
  
    # Apply the function to the "perioden" column and create new columns
    # data_incompleet[["year", "week", "days"]] = data_incompleet["Perioden"].apply(
    #     lambda x: pd.Series(extract_period_info(x))
    # )
    data_compleet = adjust_overledenen(data_compleet)
    data_incompleet = adjust_overledenen(data_incompleet)
    data = pd.concat([data_compleet, data_incompleet])
    data = data[data["week"].notna()]
    data["week"] = data["week"].astype(int)

    data = data.sort_values(by=["jaar", "week"]).reset_index()

    # Combine the adjusted rows with the remaining rows

    df_ = data.pivot(
        index=["periodenr", "jaar", "week"], columns="categorie", values="Overledenen_1"
    ).reset_index()
    df_["week"] = df_["week"].astype(int)
    df_["jaar"] = df_["jaar"].astype(int)

    # dit moet nog ergens anders
    df_[["periodenr", "delete"]] = df_.periodenr.str.split(
        r" \(",
        expand=True,
    )
    df_ = df_.replace("2015_1", "2015_01")
    df_ = df_.replace("2020_1", "2020_01")
    df_ = df_.replace("2019_1", "2019_01")
    df_ = df_.replace("2025_1", "2025_01")
    #df_ = df_[~df_["week"].isin([0, 53])]
    df_ = df_[(df_["jaar"] > 2014)]

    df = df_[["jaar", "periodenr", "week", seriename]].copy(deep=True)

    df = df.sort_values(by=["jaar", "periodenr"]) #.reset_index()


    df_extended = extend_dataframe_to_future(df, seriename)
  

    df_x = get_data_for_series_wrapper(df_extended, seriename, vanaf_jaar)
    
    return df_x

def make_row_df_quantile(series_name, year, df_to_use, w_, period):
    """DONE
    Calculate the percentiles of a certain week
        make_df_quantile -> make_df_quantile -> make_row_quantile

    Args:
        series_name (_type_): _description_
        year (_type_): _description_
        df_to_use (_type_): _description_
        w_ (_type_): _description_
        period 
    Returns:
        _type_: _description_
    """
    if w_ == 53:
        w = 52
    else:
        w = w_
    
    if period == "week":
        #eurostats
        df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)
        # if len (df_to_use_)==0:
        #     #oversterftecompleet
        #     df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)
    elif period == "maand":
        
        df_to_use_ = df_to_use[(df_to_use["maand"] == w)].copy(deep=True)

    column_to_use = series_name + "_factor_" + str(year)
  
    data = df_to_use_[column_to_use]  # .tolist()
    avg = round(data.mean(), 0)
      
          
    # try:
    if 1==1:
        q05 = np.percentile(data, 5)
        q25 = np.percentile(data, 25)
        q50 = np.percentile(data, 50)
        q75 = np.percentile(data, 75)
        q95 = np.percentile(data, 95)
    # except:
    #     q05, q25, q50, q75, q95 = 0, 0, 0, 0, 0

    

    sd = round(data.std(), 0)
    low05 = round(avg - (2 * sd), 0)
    high95 = round(avg + (2 * sd), 0)

    df_quantile_ = pd.DataFrame(
        [
            {
                "week": w_,
                "jaar": year,
                "q05": q05,
                "q25": q25,
                "q50": q50,
                "avg_": avg,
                "avg": avg,
                "q75": q75,
                "q95": q95,
                "low05": low05,
                "high95": high95,
            }
        ]
    )

    return df_quantile_

def make_df_quantile_year(series_name, df_data, year, period):

    """Calculate the quantiles for a certain year
        make_df_quantile -> make_df_quantile_year -> make_row_quantile

    Returns:
        _type_: _description_
    """

    

    df_to_use = df_data[(df_data["jaar"] >= 2015) & (df_data["jaar"] < 2020)].copy(
        deep=True
    )
   
    df_quantile = None
    end = 53 if period == "week" else 13

    #week_list = df_to_use["periodenr"].unique().tolist()
    for w in range(1, end):
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w, period)
        df_quantile = pd.concat([df_quantile, df_quantile_], axis=0)
    if ((year==2020) or (year==2024) ) and period == "week":
        #2020 has a week 53
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, 53, period)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
    
    return df_quantile

def make_df_quantile(series_name, df_data, period):
    """_Makes df quantile
    make_df_quuantile -> make_df_quantile_year -> make_row_quantile

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        df : merged df
        df_corona: df with baseline
        df_quantiles : df with quantiles
    """

    


    df_coronas = []
    df_corona = df_data[df_data["jaar"].between(2015, datetime.now().year+1)]

    # List to store individual quantile DataFrames
    df_quantiles = []

    # Loop through the years 2014 to 2024
    for year in range(2015, datetime.now().year+1):
        df_corona_year = df_data[(df_data["jaar"] ==year)].copy(deep=True)
        
        df_quantile_year = make_df_quantile_year(series_name, df_data, year, period)
        df_quantiles.append(df_quantile_year)
        df_coronas.append(df_corona_year)
    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)
    if_corona = pd.concat(df_coronas,axis=0)
    
    df_quantile["periodenr"] = (
        df_quantile["jaar"].astype(str)
        + "_"
        + df_quantile["week"].astype(str).str.zfill(2)
    )


 
    df = pd.merge(df_corona, df_quantile, on="periodenr", how="inner")

    return df, df_corona, df_quantile
