import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime


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

    noemer = 149832 # average deaths per year 2015-2019
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

#           # 2015	16,9	0,5
            # 2016	17	0,6
            # 2017	17,1	0,6
            # 2018	17,2	0,6
            # 2019	17,3	0,6
            # 2020	17,4	0,7
            # 2021	17,5	0,4
            # 2022	17,6	0,7
            # 2023	17,8	1,3
            # 2024	17,9	0,7
    # avg_overledenen_2015_2019 = (som_2015_2019/5)
    # st.write(avg_overledenen_2015_2019)
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, datetime.now().year+1):
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

   
    # Filter rows where Geslacht is 'Totaal mannen en vrouwen' and LeeftijdOp31December is 'Totaal leeftijd'
    # data_ruw = data_ruw[(data_ruw['Geslacht'] == 'Totaal mannen en vrouwen') & (data_ruw['LeeftijdOp31December'] == 'Totaal leeftijd')]
    
    # week 2024-53 mist aanduiding 2 dagen
    data_ruw['Perioden'] = data_ruw['Perioden'].replace('2024 week 53', '2024 week 53 (2 dagen)')
    data_ruw[["jaar", "week"]] = data_ruw.Perioden.str.split(
        " week ",
        expand=True,
    
    )
    data_ruw["week"].fillna("_", inplace=True)
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

    df = df.sort_values(by=["jaar", "periodenr"]).reset_index()

    df = get_data_for_series_wrapper(df, seriename, vanaf_jaar)
    
    return df

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
