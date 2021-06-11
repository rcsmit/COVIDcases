# Berekenen van percentage vd bevolking dat COVID heeft gehad adh van de prevalentie
# Tevens berekening van IFR
# René Smit, 18 april 2021 - MIT License

# Oorspronkelijke code door Guido Smeets
# https://twitter.com/Guido_Smeets/status/1383784146972004359

from tabulate import tabulate
import pandas as pd
import streamlit as st


def main():
    inhabitants = (st.sidebar.number_input('Total population', 17_483_471))

    days = (st.sidebar.number_input('Number of days contagious', None,None, 8))

    deaths = (st.sidebar.number_input('Number of deaths', None,None, 30_000))

    url = "https://data.rivm.nl/covid-19/COVID-19_prevalentie.json"
    df = pd.read_json(url)
    df["prev_low_cum"] = df["prev_low"].cumsum()
    df["prev_up_cum"] = df["prev_up"].cumsum()
    df["prev_avg_cum"] = df["prev_avg"].cumsum()

    number_low = round(df["prev_low_cum"].max() / days )
    number_avg = round(df["prev_avg_cum"].max() / days)
    number_up =  round(df["prev_up_cum"].max() / days)

    perc_low = round((df["prev_low_cum"].max() / days ) / inhabitants * 100,4)
    perc_avg = round((df["prev_avg_cum"].max() / days) / inhabitants * 100,4)
    perc_up = round( (df["prev_up_cum"].max() / days) / inhabitants * 100,4)



    ifr_low = round(deaths / (df["prev_low_cum"].max() /days) * 100,4)
    ifr_avg = round(deaths / (df["prev_avg_cum"].max() /days) * 100,4)
    ifr_up = round(deaths / (df["prev_up_cum"].max() /days) * 100,4)

    st.subheader("Aantal COVID besmettingen en IFR berekening")
    st.write("Berekenen van percentage vd bevolking dat COVID heeft gehad adh van de prevalentie")
    st.write("De cummulatieve prevelantie wordt gedeeld door het aantal dagen dat men gemiddeld besmettelijk is")
    st.write("Tevens berekening van IFR")

    data = {'_':["Low", "Avg", "High"],
            'Number': [number_low, number_avg, number_up],
            '% of population': [perc_low, perc_avg, perc_up],
            'IFR': [ifr_low, ifr_avg, ifr_up]}
    df = pd.DataFrame(data)
    st.write (df)
if __name__ == "__main__":
    main()
