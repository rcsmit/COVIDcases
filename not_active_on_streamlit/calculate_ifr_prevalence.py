# Berekenen van percentage vd bevolking dat COVID heeft gehad adh van de prevalentie
# Tevens berekening van IFR
# René Smit, 18 april 2021 - MIT License

# Oorspronkelijke code door Guido Smeets
# https://twitter.com/Guido_Smeets/status/1383784146972004359

from tabulate import tabulate
import pandas as pd

days = 8
inhabitants = 17_483_471


url = "https://data.rivm.nl/covid-19/COVID-19_prevalentie.json"
df = pd.read_json(url)
df["prev_low_cum"] = df["prev_low"].cumsum()
df["prev_up_cum"] = df["prev_up"].cumsum()
df["prev_avg_cum"] = df["prev_avg"].cumsum()

number_low = (df["prev_low_cum"].max() / days )
number_avg = (df["prev_avg_cum"].max() / days)
number_up =  (df["prev_up_cum"].max() / days)

perc_low = (df["prev_low_cum"].max() / days ) / inhabitants * 100
perc_avg = (df["prev_avg_cum"].max() / days) / inhabitants * 100
perc_up = (df["prev_up_cum"].max() / days) / inhabitants * 100

ifr17_low = 17000 / (df["prev_low_cum"].max() /days) * 100
ifr17_avg = 17000 / (df["prev_avg_cum"].max() /days) * 100
ifr17_up = 17000 / (df["prev_up_cum"].max() /days) * 100

ifr21_low = 21000 / (df["prev_low_cum"].max() /days) * 100
ifr21_avg = 21000 / (df["prev_avg_cum"].max() /days) * 100
ifr21_up = 21000 / (df["prev_up_cum"].max() /days) * 100


data = [["low", number_low, perc_low, ifr17_low, ifr21_low],
["avg",  number_avg, perc_avg, ifr17_avg, ifr21_avg],
["max",  number_up, perc_up, ifr17_up, ifr21_up]]
print(tabulate(data, headers=["", "number", "% of population", "IFR16k", "IFR20k"]))

