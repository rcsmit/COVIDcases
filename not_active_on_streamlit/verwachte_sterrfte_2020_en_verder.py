# We berekenen de sterfte in 2020-2024 aan de hand van de sterftekansen per leeftijdsgroep (5jr, Eurostats) in 2019 en 
# de bevolkingsgrootte in 2020-2024(CBS) 
# Naar aanleiding van https://twitter.com/PeeNee2/status/1855384448478609656

import pandas as pd

# Load the data from CSV files
sterfte_df = pd.read_csv("https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/sterfte_eurostats_NL.csv")
bevolking_df = pd.read_csv("https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/bevolking_leeftijd_NL.csv", delimiter=";")
sterfte_df['jaar'] = sterfte_df['TIME_PERIOD'].str[:4].astype(int)
sterfte_df['week'] = sterfte_df['TIME_PERIOD'].str[6:].astype(int)

total_sterfte = sterfte_df[(sterfte_df["sex"] == "T") & (sterfte_df["age"] == "TOTAL") & (sterfte_df["jaar"] > 2019)]
total_sterfte = total_sterfte.groupby(['age','jaar']).sum().reset_index()
total_sterfte=total_sterfte[["jaar", "OBS_VALUE"]]

# Filter data for the year 2019
sterfte_df=sterfte_df[sterfte_df['sex'] == "T"]
bevolking_df=bevolking_df[bevolking_df["geslacht"] =="T"]
# Extract the year from the TIME_PERIOD column in sterfte_df
sterfte_df = sterfte_df.groupby(['age','jaar']).sum().reset_index()
                   
sterfte_2019 = sterfte_df[sterfte_df['jaar'] == 2019]

# Maak een pivot table
pivot_bevolking = bevolking_df.pivot_table(
    values='aantal',  # Kolom met waarden voor de pivot (aantal overlijdens)
    index='leeftijd',    # Rijen (leeftijdsgroepen)
    columns='jaar',      # Kolommen (jaren)
    aggfunc='sum',       # Aggregatiefunctie (som van overlijdens per leeftijdsgroep per jaar)
    fill_value=0         # Vervang lege waarden door 0
).reset_index()

# Group the population data into 5-year age groups that match sterfte_2019 age ranges
def map_population_age_group(age):
    if age < 5:
        return 'Y_LT5'
    elif age >= 90:
        return 'Y_GE90'
    else:
        low = (age // 5) * 5
        high = low + 4
        return f'Y{low}-{high}'

pivot_bevolking['age'] = pivot_bevolking['leeftijd'].apply(map_population_age_group)

# Aggregate the population data by age group and gender
pivot_bevolking_grouped = pivot_bevolking.groupby(['age']).sum().reset_index()

# Perform the merge based on age range
merged_df = sterfte_2019.merge(
    pivot_bevolking_grouped,
    left_on=['age'],
    right_on=['age']
)
merged_df.columns = merged_df.columns.astype(str)

# Calculate deaths per 100,000 population
merged_df['overlijdens_per_100k_2019'] = (merged_df['OBS_VALUE'] / merged_df['2019']) * 100000
for y in ['2019','2020','2021','2022','2023','2024']:
    merged_df[F'exp_overlijdens_per_100k_{y}'] = (merged_df['overlijdens_per_100k_2019'] * merged_df[y]) / 100000
    print(f"Verwachte sterfte {y} : {int(merged_df[F'exp_overlijdens_per_100k_{y}'].sum())} ") 
print()
print("Verwachte sterfte CBS: [https://www.cbs.nl/en-gb/news/2024/06/fewer-deaths-in-2023/excess-mortality-and-expected-mortality]")
print("2020: 153402")
print("2021: 154887")
print("2022: 155494")
print("2023: 156666")
print("2024: 157846")
print ("*) 2024 eigen interpolatie adhv 2023/2022")
print()
print ("WERKELIJKE STERFTE (2024 tot en met week 19)")
print (total_sterfte)
print ("*) 2024 tot en met week 19")
