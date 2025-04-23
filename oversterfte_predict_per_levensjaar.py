import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cbsodata
from scipy.stats import linregress
from scipy.optimize import curve_fit

def get_dataframe(file, delimiter=";"):
    """Get data from a file and return as a pandas DataFrame.

    Args:
        file (str): url or path to the file.
        delimiter (str, optional): _description_. Defaults to ";".

    Returns:
        pd.DataFrame: dataframe
    """   
    
    data = pd.read_csv(
        file,
        delimiter=delimiter,
        low_memory=False,
    )
    return data


def voorspel_sterftekans_linaire_regressie(overlijdens, startjaar=2000, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023, 2024]):
    voorspellingen = []
    for (leeftijd, geslacht), groep in overlijdens.groupby(["leeftijd", "Geslacht"]):
        # Filter data voor regressie
        data = groep[(groep["jaar"] >= startjaar) & (groep["jaar"] <= eindjaar)]
        if len(data) < 2:
            continue  # Skip als er te weinig data is

        # Lineaire regressie
        slope, intercept, _, _, _ = linregress(data["jaar"], data["werkelijke_sterftekans"])

        # Voorspel sterftekans voor de voorspeljaren
        for jaar in voorspeljaren:
            voorspelde_kans = intercept + slope * jaar
            voorspellingen.append({"leeftijd": leeftijd, "Geslacht": geslacht, "jaar": jaar, "voorspelde_sterftekans": voorspelde_kans})
    
    return pd.DataFrame(voorspellingen)


def voorspel_sterftekans(overlijdens, startjaar=2000, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023], model_type="linear"):
    """Predict mortality rates using the specified regression model.

    Args:
        overlijdens (pd.DataFrame): DataFrame containing mortality data.
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

    voorspellingen = []
    for (leeftijd, geslacht), groep in overlijdens.groupby(["leeftijd", "Geslacht"]):
        # Filter data for regression
        data = groep[(groep["jaar"] >= startjaar) & (groep["jaar"] <= eindjaar)]
        if len(data) < len(p0):  # Ensure enough data points for the model
            continue  # Skip if there is insufficient data

        # Fit the selected regression model
        try:
            popt, _ = curve_fit(func, data["jaar"], data["werkelijke_sterftekans"], p0=p0)
            params = dict(zip(model["params"], popt))

            # Predict mortality rates for the prediction years
            for jaar in voorspeljaren:
                voorspelde_kans = func(jaar, *popt)
                voorspellingen.append({
                    "leeftijd": leeftijd,
                    "Geslacht": geslacht,
                    "jaar": jaar,
                    "voorspelde_sterftekans": voorspelde_kans,
                    **params
                })
            for jaar in range(startjaar,eindjaar+1):
                voorspelde_kans = func(jaar, *popt)
                voorspellingen.append({
                    "leeftijd": leeftijd,
                    "Geslacht": geslacht,
                    "jaar": jaar,
                    "voorspelde_sterftekans": voorspelde_kans,
                    **params
                })
        except RuntimeError:
            # Skip if the curve fitting fails
            continue

    return pd.DataFrame(voorspellingen),params

def bereken_verschil(overlijdens, voorspellingen, bevolking):
    # Voeg voorspellingen toe aan overlijdens
    overlijdens = overlijdens.merge(voorspellingen, on=["leeftijd", "Geslacht", "jaar"], how="left")

    # Bereken voorspelde sterfte
    overlijdens["voorspelde_sterfte"] = overlijdens["voorspelde_sterftekans"] * bevolking["aantal"]

    # Bereken verschil
    overlijdens["verschil"] = overlijdens["OverledenenLeeftijdBijOverlijden_1"] - overlijdens["voorspelde_sterfte"]

    # Bereken totaal verschil
    totaal_verschil = overlijdens["verschil"].sum()
    return overlijdens, totaal_verschil


def main():
    st.header("Oversterfte berekening")
    st.info("""
1. We delen het aantal overlijdens per leeftijd *l* door het aantal mensen van diezelfde leeftijd *l*.
2. We voorspellen de sterftekans vanaf 2020 met een lineaire of kwadratische regressie, op basis van data van het gekozen beginjaar tot en met 2019.
3. We vermenigvuldigen de voorspelde sterftekans met het aantal inwoners van leeftijd *l* in het betreffende jaar.
4. We berekenen het verschil tussen de werkelijke sterfte en de voorspelde sterfte.
5. We tellen deze verschillen op per jaar en per geslacht.
""")


    # sterfte = get_sterftedata()

    # https://www.cbs.nl/nl-nl/visualisaties/dashboard-bevolking/bevolkingspiramide ???
    bevolking = get_dataframe(r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv")
    
    # https://www.cbs.nl/nl-nl/cijfers/detail/37168
    # overlijdens = get_dataframe(r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_geslacht_leeftijd_burgelijkstaat.csv", ",")
    # overlijdens = get_dataframe(r"sualisaties/dashboard-bevolking/bevolkingspiramide ???
    overlijdens = get_dataframe(r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_geslacht_leeftijd_burgelijkstaat.csv", ",")
   
   # Replace "M" with "Mannen" and "F" with "Vrouwen" in the "Geslacht" column
    bevolking["Geslacht"] = bevolking["geslacht"].replace({"M": "Mannen", "F": "Vrouwen"})

    
     # Filter data
    overlijdens = overlijdens[
        ((overlijdens["Geslacht"].isin(["Mannen", "Vrouwen"])) &
        (overlijdens["BurgerlijkeStaat"] == "Totaal burgerlijke staat"))
    ]

     # Filter "Leeftijd" column to only include values like "xx jaar"
    overlijdens = overlijdens[overlijdens["Leeftijd"].str.match(r"^\d{1,2} jaar$")]

    # Convert "24 jaar" to integer 24
    overlijdens["leeftijd"] = overlijdens["Leeftijd"].str.extract(r"(\d+)").astype(int)

    overlijdens["jaar"] = overlijdens["Perioden"]

     # 37168
   
    # overlijdens.to_csv("overlijdens_bewerkt.csv", index=False, encoding="utf-8")
    totaal_tabel = bevolking.merge(overlijdens, on=["jaar", "leeftijd", "Geslacht"], how="right")
   

    
      # Allow user to select model type
    col1,col2,col3,col4=st.columns(4)
    with col1:
        model_type = st.selectbox("Select regression model", ["linear", "quadratic"])
    with col2:
        startjaar = st.number_input("Start year", min_value=1960, max_value=2019, value=2015)
    with col3:
        leeftijd_min = st.number_input("Min leeftijd", min_value=0, max_value=99, value=0)
    with col4:
        leeftijd_max = st.number_input("Max leeftijd", min_value=0, max_value=99, value=99)
    
    eindresultaat= pd.DataFrame()
    # Bereken sterftekans
    col_mannen, col_vrouwen = st.columns(2)

    for geslacht, col in zip(["Mannen", "Vrouwen"], [col_mannen, col_vrouwen]):
        totaal_tabel_geslacht = totaal_tabel[totaal_tabel["Geslacht"] == geslacht]
        totaal_tabel_geslacht = totaal_tabel_geslacht[(totaal_tabel_geslacht["leeftijd"] >= leeftijd_min) & (totaal_tabel_geslacht["leeftijd"] <= leeftijd_max)]
        
        totaal_tabel_geslacht = totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= startjaar]
        totaal_tabel_geslacht = totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] <= 2023]
        totaal_tabel_geslacht["werkelijke_sterftekans"] = totaal_tabel_geslacht["OverledenenLeeftijdBijOverlijden_1"] / totaal_tabel_geslacht["aantal"]
    
        # Predict mortality rates
        voorspellingen,parameters = voorspel_sterftekans(totaal_tabel_geslacht, startjaar=startjaar, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023], model_type=model_type)
        
        totaal_tabel_geslacht = voorspellingen.merge(totaal_tabel_geslacht, on=["jaar", "leeftijd", "Geslacht"], how="outer")
        totaal_tabel_geslacht["voorspelde_sterfte"] = totaal_tabel_geslacht["voorspelde_sterftekans"] * totaal_tabel_geslacht["aantal"]
        totaal_tabel_geslacht["oversterfte"] = totaal_tabel_geslacht["OverledenenLeeftijdBijOverlijden_1"] - totaal_tabel_geslacht["voorspelde_sterfte"]
        
        # Group by year and display
        totaal_oversterfte_per_jaar = totaal_tabel_geslacht.groupby("jaar")["oversterfte"].sum().astype(int).reset_index()
        totaal_oversterfte_per_jaar = totaal_oversterfte_per_jaar[totaal_oversterfte_per_jaar["jaar"] >=2020]
       
         
            
        eindresultaat = pd.concat([eindresultaat, totaal_tabel_geslacht], ignore_index=True)    
    
        totaal_tabel_geslacht = totaal_tabel_geslacht.groupby("jaar").sum().reset_index()
        if leeftijd_min == leeftijd_max:
            wx = [["werkelijke_sterftekans", "voorspelde_sterftekans", "Sterftekans"], ["OverledenenLeeftijdBijOverlijden_1", "voorspelde_sterfte", "Overledenen"]]
            col.write(f"Formule voor de sterftekans voorspelling - {model_type}")
            if model_type =="lineair":
                # Extract parameters
                a = parameters.get("a", 0)
                b = parameters.get("b", 0)
                col.write("ax + b")
                col.write("a = {a}")
                col.write("b = {b}")  
            elif model_type == "quadratic": 
                a = parameters.get("a", 0)
                b = parameters.get("b", 0)
            
                c = parameters.get("c", 0) 
                col.write("ax^2 + bx + c")
                col.write(f"a = {a}")
                col.write(f"b = {b}")
                col.write(f"c = {c}")
        else:
            # De voorspelling gaat per levensjaar
            # dus geen grafiek van de voorspelling van de overlijdenskansen hier
            wx = [["OverledenenLeeftijdBijOverlijden_1", "voorspelde_sterfte", "Overledenen"]]
        for w in wx:
            # totaal_tabel_geslacht[w] = totaal_tabel_geslacht[w].astype(float)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht[w[0]], mode='markers', name=f'{w[0]}', ))
            fig.add_trace(go.Scatter(
                    x=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= 2020]["jaar"],
                    y=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= 2020][w[0]],
                    mode='markers',
                    name=f'{w[0]} (2020+)',
                    marker=dict(color='red')
                ))
            fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht[w[1]], mode='lines', name=f'{w[1]}', marker=dict(color='green') ))       
            fig.add_trace(go.Scatter(
                    x=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= 2020]["jaar"],
                    y=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= 2020][w[1]],
                    mode='lines+markers',
                    name=f'{w[1]} (2020+)',
                    marker=dict(color='green')
                ))
            # Add the line to the graph
            
            #fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht["y_line"], mode='lines', name=f'Voorspelde lijn', line=dict(dash='dash', color='red')))
            fig.update_layout(title=f"{w[2]} {geslacht}", xaxis_title="Jaar", yaxis_title="Waarde")
            col.plotly_chart(fig, use_container_width=True)

    # Maak een pivot table met jaren als rijen, geslacht als kolommen en oversterfte als waarden
    oversterfte_tabel = eindresultaat.pivot_table(
        index="jaar",
        columns="Geslacht",
        values="oversterfte",
        aggfunc="sum"
    ).reset_index()

    # Voeg een totaal kolom toe per jaar
    oversterfte_tabel["Totaal"] = oversterfte_tabel["Mannen"] + oversterfte_tabel["Vrouwen"]

    # Converteer alle cellen naar integers
    oversterfte_tabel = oversterfte_tabel.fillna(0).astype(int)
    oversterfte_tabel=oversterfte_tabel[oversterfte_tabel["jaar"]>=2020]

    # Toon de tabel
    st.write(oversterfte_tabel)
    st.write(f"Totale oversterfte : {oversterfte_tabel['Totaal'].sum()}")


if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    
    main()