import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit


def load_data(source, delimiter=","):
    """
    Load data from a CSV source and return a pandas DataFrame.
    """
    return pd.read_csv(
        source,
        delimiter=delimiter,
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip'
    )


def predict_diagnosis_chance_linear(data, start_year=2000, end_year=2019, years_to_predict=None):
    """
    Use linear regression to predict diagnosis rates for colon cancer.
    """
    if years_to_predict is None:
        years_to_predict = range(2020, 2036)

    results = []
    grouped = data.groupby(["age_group", "gender"])

    for (age_group, gender), group in grouped:
        subset = group[(group["year"] >= start_year) & (group["year"] <= end_year)]
        if len(subset) < 2:
            continue

        slope, intercept, _, _, _ = linregress(
            subset["year"], subset["actual_diagnosis_rate"]
        )

        for year in years_to_predict:
            rate = intercept + slope * year
            results.append({
                "age_group": age_group,
                "gender": gender,
                "year": year,
                "predicted_rate": rate
            })

    return pd.DataFrame(results)


def predict_diagnosis_chance(data, start_year=2000, end_year=2023,
                             years_to_predict=None, model_type="linear"):
    """
    Predict diagnosis rates using linear or quadratic regression.
    """
    models = {
        "linear": {
            "func": lambda x, a, b: a * x + b,
            "p0": [1, 1],
            "params": ["a", "b"]
        },
        "quadratic": {
            "func": lambda x, a, b, c: a * x**2 + b * x + c,
            "p0": [1, 1, 1],
            "params": ["a", "b", "c"]
        }
    }

    if years_to_predict is None:
        years_to_predict = range(2020, 2036)

    data = data[data["year"] <= 2023]
    model = models[model_type]
    func = model["func"]
    p0 = model["p0"]
    all_results = []
    fit_params = {}
  
    groups = data.groupby(["age_group"])
    
    for age_group, group in groups:
       
        subset = group[(group["year"] >= start_year) & (group["year"] <= end_year)]
       

        if len(subset) < len(p0):
            continue

        # try:
        if 1==1:
            popt, _ = curve_fit(func, subset["year"], subset["actual_diagnosis_rate"], p0=p0)
            params = dict(zip(model["params"], popt))
            
            

            
            #fit_params[(age_group)] = params

            for year in list(range(start_year, end_year + 1)) + list(years_to_predict):
                rate = func(year, *popt)
                record = {
                    "age_group": age_group[0],
            
                    "year": year,
                    "predicted_rate": rate,
                    **params
                }
                all_results.append(record)

        # except RuntimeError:
        #     continue
    
    return pd.DataFrame(all_results), fit_params


def calculate_difference(diagnoses, predictions, population):
    """
    Calculate the difference between actual and predicted diagnoses.
    """
    merged = diagnoses.merge(
        predictions,
        on=["age_group", "gender", "year"],
        how="left"
    )
    merged["predicted_count"] = (
        merged["predicted_rate"] * population.set_index(["age", "gender", "year"]) ["count"] / 100000
    ).values
    merged["difference"] = merged["diagnosis_count"] - merged["predicted_count"]
    total = merged["difference"].sum()
    return merged, total


def main():
    st.header("Colon Cancer Overdiagnosis Calculator")
    st.info(
        """
1. Compute actual rate: diagnoses of colon cancer / population  
2. Fit regression model on data up to 2019  
3. Predict rates from 2020 onward  
4. Multiply predicted rate by population to get predicted cases  
5. Subtract predicted cases from actual to find overdiagnosis
        """
    )

    # Load data
 
    population = load_data(r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv", delimiter=";")
    diagnoses = load_data(r"C:\Users\rcxsm\Downloads\darmkanker.csv",delimiter=";")

    # Standardize columns
    population.rename(columns={
        "leeftijd": "age",
        "geslacht": "gender",
        "jaar": "year",
        "aantal": "count"
    }, inplace=True)
   
    diagnoses.rename(columns={
        "leeftijdsgroep": "age_group",
        "Geslacht": "gender",
        "jaar": "year",
        "aantal_darmkanker_diagnoses": "diagnosis_count"
    }, inplace=True)
 
    # Create age groups in population
    bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
            54, 59, 64, 69, 74, 79, 84, 120]
    labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
              "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74",
              "75-79", "80-84", "85-120"]
    population["age_group"] = pd.cut(population["age"], bins=bins,
                                      labels=labels)

    # Merge totals per group
    group_totals = population.groupby(["age_group", "year"]) ["count"].sum().reset_index()
    merged = group_totals.merge(diagnoses, on=["age_group", "year"], how="inner").fillna(0)
    merged["actual_diagnosis_rate"] = merged["diagnosis_count"] / merged["count"] * 100000

    # User inputs
    model_type = st.selectbox("Regression model", ["linear", "quadratic"])
    start_year = st.number_input("Start year", min_value=1960, max_value=2023, value=2000)
    label_options = labels
    default = ["40-44", "45-49", "50-54", "55-59", "60-64"]
    selected_groups = st.multiselect("Select age groups", label_options, default)

    # Predict rates
    data_for_pred = merged[(merged["year"] >= start_year)]
   
    predictions, params = predict_diagnosis_chance(
        data_for_pred,
        start_year=start_year,
        end_year=2019,
        model_type=model_type
    )
    
    full = predictions.merge(merged, on=["age_group", "year"], how="outer")
     
    full["predicted_count"] = full["predicted_rate"] * full["count"] / 100000
    full["overdiagnosis"] = full["diagnosis_count"] - full["predicted_count"]

    # Plot results per group
    for metric, label in [
        ("actual_diagnosis_rate", "Actual Rate"),
        ("predicted_rate", "Predicted Rate")
    ]:
        fig = go.Figure()
        for group in selected_groups:
            dfg = full[full["age_group"] == group]
            fig.add_trace(go.Scatter(
                x=dfg["year"],
                y=dfg[metric],
                mode='lines+markers',
                name=group
            ))
        fig.update_layout(
            title=f"{label} by Year",
            xaxis_title="Year",
            yaxis_title="Rate per 100k"
        )
        st.plotly_chart(fig, use_container_width=True)


    for w in [["actual_diagnosis_rate", "predicted_rate", "diagnosis_rate"],["count","count","inhabitants"] ,["diagnosis_count", "predicted_count", "Absolute numbers"]]:
        fig = go.Figure()   
        for group in selected_groups:
            
            dfg = full[full["age_group"] == group]
        
            
            
            fig.add_trace(go.Scatter(
                    x=dfg[dfg["year"] <= 2024]["year"],
                    y=dfg[dfg["year"] <= 2024][w[0]],
                    mode='markers',
                    name=f' {group}',
                    
                ))
            
            
            # fig.add_trace(go.Scatter(
            #         x=dfg[dfg["year"] >= 2020]["year"],
            #         y=dfg[dfg["year"] >= 2020][w[0]],
            #         mode='markers',
            #         name=f' {age}',
                    
            #     ))
            
            fig.add_trace(go.Scatter(x=dfg["year"], y=dfg[w[1]], mode='lines', name=f'{group}',  ))       
            fig.add_trace(go.Scatter(
                    x=dfg[dfg["year"] >= 2025]["year"],
                    y=dfg[dfg["year"] >= 2025][w[1]],
                    mode='lines+markers',
                    name=f'{group}',
                )
                )

            # Add the line to the graph
        fig.update_layout(title=f"{w[2]}", xaxis_title="Year", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    # Summarize overdiagnosis
    summary = full.groupby("year")["overdiagnosis"].sum().reset_index()
    st.write("### Overdiagnosis by Year")
    st.bar_chart(summary.set_index("year"))


if __name__ == "__main__":
    main()
