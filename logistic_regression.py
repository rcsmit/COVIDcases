import streamlit as st

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm


# https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/
# https://timeseriesreasoning.com/contents/survival-analysis/
# Generate dummy data
np.random.seed(42)


# Generate dummy data
np.random.seed(42)


def generate_dummy_data_method2(n_samples, year_range):
    data = []
    for year in year_range:
        age = np.random.normal(50, 15, n_samples)
        gender = np.random.choice([0, 1], n_samples)
        mortality = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        data.extend(list(zip([year] * n_samples, age, gender, mortality)))
    return pd.DataFrame(data, columns=["year", "age", "gender", "mortality"])


def main_claude_method2():
    st.subheader("Claude method2")
    # Generate data for reference years (2015-2019) and test years (2020-2021)
    reference_data = generate_dummy_data_method2(1000, range(2015, 2020))
    test_data = generate_dummy_data_method2(1000, range(2020, 2022))

    # Prepare data for logistic regression
    X_ref = reference_data[["age", "gender"]]
    y_ref = reference_data["mortality"]

    # Center age around its mean
    age_mean = X_ref.loc[:, "age"].mean()
    X_ref.loc[:, "age_centered"] = X_ref["age"] - age_mean

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_ref[["age_centered", "gender"]], y_ref)

    # Function to calculate confidence interval
    def calculate_ci(data, confidence=0.95):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
        return mean, ci

    # Analyze test data
    results = []
    for year in [2020, 2021]:
        year_data = test_data[test_data["year"] == year]
        X_test = year_data[["age", "gender"]].copy()

        X_test.loc[:, "age_centered"] = X_test["age"] - age_mean

        # Calculate predicted probabilities
        predicted_probs = model.predict_proba(X_test[["age_centered", "gender"]])[:, 1]

        # Calculate actual mortality rate
        actual_mortality_rate = year_data["mortality"].mean()

        # Calculate confidence interval for predicted probabilities
        mean_prob, ci = calculate_ci(predicted_probs)

        # Determine if there's over/under mortality
        if actual_mortality_rate < ci[0]:
            status = "Under mortality"
        elif actual_mortality_rate > ci[1]:
            status = "Over mortality"
        else:
            status = "Within expected range"

        results.append(
            {
                "Year": year,
                "Actual Mortality Rate": actual_mortality_rate,
                "Predicted Mortality Rate": mean_prob,
                "CI Lower": ci[0],
                "CI Upper": ci[1],
                "Status": status,
            }
        )

    # Print results
    results_df = pd.DataFrame(results)
    st.write(results_df)


def generate_dummy_data_method3(n_samples, year_range):
    data = []
    for year in year_range:
        age = np.random.normal(50, 15, n_samples)
        gender = np.random.choice([0, 1], n_samples)
        # Medical history: 0 - No condition, 1 - Acute, 2 - Long-term, 3 - Chronic
        medical_history = np.random.choice([0, 1, 2, 3], n_samples)
        # Migration background: 0 - No migration background, 1 - Western, 2 - Non-Western
        migration_background = np.random.choice([0, 1, 2], n_samples)
        # Household income (in thousands)
        household_income = np.random.lognormal(mean=3.5, sigma=0.5, size=n_samples)

        # Calculate mortality probabilities individually for each sample
        mortality = []
        for i in range(n_samples):
            base_mortality_prob = 0.1
            individual_prob = (
                base_mortality_prob
                + 0.02 * (age[i] > 65)
                + 0.01 * (medical_history[i] > 1)
                - 0.01 * (household_income[i] > 50)
            )
            mortality.append(
                np.random.choice([0, 1], p=[1 - individual_prob, individual_prob])
            )

        data.extend(
            list(
                zip(
                    [year] * n_samples,
                    age,
                    gender,
                    medical_history,
                    migration_background,
                    household_income,
                    mortality,
                )
            )
        )

    return pd.DataFrame(
        data,
        columns=[
            "year",
            "age",
            "gender",
            "medical_history",
            "migration_background",
            "household_income",
            "mortality",
        ],
    )


def main_claude_method3():

    st.subheader("Claude method3")
    # https://claude.ai/chat/ee977786-8e48-4c49-837c-92ef0114e054

    # Generate data for reference years (2015-2019) and test years (2020-2021)
    reference_data = generate_dummy_data_method3(5000, range(2015, 2020))
    test_data = generate_dummy_data_method3(2000, range(2020, 2022))

    # Prepare data for logistic regression
    X_ref = reference_data[
        ["age", "gender", "medical_history", "migration_background", "household_income"]
    ]
    y_ref = reference_data["mortality"]

    # Center and scale numerical variables
    scaler = StandardScaler()
    X_ref.loc[:, ["age", "household_income"]] = scaler.fit_transform(
        X_ref[["age", "household_income"]]
    )

    # Convert categorical variables to dummy variables
    X_ref = pd.get_dummies(
        X_ref, columns=["medical_history", "migration_background"], drop_first=True
    )

    # Fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_ref, y_ref)

    # Function to calculate confidence interval
    def calculate_ci(data, confidence=0.95):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
        return mean, ci

    # Analyze test data
    results = []
    for year in [2020, 2021]:
        year_data = test_data[test_data["year"] == year]
        X_test = year_data[
            [
                "age",
                "gender",
                "medical_history",
                "migration_background",
                "household_income",
            ]
        ]

        # Apply the same scaling and dummy variable creation as for the reference data
        X_test.loc[:, ["age", "household_income"]] = scaler.transform(
            X_test[["age", "household_income"]]
        )
        X_test = pd.get_dummies(
            X_test, columns=["medical_history", "migration_background"], drop_first=True
        )

        # Ensure all columns from training data are present in test data
        for col in X_ref.columns:
            if col not in X_test.columns:
                X_test[col] = 0

        # Reorder columns to match training data
        X_test = X_test[X_ref.columns]

        # Calculate predicted probabilities
        predicted_probs = model.predict_proba(X_test)[:, 1]

        # Calculate actual mortality rate
        actual_mortality_rate = year_data["mortality"].mean()

        # Calculate confidence interval for predicted probabilities
        mean_prob, ci = calculate_ci(predicted_probs)

        # Determine if there's over/under mortality
        if actual_mortality_rate < ci[0]:
            status = "Under mortality"
        elif actual_mortality_rate > ci[1]:
            status = "Over mortality"
        else:
            status = "Within expected range"

        results.append(
            {
                "Year": year,
                "Actual Mortality Rate": actual_mortality_rate,
                "Predicted Mortality Rate": mean_prob,
                "CI Lower": ci[0],
                "CI Upper": ci[1],
                "Status": status,
            }
        )

    # st.write results
    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Function to analyze data by pandemic phase
    def analyze_by_phase(data, phase_ranges):
        phase_results = []
        for phase, (start, end) in phase_ranges.items():
            phase_data = data[(data["year"] >= start) & (data["year"] <= end)]
            X_phase = phase_data[
                [
                    "age",
                    "gender",
                    "medical_history",
                    "migration_background",
                    "household_income",
                ]
            ]
            # Apply scaling and dummy variable creation
            X_phase.loc[:, ["age", "household_income"]] = scaler.transform(
                X_phase[["age", "household_income"]]
            )
            X_phase = pd.get_dummies(
                X_phase,
                columns=["medical_history", "migration_background"],
                drop_first=True,
            )

            # Ensure all columns from training data are present
            for col in X_ref.columns:
                if col not in X_phase.columns:
                    X_phase[col] = 0

            # Reorder columns to match training data
            X_phase = X_phase[X_ref.columns]

            predicted_probs = model.predict_proba(X_phase)[:, 1]
            actual_mortality_rate = phase_data["mortality"].mean()
            mean_prob, ci = calculate_ci(predicted_probs)

            if actual_mortality_rate < ci[0]:
                status = "Under mortality"
            elif actual_mortality_rate > ci[1]:
                status = "Over mortality"
            else:
                status = "Within expected range"

            phase_results.append(
                {
                    "Phase": phase,
                    "Actual Mortality Rate": actual_mortality_rate,
                    "Predicted Mortality Rate": mean_prob,
                    "CI Lower": ci[0],
                    "CI Upper": ci[1],
                    "Status": status,
                }
            )

        return pd.DataFrame(phase_results)

    # Define pandemic phases (example dates, adjust as needed)
    pandemic_phases = {
        "Pre-pandemic": (2019, 2019),
        "First wave": (2020, 2020),
        "Second wave": (2021, 2021),
    }

    # Analyze by pandemic phase
    phase_results = analyze_by_phase(
        pd.concat([reference_data, test_data]), pandemic_phases
    )
    st.write("\nResults by Pandemic Phase:")
    st.write(phase_results)


def main_chatgpt_method2():
    st.subheader("ChatGPT method2")

    # Step 1: Generate dummy data
    np.random.seed(42)  # For reproducibility

    # Create a dummy dataframe
    num_patients = 1000
    data = {
        "age": np.random.normal(
            70, 10, num_patients
        ),  # Average age around 70 years, std dev of 10
        "gender": np.random.choice(
            [0, 1], size=num_patients
        ),  # 0 for female, 1 for male
        "mortality": np.random.choice(
            [0, 1], size=num_patients, p=[0.9, 0.1]
        ),  # 10% mortality rate
    }

    df_reference = pd.DataFrame(data)

    # Step 2: Logistic regression model
    X = df_reference[["age", "gender"]]
    y = df_reference["mortality"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 3: Make predictions and assess model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    # Step 4: Generate new data for 2020 and 2021 and predict mortality
    num_patients_2020_2021 = 500
    new_data = {
        "age": np.random.normal(70, 10, num_patients_2020_2021),
        "gender": np.random.choice([0, 1], size=num_patients_2020_2021),
    }
    df_2020_2021 = pd.DataFrame(new_data)

    # Predict mortality for 2020-2021 patients
    predicted_mortality = model.predict(df_2020_2021)
    df_2020_2021["predicted_mortality"] = predicted_mortality

    # Display the new data with predictions
    st.write("the new data with predictions")
    st.write(df_2020_2021)


def main_chatgpt_method3():

    # https://chatgpt.com/c/66e031f9-4d30-8004-95c8-fa6cf0f11032

    # Step 1: Generate dummy data with new features
    np.random.seed(42)  # For reproducibility
    st.subheader("ChatGPT method3")
    num_patients = 1000
    data = {
        "age": np.random.normal(
            70, 10, num_patients
        ),  # Average age around 70 years, std dev of 10
        "gender": np.random.choice(
            [0, 1], size=num_patients
        ),  # 0 for female, 1 for male
        "medical_history": np.random.choice(
            [0, 1, 2], size=num_patients, p=[0.5, 0.3, 0.2]
        ),  # 0: No, 1: Acute, 2: Chronic
        "migration_background": np.random.choice(
            [0, 1], size=num_patients, p=[0.8, 0.2]
        ),  # 0: No, 1: Yes
        "household_income": np.random.normal(
            35000, 10000, num_patients
        ),  # Household income mean of 35k with std dev 10k
        "mortality": np.random.choice(
            [0, 1], size=num_patients, p=[0.9, 0.1]
        ),  # 10% mortality rate
    }

    df_reference = pd.DataFrame(data)

    # Step 2: Logistic regression model with additional determinants
    X = df_reference[
        ["age", "gender", "medical_history", "migration_background", "household_income"]
    ]
    y = df_reference["mortality"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 3: Make predictions and assess model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix to evaluate model performance
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"Confusion Matrix:\n{cm}")

    # Step 4: Generate new data for 2020 and 2021 and predict mortality
    num_patients_2020_2021 = 500
    new_data = {
        "age": np.random.normal(70, 10, num_patients_2020_2021),
        "gender": np.random.choice([0, 1], size=num_patients_2020_2021),
        "medical_history": np.random.choice(
            [0, 1, 2], size=num_patients_2020_2021, p=[0.5, 0.3, 0.2]
        ),
        "migration_background": np.random.choice(
            [0, 1], size=num_patients_2020_2021, p=[0.8, 0.2]
        ),
        "household_income": np.random.normal(35000, 10000, num_patients_2020_2021),
    }
    df_2020_2021 = pd.DataFrame(new_data)

    # Predict mortality for 2020-2021 patients
    predicted_mortality = model.predict(df_2020_2021)
    df_2020_2021["predicted_mortality"] = predicted_mortality

    # Display the new data with predictions
    st.write("the new data with predictions")
    st.write(df_2020_2021)

    # Step 5: Calculate 95% confidence interval for the predictions
    # Add intercept for statsmodels logistic regression to calculate CI
    X_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_sm)
    result = logit_model.fit()

    # 95% confidence interval
    conf = result.conf_int(alpha=0.05)
    st.write(f"95% Confidence Interval:\n{conf}")

    # Predict probability for the test set
    pred_prob = result.predict(sm.add_constant(X_test))

    # Adding confidence intervals to the predicted probabilities
    df_test = pd.DataFrame({"actual_mortality": y_test, "predicted_prob": pred_prob})

    # Classify based on 95% confidence level
    df_test["predicted_mortality"] = np.where(df_test["predicted_prob"] >= 0.5, 1, 0)

    # Step 6: Compare actual vs predicted mortality and assess under/over mortality
    st.write("Compare actual vs predicted mortality and assess under/over mortality")
    st.write(df_test)


def main():
    st.header("Reverse engineering Nivel")
    # nav deze tweet https://twitter.com/dimgrr/status/1833246948914041098
    st.info(
        """Methode 2: **Verwachte sterfte op basis van een logistisch regressiemodel met 
     leeftijd en geslacht.** In bovenstaande methode werd geen rekening gehouden met verschillen 
     in populatiekenmerken,  daarom is een aanvullende analyse uitgevoerd. Hiervoor is een 
     logistische regressie analyse  uitgevoerd over de referentiejaren (2015-2019) met 
     overlijden als uitkomstmaat en als  determinanten leeftijd (gecentreerd rondom het gemiddelde) en 
     geslacht. Voor elke patiënt in 2020  en 2021 zijn nieuwe coëfficiënten berekend aan de hand van de 
     coëfficiënten (leeftijd en geslacht)  uit de referentiejaren, hiervoor werd een random waarde 
     getrokken uit de normale verdeling van de  coëfficiënten over de referentiejaren. Vervolgens 
     is voor elke patiënt de regressieformule met de  nieuwe coëfficiënten ingevuld, waarmee op 
     patiëntniveau de kans op overlijden werd berekend. Om  de kans op overlijden voor de gehele 
     populatie te bepalen is het gemiddelde genomen over alle  patiënten. Om te bepalen of er 
     sprake was van oversterfte werd gekeken of de werkelijke sterfte (in  percentage) onder 
     (ondersterfte), binnen (geen over- of ondersterfte) of boven (oversterfte) het 95%  
     betrouwbaarheidsinterval (in percentage) lag van de verwachte kans op overlijden. 
     Dit is bepaald per  jaar, maar ook voor de verschillende fases van de pandemie (zie 2.3). 
     Hierdoor werden dezelfde  periodes met elkaar vergeleken en werd gecorrigeerd voor 
     seizoenseffecten.De werkelijke sterfte  werd vergeleken met de bovenkant van het 
     95%-betrouwbaarheidsinterval van de verwachte sterfte  en met de gemiddelde verwachte sterfte, 
     waardoor de uitkomst een range van over- of ondersterfte  werd. """
    )
    main_claude_method2()
    main_chatgpt_method2()
    st.info(
        """Methode 3: In deze laatste methode is het bovenstaande regressiemodel verder uitgebreid met aanvullende
            determinanten: medische voorgeschiedenis (wel/geen acute, langdurige of chronische aandoening), 
            sociaal-demografische (migratieachtergrond) en sociaaleconomische kenmerken (huishoudinkomen). 
            Daarna is op een vergelijkbare manier als bij methode 2 de verwachte sterfte berekend, inclusief 95% 
            betrouwbaarheidsinterval. Dit werd gedaan om te kunnen vergelijken met de werkelijke sterfte en 
            vast te stellen of er sprake was van ondersterfte, geen over- of ondersterfte of oversterfte. Dit is 
            bepaald per jaar, maar ook voor de verschillende fases van de pandemie (zie 2.3)."""
    )
    main_claude_method3()
    main_chatgpt_method3()

    st.info("Script: https://github.com/rcsmit/COVIDcases/blob/main/logistic_regression.py")
if __name__ == "__main__":
    print("Go-----------------")
    main()
