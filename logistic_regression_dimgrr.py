import pandas as pd
import statsmodels.api as sm

# Creating the initial data
data = {
    'Persoon': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Leeftijd (jaar)': [65, 72, 50, 45, 80, 60, 55, 68, 49, 75],
    'Lengte (cm)': [175, 168, 180, 160, 170, 165, 178, 172, 169, 160],
    'Gewicht (kg)': [80, 72, 90, 60, 75, 85, 70, 78, 65, 68],
    'Bloeddruk (mmHg)': ['120/80', '130/85', '140/90', '110/70', '150/95', '135/88', '125/80', '140/85', '120/75', '145/90'],
    'Overleden (0/1)': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
}

# Creating the dataframe
df = pd.DataFrame(data)

# Splitting the blood pressure into two separate columns
df[['Systolic (mmHg)', 'Diastolic (mmHg)']] = df['Bloeddruk (mmHg)'].str.split('/', expand=True)

# Dropping the original 'Bloeddruk (mmHg)' column
df = df.drop(columns=['Bloeddruk (mmHg)'])

# Converting the new columns to numeric values
df['Systolic (mmHg)'] = pd.to_numeric(df['Systolic (mmHg)'])
df['Diastolic (mmHg)'] = pd.to_numeric(df['Diastolic (mmHg)'])

# Defining the dependent and independent variables
X = df[['Leeftijd (jaar)', 'Lengte (cm)', 'Gewicht (kg)', 'Systolic (mmHg)', 'Diastolic (mmHg)']]
y = df['Overleden (0/1)']

# Adding a constant (intercept) to the model
X = sm.add_constant(X)

# Fitting the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Displaying the summary of the logistic regression
print(result.summary())

import numpy as np

# Assuming the logistic regression model from the previous step has been fitted and is stored in 'result'

# Creating 5 dummy individuals with specified values for the independent variables
dummy_data = {
    'Leeftijd (jaar)': [65, 50, 70, 40, 55],
    'Lengte (cm)': [180, 165, 175, 160, 170],
    'Gewicht (kg)': [80, 68, 85, 60, 75],
    'Systolic (mmHg)': [120, 140, 130, 110, 135],
    'Diastolic (mmHg)': [80, 90, 85, 70, 88]
}

# Creating a dataframe for the dummy individuals
dummy_df = pd.DataFrame(dummy_data)

# Adding a constant (intercept) to the dummy data
dummy_df = sm.add_constant(dummy_df)

# Predicting the probabilities of death (Overleden = 1) for the dummy individuals
predicted_probabilities = result.predict(dummy_df)

# Showing the predicted probabilities for the 5 dummy individuals
for i, prob in enumerate(predicted_probabilities, start=1):
    print(f"Individual {i} - Probability of Death: {prob:.4f}")
