import pandas as pd
import statsmodels.api as sm

# reproducing https://twitter.com/dimgrr/status/1844338823184146803

# another example : https://www.kaggle.com/code/anshigupta01/diabetes-prediction-eda-models

import numpy as np

# Data
Xm2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [65, 72, 50, 45, 80, 60, 55, 68, 49, 75],
    [175, 168, 180, 160, 170, 165, 178, 172, 169, 160],
    [80, 72, 90, 60, 75, 85, 70, 78, 65, 68],
    [120, 130, 140, 110, 150, 135, 125, 140, 120, 145],
    [80, 85, 90, 70, 95, 88, 80, 85, 75, 90]
])



def chatgpt_statsmodels():
    print ("USE OF STATSMODELS")
    # https://chatgpt.com/c/67082efc-8a44-8004-b070-c47794980ae5
    # https://chatgpt.com/share/67085891-d000-8004-bd2b-a10e89538ae0
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
    dummy_data_ = {
        'Leeftijd (jaar)': [65, 50, 70, 40, 55],
        'Lengte (cm)': [180, 165, 175, 160, 170],
        'Gewicht (kg)': [80, 68, 85, 60, 75],
        'Systolic (mmHg)': [120, 140, 130, 110, 135],
        'Diastolic (mmHg)': [80, 90, 85, 70, 88]
    }

    dummy_data = {
        'Leeftijd (jaar)': [85, 90, 90, 80, 85],
        'Lengte (cm)': [180, 165, 175, 160, 170],
        'Gewicht (kg)': [80, 98, 105, 80, 95],
        'Systolic (mmHg)': [120, 180, 160, 110, 135],
        'Diastolic (mmHg)': [80, 120, 85, 120, 128]
    }

    dummy_data = {
        'Leeftijd (jaar)': [65, 72, 50, 45, 80, 60, 55, 68, 49, 75],
        'Lengte (cm)': [175, 168, 180, 160, 170, 165, 178, 172, 169, 160],
        'Gewicht (kg)': [80, 72, 90, 60, 75, 85, 70, 78, 65, 68],
        'Bloeddruk (mmHg)': ['120/80', '130/85', '140/90', '110/70', '150/95', '135/88', '125/80', '140/85', '120/75', '145/90'],
    
    }
    # Creating a dataframe for the dummy individuals
    #dummy_df = pd.DataFrame(dummy_data)
    dummy_df = pd.DataFrame(dummy_data)

    #splitting the blood pressure into two separate columns
    dummy_df[['Systolic (mmHg)', 'Diastolic (mmHg)']] = dummy_df['Bloeddruk (mmHg)'].str.split('/', expand=True)

    # Dropping the original 'Bloeddruk (mmHg)' column
    dummy_df = dummy_df.drop(columns=['Bloeddruk (mmHg)'])

    # Converting the new columns to numeric values
    dummy_df['Systolic (mmHg)'] = pd.to_numeric(dummy_df['Systolic (mmHg)'])
    dummy_df['Diastolic (mmHg)'] = pd.to_numeric(dummy_df['Diastolic (mmHg)'])
    # Adding a constant (intercept) to the dummy data
    dummy_df = sm.add_constant(dummy_df)
   
    # Predicting the probabilities of death (Overleden = 1) for the dummy individuals
    predicted_probabilities = result.predict(dummy_df)

    # Showing the predicted probabilities for the 5 dummy individuals

    p=[]
    for i, prob in enumerate(predicted_probabilities, start=1):
        p.append(round(prob))
        #print(f"Individual {i} - Probability of Death: {prob:.4f}")
    print (p)

    return result.params

  

def claude_ai():

    # https://claude.ai/chat/83047b13-f069-4530-8c56-cbf65bd991e2
    

    Y = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    theta = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def pretzel(x):
        mean = np.mean(x, axis=1, keepdims=True)
        range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
        normalized = (x - mean) / range
        return normalized

    # Normalize Xm2
    xmN = pretzel(Xm2.T).T

    def training_step(theta, Xm2, xmN, Y):
        h = sigmoid(np.dot(theta, xmN))
        gradient = np.dot(Xm2, (h - Y)) / Xm2.shape[1]
        theta = theta - 0.003 * gradient
        return theta

    # Training loop
    for _ in range(100000):
        theta = training_step(theta, Xm2, xmN, Y)

    print("Final theta:", " ".join(f"{t:.8f}" for t in theta))

    # Predictions
    predictions = sigmoid(np.dot(theta, xmN)) > 0.5
    print("Predictions:", predictions.astype(int))
    print("Actual Y:   ", Y)
    return theta
if __name__ == "__main__":
    print("Go-----------------")
   

  
    coeffs = chatgpt_statsmodels()
    print ("Expected output 0.8808215577 87.35167466 ¯10.09665304 ¯28.85232926 60.99310998 10.85689242")
    theta = claude_ai()


   

    # To make our theta more comparable, we might need to adjust for feature scaling
    # This is an approximation, assuming 'pretzel' function normalizes features
    feature_means = np.mean(Xm2, axis=1)
    feature_ranges = np.max(Xm2, axis=1) - np.min(Xm2, axis=1)
    
    adjusted_theta = theta / feature_ranges
    adjusted_theta[0] = theta[0] - np.sum(theta * feature_means / feature_ranges)

    print("Adjusted theta (approximate comparison to statsmodels):")
    print(adjusted_theta)
    print ("Expectations")
    print (coeffs)
  