import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import datetime as dt
import platform
import time

def derivate(x, a, b, c, d):
    ''' First derivate of the sigmoidal function. Might contain an error'''
    return  (np.exp(b * (-1 * np.exp(-c * x)) - c * x) * a * b * c ) + d

def fit_the_values(country_, y_values):
    try:
        base_value = y_values[0]
        # some preperations
        number_of_y_values = len(y_values)
        x_values = np.linspace(start=0, stop=number_of_y_values - 1, num=number_of_y_values)
        popt_d, pcov_d = curve_fit(            f=derivate,            xdata=x_values,            ydata=y_values,               p0 = [26660, 9, 0.03, base_value],             bounds=(-np.inf, np.inf),            maxfev=10000,        )
        residuals = y_values - derivate(x_values, *popt_d)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r_squared = round(  1 - (ss_res / ss_tot),4)
        l = (f"derivate fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f / r2 = {r_squared}" % tuple(popt_d))
        a,b,c,d = popt_d
        r_total = sum(
            (derivate(i, a, b, c, d) / derivate(i - 1, a, b, c, d)) ** (4 / 1)
            for i in range(1, number_of_y_values)
        )

        r_avg_formula = r_total/(number_of_y_values-1)
    except RuntimeError as e:
        pass
    try:
        a = 1* r_avg_formula
    except NameError:
        r_avg_formula = None

    return r_avg_formula

#################################################################
def getdata():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\owid-covid-data.csv"
    else:
        url1= "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    return pd.read_csv(url1, delimiter=",", low_memory=False)

def main():
    s1 = (int(time.time()))
    df_getdata = getdata()
    df = df_getdata.copy(deep=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.fillna(value=0, inplace=True)

    FROM = dt.datetime.strptime("2021-05-22", "%Y-%m-%d").date()
    UNTIL = dt.datetime.strptime("2021-06-12", "%Y-%m-%d").date()
    what_to_fit = "new_cases"

    countrylist =  df['location'].drop_duplicates().sort_values().tolist()
    df_country_r = pd.DataFrame(columns=['country', "R_value"])
    for country_ in countrylist:
        df_to_fit = df.loc[df['location'] == country_]
        mask = (df_to_fit["date"].dt.date >= FROM) & (df_to_fit["date"].dt.date <= UNTIL)
        df_to_fit = df_to_fit.loc[mask]
        df_to_use = df_to_fit.reset_index()
        df_to_use.fillna(value=0, inplace=True)
        values_to_fit = df_to_use[what_to_fit].tolist()
        if len(values_to_fit) != 0:
            R_value_country = fit_the_values(country_, values_to_fit)
            if R_value_country != None and R_value_country < 5:
                print (f"{country_}  - {R_value_country}")
                df_country_r = df_country_r.append({'country': country_, "R_value": R_value_country}, ignore_index=True)
    df_country_r.sort_values(by='R_value', ascending=False)
    print (df_country_r)
    s3 = (int(time.time()))
    s4 = s3-s1
    print ("\nCOMPLETE - Totaal aantal sec : "+ str(s4))

if __name__ == "__main__":
    main()
