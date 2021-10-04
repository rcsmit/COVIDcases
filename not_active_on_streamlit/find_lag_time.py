# Find the lagtime resulting in the highest correlation betwen two fields

# Uses a file of Marino van Zelst.

# fields
# "date","cases","hospitalization","deaths","positivetests","hospital_intake_rivm","Hospital_Intake_Proven",
# "Hospital_Intake_Suspected","IC_Intake_Proven","IC_Intake_Suspected","IC_Current","ICs_Used","IC_Cumulative",
# "Hospital_Currently","IC_Deaths_Cumulative","IC_Discharge_Cumulative","IC_Discharge_InHospital","Hospital_Cumulative",
# "Hospital_Intake","IC_Intake","Hosp_Intake_Suspec_Cumul","IC_Intake_Suspected_Cumul","IC_Intake_Proven_Cumsum",
# "new.infection","corrections.cases","net.infection","new.hospitals","corrections.hospitals","net.hospitals",
# "new.deaths","corrections.deaths","net.deaths","positive_7daverage","infections.today.nursery","infections.total.nursery",
# "deaths.today.nursery","deaths.total.nursery","mutations.locations.nursery","total.current.locations.nursery",
# "values.tested_total","values.infected","values.infected_percentage","pos.rate.3d.avg","IC_Bedden_COVID",
# "IC_Bedden_Non_COVID","Kliniek_Bedden","IC_Nieuwe_Opnames_COVID","Kliniek_Nieuwe_Opnames_COVID","Totaal_bezetting",
# "IC_Opnames_7d","Kliniek_Opnames_7d","Totaal_opnames","Totaal_opnames_7d","Totaal_IC","IC_opnames_14d",
# "Kliniek_opnames_14d","OMT_Check_IC","OMT_Check_Kliniek"


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def get_data():

    url1 = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"

    df = pd.read_csv(url1, delimiter=",", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df

def select_period_oud(df, field, show_from, show_until):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).

    Args:
        df (df): dataframe
        field (string): Field containing the date

    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from =  dt.datetime.strptime( "2020-7-1", "%Y-%m-%d").date()


    if show_until is None:
        show_until = dt.datetime.strptime( "2022-7-1", "%Y-%m-%d").date()
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


def smooth_columnlist(df, columnlist, WDW2, centersmooth):
    """  _ _ _ """


    if columnlist is not None:
        if type(columnlist) == list:
            columnlist_ = columnlist
        else:
            columnlist_ = [columnlist]
            # print (columnlist)
        c_smoothen=[]
        for c in columnlist_:
            print(f"Smoothening {c}")

            new_column =  "SMA_" + str(WDW2) +"_" + c
            print("Generating " + new_column + "...")
            df[new_column] = (
                df.iloc[:, df.columns.get_loc(c)]
                .rolling(window=WDW2, center=centersmooth)
                .mean()
            )

            c_smoothen.append(new_column)
    return df, c_smoothen

def make_moved_columns(df, name):
    new_hosp=[]
    new_hosp_sma = []
    iz = []
    for i in range (-15,15):

        h_new = name + "_moved_"+str(i)
        h_sma_new = "SMA_7_"+ name + "_moved_"+str(i)

        df[h_new] = df[name].shift(i)
        df[h_sma_new] = df["SMA_7_"+ name ].shift(i)
        new_hosp.append(h_new)
        new_hosp_sma.append(h_sma_new)
        iz.append(i)
    return df, new_hosp, iz

def make_graph(x, hospital, hospital_sma):
   # with _lock:

    plt.plot(x,hospital, label = "hospital")
    plt.plot(x,hospital_sma, label = "hospital_sma")
    plt.legend()
    plt.grid()
    plt.title("Crrelatie tussen cases en verschoven ziekenhuisopnames")
    plt.show()

def main():
    df_getdata = get_data()
    df = df_getdata.copy(deep=False)

    df = select_period_oud(df, "date", None, None)
    df["cases_diff"] = df["cases"].diff()


    columnlist = ["cases_diff","hospital_intake_rivm"] # change this to change the columns to use

    df, c_smoothen = smooth_columnlist(df, columnlist, 7,True)

    df, new_hosp, iz= make_moved_columns(df, columnlist[1])

    c1= []
    c2 =[]

    for iy in new_hosp:
        sma = "SMA_7_" + iy
        c1.append(round(df[iy].corr(df["cases_diff"]), 3))
        c2.append(round(df[sma].corr(df["cases_diff"]), 3))

    d = {'date_lag': iz, 'hospital': c1, 'hospital_sma': c2}
    df_output = pd.DataFrame(data=d)
    print (df_output)
    make_graph(iz, c1, c2, cp)

if __name__ == "__main__":
    main()