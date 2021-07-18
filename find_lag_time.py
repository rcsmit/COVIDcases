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

    #if __name__ = "covid_dashboard_rcsmit":
    # global WDW2, centersmooth, show_scenario
    # WDW2=7
    # st.write(__name__)
    # centersmooth = False
    # show_scenario = False
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
    columnlist = ["cases_diff","hospital_intake_rivm"]
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
    make_graph(iz, c1, c2)


main()

