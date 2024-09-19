# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
import matplotlib.pyplot as plt

# import matplotlib as mpl
# import matplotlib.dates as mdates
import numpy as np
import streamlit as st
import matplotlib.cm as cm
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter
import pandas as pd
import datetime as dt
from datetime import datetime
#from streamlit import caching
import plotly.express as px
from  matplotlib import pyplot
import seaborn


def cell_background_helper(val,method, max, color):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
       st.write (df.style.format(None, na_rep="-").applymap(lambda x:  cell_background_number_of_cases(x,[method], [top_waarde])).set_precision(2))
    Args:
        val ([int]): the value of the cell
        metohod (string): "exponential" / "lineair" / "percentages"
        max : max value (None bij percentage)
        color (string): color 'r, g, b' (None bij percentages)

    Returns:
        [string]: the css code for the cell
    """
    if color == None : color = '193, 57, 43'
    opacity = 0
    try:

        v = abs(val)
        if method == "percentages":
            # scale from -100 to 100
            opacity = 1 if v >100 else v/100
            # color = 'green' if val >0 else 'red'
            if val > 0 :
                color = '193, 57, 43'
            elif val < 0:
                color = '1, 152, 117'
            else:
                color = '255,255,173'
        elif method == "lineair":
            opacity = v / max
        elif method == "kwartiel":
            if val <100: color = '255,199,206'
            if val <75: color = '255,235,156'
            if val <50: color = '225,237,217'
            if val <25: color = '198,223,180'
            opacity = 1
        else:
            if method == "exponential":
                value_table = [ [0,0],
                                [0.00390625,0.0625],
                                [0.0078125, 0.125],
                                [0.015625,0.25],
                                [0.03125,0.375],
                                [0.0625,0.50],
                                [0.125,0.625],
                                [0.25,0.75],
                                [0.50,0.875],
                                [0.75,0.9375],
                                [1,1]]
            elif method == "lineair2":
                value_table = [ [0,0],
                                [0.1,0.0625],
                                [0.2, 0.125],
                                [0.3,0.25],
                                [0.4,0.375],
                                [0.5,0.50],
                                [0.6,0.625],
                                [0.7,0.75],
                                [0.8,0.875],
                                [0.9,0.9375],
                                [1,1]]


            for vt in value_table:
                if v >= round(vt[0]*max) :
                    opacity = vt[1]
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,0'
        opacity = 1


    return f'background: rgba({color}, {opacity})'

def show_heatmap(df, method, max_value, color):
    """Show heatmap from a df

    Args:
        df ([type]): [description]
        method ([type]): [description]
        max_value ([type]): [description]
        color ([type]): [description]
    """

    try:
        st.write (df.style.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,method, max_value, color)).set_precision(2))
    except:
        st.write (df.applymap(lambda x:  cell_background_helper(x,method, max_value, color)))

    make_legenda("lineair", max_value)
def  make_legenda(method, max_value):
    if method == "exponential":
        stapfracties =   [0, 0.00390625, 0.0078125, 0.015625,  0.03125,  0.0625 , 0.125,  0.25,  0.50, 0.75,  1]
    else:
        stapfracties = [ 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        stapjes =[]
        for i in range(len(stapfracties)):
            stapjes.append((stapfracties[i]*max_value))
        d = {'legenda': stapjes}

        df_legenda = pd.DataFrame(data=d)
        st.write (df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,"lineair", max_value,None)))#.set_precision(2))



def cell_background(val):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
    Args:
        val (int): the value of the cell

    Returns:
        [string]: the css code for the cell
    """
    try:
        v = abs(val)
        opacity = 1 if v >100 else v/100
        # color = 'green' if val >0 else 'red'
        if val > 0 :
             color = '193, 57, 43'
        elif val < 0:
            color = '1, 152, 117'
        else:
            color = '255,255,173'
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,255'
        opacity = 1
    return f'background: rgba({color}, {opacity})'

def select_period_input_cache():
    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
         
            st.success("Cache is NOT cleared")

def select_period_input():

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()
    return FROM, UNTIL
def select_period(df, field):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).

    Args:
        df (df): dataframe
        field (string): Field containing the date

    Returns:
        df: filtered dataframe
    """
    show_from, show_until = select_period_input()
    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


def select_period_oud(df, field):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).

    Args:
        df (df): dataframe
        field (string): Field containing the date

    Returns:
        df: filtered dataframe
    """
    show_from, show_until = select_period_input()
    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\output\\"
    )

    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


def drop_columns(df, what_to_drop):
    """  drop columns. what_to_drop : list """
    if what_to_drop != None:
        print("dropping " + str(what_to_drop))
        for d in what_to_drop:
            df = df.drop(columns=[d], axis=1)
    return df

def find_correlation_pair(df, first, second):
    al_gehad = []
    paar = []
    if type(first) == list:
        first = first
    else:
        first = [first]
    if type(second) == list:
        second = second
    else:
        second = [second]
    for i in first:
        for j in second:
            c = round(df[i].corr(df[j]), 3)
    return c

def decimalToHexa(n):
    hexaDecimalNumber = ['0'] * 100
    i = 0
    while (n != 0):
        _temp = 0
        _temp = n % 16
        if (_temp < 10):
            hexaDecimalNumber[i] = chr(_temp + 48)
            i = i + 1
        else:
            hexaDecimalNumber[i] = chr(_temp + 55)
            i = i + 1
        n = int(n / 16)
    hexadecimalCode = ""
    if (i == 2):
        hexadecimalCode = hexadecimalCode + hexaDecimalNumber[0]
        hexadecimalCode = hexadecimalCode + hexaDecimalNumber[1]
    elif (i == 1):
        hexadecimalCode = "0"
        hexadecimalCode = hexadecimalCode + hexaDecimalNumber[0]
    elif (i == 0):
        hexadecimalCode = "00"
    return hexadecimalCode


# Function to convert the RGB to Hexadecimal color code
def RGBtoHexConverion(R, G, B):
    if ((R >= 0 and R <= 255) and
            (G >= 0 and G <= 255) and
            (B >= 0 and B <= 255)):
        hexadecimalCode = "#"
        hexadecimalCode = hexadecimalCode + decimalToHexa(R)
        hexadecimalCode = hexadecimalCode + decimalToHexa(G)
        hexadecimalCode = hexadecimalCode + decimalToHexa(B)
        return hexadecimalCode
    else:
        return "-1"    # If the hexadecimal color code does not exist, return -1

def make_scatterplot(df_temp, what_to_show_l, what_to_show_r, FROM, UNTIL,  show_month, smoothed):
    seaborn.set(style='ticks')

    what_to_show_l = what_to_show_l if type(what_to_show_l) == list else [what_to_show_l]
    what_to_show_r = what_to_show_r if type(what_to_show_r) == list else [what_to_show_r]
    colorlegenda=""
    if 1==1: #with _lock:
            fig1xy,ax = plt.subplots()

            x_ = np.array(df_temp[what_to_show_l])
            y_ = np.array(df_temp[what_to_show_r])
            cat_ = df_temp['year_month'].to_numpy()

            #we converting it into categorical data
            cat_col = df_temp['year_month'].astype('category')

            #we are getting codes for it
            cat_col_ = cat_col.cat.codes
            scatter = plt.scatter(x_, y_, c = cat_col_, label=cat_)

            legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best")
            ax.add_artist(legend1)


            # months = df_temp["year_month"].drop_duplicates()
            # months = df_temp['year_month'].astype('category')
            # df_temp['year_month'] = df_temp['year_month'].astype('category')
            # fg = seaborn.FacetGrid(data=df_temp, hue='year_month', hue_order=months, aspect=1.61)
            # # fg.map(pyplot.scatter, df_temp[what_to_show_l],df_temp[what_to_show_r]).add_legend()

            # #fg.map(pyplot.scatter, x_, y_).add_legend()
            # fg.map(plt.scatter, x_, y_).add_legend()

            plt.show()
            #obtain m (slope) and b(intercept) of linear regression line


            idx = np.isfinite(x_) & np.isfinite(y_)
            m, b = np.polyfit(x_[idx], y_[idx], 1)
            model = np.polyfit(x_[idx], y_[idx], 1)

            predict = np.poly1d(model)
            r2 = r2_score  (y_[idx], predict(x_[idx]))

            # De kolom 'R square' is een zogenaamde goodness-of-fit maat.
            # Deze maat geeft uitdrukking aan hoe goed de geobserveerde data clusteren rond de geschatte regressielijn.
            # In een enkelvoudige lineaire regressie is dat het kwadraat van de correlatie.
            # De proportie wordt meestal in een percentage ‘verklaarde variantie’ uitgedrukt.
            #  In dit voorbeeld betekent R square dus dat de totale variatie in vetpercentages voor 66% verklaard
            #    kan worden door de lineaire regressie c.q. de verschillen in leeftijd.
            # https://wikistatistiek.amc.nl/index.php/Lineaire_regressie

            #print (r2)
            #m, b = np.polyfit(x_, y_, 1)
            # print (m,b)

            #add linear regression line to scatterplot
            plt.plot(x_, m*x_+b, 'r')

            if smoothed:
                title_scatter = (f"Smoothed: {what_to_show_l[0]} -  {what_to_show_r[0]}\n({FROM} - {UNTIL})\nCorrelation = {find_correlation_pair(df_temp, what_to_show_l, what_to_show_r)}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")
            else:
                title_scatter = (f"{what_to_show_l[0]} -  {what_to_show_r[0]}\n({FROM} - {UNTIL})\nCorrelation = {find_correlation_pair(df_temp, what_to_show_l, what_to_show_r)}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")


            plt.title(title_scatter)


            ax.text(
                1,
                1.1,
                "Created by Rene Smit — @rcsmit",
                transform=ax.transAxes,
                fontsize="xx-small",
                va="top",
                ha="right",
            )
            st.pyplot(fig1xy)





            fig1xyz = px.scatter(df_temp, x=what_to_show_l[0], y=what_to_show_r[0],  color =cat_col,hover_data=["date"],
                                trendline="ols", trendline_scope = 'overall',trendline_color_override = 'black'
                    )

            correlation_sp = round(df_temp[what_to_show_l[0]].corr(df_temp[what_to_show_r[0]], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
            correlation_p = round(df_temp[what_to_show_l[0]].corr(df_temp[what_to_show_r[0]], method='pearson'), 3)

            title_scatter_plotly = (f"{what_to_show_l[0]} -  {what_to_show_r[0]}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

            fig1xyz.update_layout(
                title=dict(
                    text=title_scatter_plotly,
                    x=0.5,
                    y=0.95,
                    font=dict(
                        family="Arial",
                        size=14,
                        color='#000000'
                    )
                ),
                xaxis_title=what_to_show_l[0],
                yaxis_title=what_to_show_r[0],

            )

            ax.text(
                1,
                1.3,
                "Created by Rene Smit — @rcsmit",
                transform=ax.transAxes,
                fontsize="xx-small",
                va="top",
                ha="right",
            )


            st.plotly_chart(fig1xyz, use_container_width=True)

def smooth_columnlist(df, columnlist, t, WDW2, centersmooth):
    """Smooth columns (helpers.py)

    Args:
        df (df): dataframe
        columnlist (list): columns to smooth
        t (string): SMA or Savgol
        WDW2 (int): window for smoothing
        centersmooth (Boolean): Smooth in center or not

    Returns:
        df: the df
        c_smoothen : list withthe names of the smoothed columns
    """

    c_smoothen = []
    wdw_savgol = 7
    #if __name__ = "covid_dashboard_rcsmit":
    # global WDW2, centersmooth, show_scenario
    # WDW2=7
    # st.write(__name__)
    # centersmooth = False
    # show_scenario = False

    if columnlist is not None:
        columnlist_ = columnlist
        for c in columnlist_:
            print(f"Smoothening {c}")
            if t == "SMA":
                new_column = c + "_SMA_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = (
                    df.iloc[:, df.columns.get_loc(c)]
                    .rolling(window=WDW2, center=centersmooth)
                    .mean()
                )

            elif t == "savgol":
                new_column = c + "_savgol_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = df[c].transform(lambda x: savgol_filter(x, WDW2, 2))

            elif t is None:
                new_column = c + "_unchanged_"
                df[new_column] = df[c]
                print("Added " + new_column + "...~")
            else:
                print("ERROR in smooth_columnlist")
                st.stop()
            c_smoothen.append(new_column)
    return df, c_smoothen



def add_walking_r(df, smoothed_columns, datefield, how_to_smooth, window_smooth, tg,d):
    """  helpers.py : Calculate walking R from a certain base. Included a second methode to calculate R
    de rekenstappen: (1) n=lopend gemiddelde over 7 dagen; (2) Rt=exp(Tc*d(ln(n))/dt)
    met Tc=4 dagen, (3) data opschuiven met rapportagevertraging (10 d) + vertraging
    lopend gemiddelde (3 d).
    https://twitter.com/hk_nien/status/1320671955796844546
    https://twitter.com/hk_nien/status/1364943234934390792/photo/1


    Args:
        df (df): the dataframe
        smoothed_columns (list): list with the smoothed columns to calculate R over
        datefield (string): datefield
        how_to_smooth (string): "SMA"/ "savgol"
        tg (int): generation time
        d (int): look back d days
    Returns:
        df (df): df with sm. columns
        column_list_r_smoothened : columnlist
    """
    WDW2 = window_smooth
    column_list_r_smoothened = []
    for base in smoothed_columns:
        column_name_R = "R_value_from_" + base + "_tg" + str(tg)

        column_name_r_smoothened = (
            "R_value_from_"
            + base
            + "_tg"
            + str(tg)
            + "_"
            + "over_" + str(d) + "_days_"
            + how_to_smooth
            + "_"
            + str(WDW2)
        )

        sliding_r_df = pd.DataFrame(
            {"date_sR": [], column_name_R: []}
        )
        rows = []

        for i in range(len(df)):
            if df.iloc[i][base] != None:
                date_ = pd.to_datetime(df.iloc[i][datefield], format="%Y-%m-%d")
                date_ = df.iloc[i][datefield]
                if df.iloc[i - d][base] != 0 or df.iloc[i - d][base] is not None:
                    slidingR_ = round(
                        ((df.iloc[i][base] / df.iloc[i - d][base]) ** (tg / d)), 2
                    )

                else:
                    slidingR_ = None

                # Initialize an empty list to collect rows
                

                # Assuming you are inside a loop where date_ and slidingR_ are defined
                rows.append({
                    "date_sR": date_,
                    column_name_R: slidingR_
                })

                # After the loop, create a DataFrame from the collected rows
        sliding_r_df = pd.DataFrame(rows)

        sliding_r_df[column_name_r_smoothened] = round(
            sliding_r_df.iloc[:, 1].rolling(window=WDW2, center=True).mean(), 2
        )


        sliding_r_df = sliding_r_df.reset_index()
        
        df = pd.merge(
                df,
                sliding_r_df,
                how="outer",
                left_on=datefield,
                right_on="date_sR",
                suffixes=('_left', '_right')  # Specify suffixes for duplicate columns
            )
        column_list_r_smoothened.append(column_name_r_smoothened)
        # Drop the unwanted columns
        df = df.drop(columns=['index', 'date_sR'], errors='ignore')

        sliding_r_df = sliding_r_df.reset_index()
    return df, column_list_r_smoothened

def find_color_x(firstday, showoneday, showday):
    COLOR_weekday = "#3e5c76"  # blue 6
    COLOR_weekend = "#e49273"  # dark salmon 7
    bittersweet = "#ff6666"  # reddish 0
    white = "#eeeeee"
    if firstday == 0:
        color_x = [
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekend,
            COLOR_weekend,
        ]
    elif firstday == 1:
        color_x = [
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekend,
            COLOR_weekend,
            COLOR_weekday,
        ]
    elif firstday == 2:
        color_x = [
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekend,
            COLOR_weekend,
            COLOR_weekday,
            COLOR_weekday,
        ]
    elif firstday == 3:
        color_x = [
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekend,
            COLOR_weekend,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
        ]
    elif firstday == 4:
        color_x = [
            COLOR_weekday,
            COLOR_weekend,
            COLOR_weekend,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
        ]
    elif firstday == 5:
        color_x = [
            COLOR_weekend,
            COLOR_weekend,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
        ]
    elif firstday == 6:
        color_x = [
            COLOR_weekend,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekday,
            COLOR_weekend,
        ]

    if showoneday:
        if showday == 0:
            color_x = [
                bittersweet,
                white,
                white,
                white,
                white,
                white,
                white,
            ]
        elif showday == 1:
            color_x = [
                white,
                bittersweet,
                white,
                white,
                white,
                white,
                white,
            ]
        elif showday == 2:
            color_x = [
                white,
                white,
                bittersweet,
                white,
                white,
                white,
                white,
            ]
        elif showday == 3:
            color_x = [
                white,
                white,
                white,
                bittersweet,
                white,
                white,
                white,
            ]
        elif showday == 4:
            color_x = [
                white,
                white,
                white,
                white,
                bittersweet,
                white,
                white,
            ]
        elif showday == 5:
            color_x = [
                white,
                white,
                white,
                white,
                white,
                bittersweet,
                white,
            ]
        elif showday == 6:
            color_x = [
                white,
                white,
                white,
                white,
                white,
                white,
                bittersweet,
            ]
    return color_x


@st.cache_data(ttl=60 * 60 * 24)
def getdata_knmi():
    with st.spinner(f"GETTING ALL DATA ..."):
        #url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        #url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=260&vars=TEMP:Q:UN:UX:UG:SQ:RH&start=20200101"
        try:
            df = pd.read_csv(url, delimiter=",", header=None,  comment="#",low_memory=False,)

            column_replacements =  [[0, 'STN'],
                                [1, 'YYYYMMDD'],
                                [2, 'temp_etmaal'],
                                [3, 'temp_min'],
                                [4, 'temp_max'],
                                [5, 'T10N'],
                                [6, 'globale_straling'],
                                [7, 'RH_min'],
                                [8, 'RH_max'],
                                [9, 'RH_avg'],
                                [10, 'zonneschijnduur'],
                                [11, 'neerslag']]

            for c in column_replacements:
                df = df.rename(columns={c[0]:c[1]})

            df["date_knmi"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
            to_divide_by_10 = ["temp_etmaal", "temp_min", "temp_max", "neerslag", "zonneschijnduur"]
            for d in to_divide_by_10:
                df[d] = df[d]/10

        except:
            st.write("FOUT BIJ HET INLADEN.")






    return df