from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import streamlit as st
import matplotlib.cm as cm
from sklearn.metrics import r2_score

def cell_background(val):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
    Args:
        val ([int]): the value of the cell

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

def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"

    mask = (df["Date_statistics"].dt.date >= show_from) & (df["Date_statistics"].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
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
    what_to_show_l = what_to_show_l if type(what_to_show_l) == list else [what_to_show_l]
    what_to_show_r = what_to_show_r if type(what_to_show_r) == list else [what_to_show_r]
    colorlegenda=""
    with _lock:
            fig1xy = plt.figure()
            ax = fig1xy.add_subplot(111)

            if show_month==True:
                num_months = (UNTIL.year - FROM.year) * 12 + (UNTIL.month - FROM.month)
                colors=cm.rainbow(np.linspace(0,1,num_months+1))

                for y in range (2020,2022):
                    for m,c in zip(range (1,13),colors):


                        df_temp_month = df_temp[(df_temp['date'].dt.month==m) & (df_temp['date'].dt.year==y)]
                        x__ = df_temp_month[what_to_show_l].values.tolist()
                        y__ = df_temp_month[what_to_show_r].values.tolist()


                        plt.scatter(x__, y__,  s=2,color=c)

                #         r,g,b,z = c
                #         colorlegenda += (f"<font color ='{RGBtoHexConverion(int(r*255),int(g*255),int(b*255))}'>{m}-{y}</font> | ")
                # st.markdown(colorlegenda, unsafe_allow_html=True)

            else:
                x_ = np.array(df_temp[what_to_show_l])
                y_ = np.array(df_temp[what_to_show_r])


                plt.scatter(x_, y_)

            x_ = np.array(df_temp[what_to_show_l])
            y_ = np.array(df_temp[what_to_show_r])



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