# Grafiek positief getest naar leeftijd door de tijd heen, per leeftijdscategorie
# René Smit, (@rcsmit) - MIT Licence

# IN: tabel met positief aantal testen en totaal aantal testen per week, gecategoriseerd naar leeftijd
#     handmatig overgenomen uit Tabel 14 vh wekelijkse rapport van RIVM
#     Wekelijkse update epidemiologische situatie COVID-19 in Nederland
#     https://www.rivm.nl/coronavirus-covid-19/actueel/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland
#
#     Needs file generatied with https://github.com/rcsmit/COVIDcases/blob/main/grafiek_pos_testen_per_leeftijdscategorie_PREPARE.py
#
# Uitdagingen : Kopieren en bewerken Tabel 14. 3 verschillende leeftijdsindelingen. Tot dec. 2020 alles
# cummulatief. X-as in de grafiek

# TODO : - Nog enkele weken toevoegen voorafgaand het huidige begin in de datafile (waren weken met weinig besmettingen).
#        - integreren in het dashboard
#        - 'Total reported' toevoegen


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from labellines import *  #https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
import streamlit as st
#from streamlit import caching
# from matplotlib.backends.backend_agg import RendererAgg
from datetime import datetime
import grafiek_pos_testen_per_leeftijdscategorie_PREPARE
# _lock = RendererAgg.lock
import perprovincieperleeftijd

def save_df(df,name):
    """  _ _ _ """
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\output\\'

    name_ =  OUTPUT_DIR + name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)
    try:
        df.to_csv(name_, index=False,
                compression=compression_opts)

        print ("--- Saving "+ name_ + " ---" )
    except:
        print("NOT saved")
@st.cache_data(ttl=60 * 60 * 24)
def read_df( kids_split_up):
    # Local file
    # url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\covid19_seir_models\input\pos_test_leeftijdscat_wekelijks.csv"
    # File before regroping the agecategories. Contains ; as delimiter and %d-%m-%Y as dateformat
    # url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/pos_test_leeftijdscat_wekelijks.csv"


    if kids_split_up:
        url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/input_latest_age_pos_test_kids_seperated.csv"
    else:
        url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/input_latest_age_pos_tests.csv"

    #id;datum;leeftijdscat;methode;mannen_pos;mannen_getest;vrouwen_pos ;vrouwen_getest ;
    # totaal_pos;totaal_getest;weeknr2021;van2021;tot2021

    df_new   = pd.read_csv(url,
                        delimiter=",",
                        low_memory=False)

    #df_new = generate_aantallen_gemeente_per_dag_grouped_per_day()

    df_new["date"]=pd.to_datetime(df_new["date"], format='%Y-%m-%d')
    return df_new

def real_action( ages_to_show_in_graph, what_to_show_l,what_to_show_r, kids_split_up, show_from, show_until):
    df_= read_df( kids_split_up)
    save_df (df_, "viertottwaalf")
    df_new = df_.copy(deep=False)
    df_new["date"]=pd.to_datetime(df_new["date"], format='%Y-%m-%d')
    df_new['percentage'] =  round((df_new['positief_testen']/df_new['totaal_testen']*100),1)

    datumveld = 'date'
    try:
        startdate = pd.to_datetime(show_from).date()
        enddate = pd.to_datetime(show_until).date()
        mask = (df_new[datumveld].dt.date >= startdate) & (df_new[datumveld].dt.date <= enddate)
        df_new = df_new.loc[mask]
    except:
        st.error("Please make sure that the dates in format yyyy-mm-dd")
        st.stop()


    print (f'Totaal aantal positieve testen : {df_new["positief_testen"].sum()}')
    print (f'Totaal aantal testen : {df_new["totaal_testen"].sum()}')
    print (f'Percentage positief  : {  round (( 100 * df_new["positief_testen"].sum() /  df_new["totaal_testen"].sum() ),2)    }')

    show_graph_ages_percentage(df_new, ages_to_show_in_graph, what_to_show_l, what_to_show_r)


def generate_aantallen_gemeente_per_dag_grouped_per_day():
    # Local file
    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_aantallen_gemeente_per_dag.csv"
    url = "https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv"
    df_new   = pd.read_csv(url, delimiter=";", low_memory=False)
    df_new["Date_of_publication"]=pd.to_datetime(df_new["Date_of_publication"], format='%Y-%m-%d')
    df_new = df_new.groupby(['Date_of_publication'], sort=True).sum().reset_index()
    return df_new

def read_cases_day():
    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_aantallen_gemeente_per_dag_grouped_per_day.csv"
    url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/COVID-19_aantallen_gemeente_per_dag_grouped_per_day.csv"
    df_new   = pd.read_csv(url,  delimiter=",", low_memory=False)
    df_new["Date_of_publication"]=pd.to_datetime(df_new["Date_of_publication"], format='%Y-%m-%d')

    df_new = df_new.groupby([pd.Grouper(key='Date_of_publication', freq='W-TUE')]).sum().reset_index().sort_values('Date_of_publication')

    return df_new
def show_graph_ages_percentage(df_new, ages_to_show_in_graph, what_to_show_l, what_to_show_r):

    color_list = [  "#ff6666",  # reddish 0
                    "#ac80a0",  # purple 1
                    "#3fa34d",  # green 2
                    "#EAD94C",  # yellow 3
                    "#EFA00B",  # orange 4
                    "#7b2d26",  # red 5
                    "#3e5c76",  # blue 6
                    "#e49273" , # dark salmon 7
                    "#1D2D44",  # 8
                    "#02A6A8",
                    "#4E9148",
                    "#F05225",
                    "#024754",
                    "#FBAA27",
                    "#302823",
                    "#F07826",
                     ]
    list_age_groups =  df_new["cat_nieuw"].unique()
    #df_new = df.shift(-2, freq = "D")
    df_cases = read_cases_day()
    df_temp_x = df_new
    df_new = pd.merge(
                df_new, df_cases, how="inner", left_on="date", right_on="Date_of_publication"
            )


    # with _lock:
    if 1==1:
        fig1y = plt.figure()
        ax = fig1y.add_subplot(111)
        #for l in ages_to_show_in_graph:
        ax.set_xticks(df_new["date"].index)
        ax.xaxis_date()
        #for i,l in enumerate(ages_to_show_in_graph):
        for l in ages_to_show_in_graph:
            df_temp = df_new[df_new['cat_nieuw']==l]
            df_temp.set_index('date', inplace=True, drop=True ) #
            #df_temp["date"]=pd.to_datetime(df_temp["date"], format='%Y-%m-%d')
            df_temp.sort_values(by = 'date')
            print (df_temp)

            #print (df_temp)
            #print (len(df_temp))
            #list_dates = df_temp["date"].tolist()
            if what_to_show_l== "percentage positieve testen":
                list_percentage = df_temp["percentage"].tolist()
                ax =  df_temp["percentage"].plot( label = l, marker = ".")
                #ax =  df_temp.plot(x="date", y = "percentage", label = l, marker = ".")
                # print(list_dates)
                # print (list_percentage)
                # plt.plot(list_dates, list_percentage,  label = l)
            elif what_to_show_l== "aantal positieve testen":
                list_percentage = df_temp["positief_testen"].tolist()
                ax =  df_temp["positief_testen"].plot(label = l, marker = ".")
            elif what_to_show_l== "totaal testen gedaan":
                list_percentage = df_temp["totaal_testen"].tolist()
                ax =  df_temp["totaal_testen"].plot(label = l, marker = ".")
            else:
                print ("error")

            #print (list_dates)

            #ax =  df_temp["percentage"].plot(label = l)
            #plt.plot(list_dates, list_percentage, color = color_list[i], label = l)
        #labelLines(plt.gca().get_lines(),align=False,fontsize=8)
        if what_to_show_r != None:
            ax3 = df_new[what_to_show_r].plot.bar( secondary_y=True, color ="r", alpha=.3, label = what_to_show_r)



        #ax.set_xticklabels(df_new["date"].dt.date, fontsize=6, rotation=90)
        # xticks = ax.xaxis.get_major_ticks()
        # for i, tick in enumerate(xticks):
        #     if i % 10 != 0:
        #         tick.label1.set_visible(False)
        #plt.xticks()

        ax.yaxis.grid(True, which="major", alpha=0.4)

        ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
                    transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
        plt.title(f"{what_to_show_l} per agegroup" , fontsize=10)
        #plt.legend(bbox_to_anchor=(1.5, 1))
        plt.tight_layout()


        # hand, labl = ax.get_legend_handles_labels()
        # handout=[]
        # lablout=[]
        # for h,l in zip(hand,labl):
        #     if l not in lablout:
        #             lablout.append(l)
        #             handout.append(h)
        # fig1y.legend(handout, lablout, bbox_to_anchor=(1.2, 1))

        handles, labels = [], []
        for ax in fig1y.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles,labels)
        plt.show()
        st.pyplot(fig1y)

    #with _lock:
    testmeout = False
    #TODO : New graph with grouped barcharts
    if testmeout:
        st.write(df_temp_x)

        df_pivot_0 = df_temp_x.pivot_table( values='percentage', index=['date'],
                    columns=['cat_nieuw'], aggfunc=np.sum,  fill_value=0)

        df_pivot = df_temp_x.pivot_table( values='percentage', index=['date'],
                    columns=['cat_nieuw'], aggfunc=np.sum,  fill_value=0).reset_index()
        st.write(df_pivot_0)
        df_pivot_transpose = df_pivot.transpose()

        df_pivot_transpose.columns = df_pivot_transpose.iloc[0]
        df_pivot_transpose = df_pivot_transpose.drop('date')
        st.write(df_pivot_transpose)
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        #for l in ages_to_show_in_graph:
        #ax = df_pivot.plot.bar()
        # if what_to_show_r != None:
        #     ax3 = df_new[what_to_show_r].plot.bar( secondary_y=True, color ="r", alpha=.3, label = what_to_show_r)
        # aantaldata = len(df_new)
        aantaldata= len(ages_to_show_in_graph)
        ax = df_pivot_transpose.plot.bar()
        width = 1

        for i,l in enumerate(ages_to_show_in_graph):

            a = i/aantaldata
            #ax.bar(df_pivot.index+ (i*width), df_pivot[l], width, align = 'center')
            ax = df_pivot.plot.bar(rot=0)
            #ax.set_xticks(df_pivot.index + width) # sets the x-ticks to the middle of the cluster of bars

            list_dates = df_pivot["date"].tolist()




                    # if what_to_show_l== "percentage positieve testen":
                    #     list_percentage = df_temp["percentage"].tolist()
                    #     ax =  df_temp["percentage"].plot(label = l)
                    # elif what_to_show_l== "aantal positieve testen":
                    #     list_percentage = df_temp["positief_testen"].tolist()
                    #     ax =  df_temp["positief_testen"].plot(label = l)
                    # elif what_to_show_l== "totaal testen gedaan":
                    #     list_percentage = df_temp["totaal_testen"].tolist()
                    #     ax =  df_temp["totaal_testen"].plot(label = l)
                    # else:
                    #     print ("error")

                    #print (list_dates)

                    # ax =  df_temp["percentage"].plot(label = l)
                    #plt.plot(list_dates, list_percentage, color = color_list[i], label = l)
        #labelLines(plt.gca().get_lines(),align=False,fontsize=8)

        # ticks = [tick.get_text() for tick in ax.get_xticklabels()]
        # ticks = pd.to_datetime(ticks).strftime('%Y-%m-%d')
        # ax.set_xticklabels(ticks)

        # ax.set_xticks(df_new["date"].index)
        # ax.set_xticklabels(df_new["date"].dt.date, fontsize=6, rotation=90)
        # xticks = ax.xaxis.get_major_ticks()
        # for i, tick in enumerate(xticks):
        #     if i % 10 != 0:
        #         tick.label1.set_visible(False)
        # plt.xticks()

        ax.yaxis.grid(True, which="major", alpha=0.4)

        ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
                    transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
        plt.title(f"{what_to_show_l} per agegroup" , fontsize=10)
        plt.legend(bbox_to_anchor=(1.5, 1))
        plt.tight_layout()
        # handles, labels = [], []
        # for ax in fig1x.axes:
        #     for h, l in zip(*ax.get_legend_handles_labels()):
        #         handles.append(h)
        #         labels.append(l)
        plt.legend(handles,labels)
        #plt.show()
        st.pyplot(fig1x)

def prepare_files():
    # ONLY TO RUN LOCALLY ! (doesnt upload the files to Github)
    print ("I am going to prepare")
    grafiek_pos_testen_per_leeftijdscategorie_PREPARE.regroup_df( True)
    grafiek_pos_testen_per_leeftijdscategorie_PREPARE.regroup_df( False)

def main_ages_percentage():
    prepare = False
    if prepare:
        prepare_files()
    # DATE_FORMAT = "%m/%d/%Y"
    start_ = "2020-11-01"
    today = datetime.today().strftime("%Y-%m-%d")
    # from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)
    show_from = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    # try:
    #     show_from = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    # except:
    #     st.error("Please make sure that the startdate is in format yyyy-mm-dd")
    #     st.stop()

    # until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)
    show_until = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)
    if show_until == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            
            st.success("Cache is NOT cleared")
    # try:
    #     show_until = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    # except:
    #     st.error("Please make sure that the enddate is in format yyyy-mm-dd")
    #     st.stop()

    # if FROM >= UNTIL:
    #     st.warning("Make sure that the end date is not before the start date")
    #     st.stop()
    kids_split_up = st.sidebar.selectbox(
        "Split up childres age categories", [True, False], index=1
    )
    if kids_split_up:
        lijst = ["0-12", "0-3", "04-12", "13-17",  "0-4",  "05-09",  "10-14",  "15-19", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld"]
    else:
        lijst = ["0-17", "0-19",  "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld"]

    ages_to_show_in_graph = st.sidebar.multiselect(
                "What to show  (multiple possible)", lijst, ["30-39", "40-49", "50-59", "60-69"])

    if ages_to_show_in_graph == []:
           st.error("Choose something for the left-axis")
           st.stop()
    st.sidebar.write("Onder de 18 heeft men 3x de groepsindeling veranderd.")
    to_show_l_list = ["aantal positieve testen","totaal testen gedaan", "percentage positieve testen"]

    to_show_r_list = [None, "Total_reported","Hospital_admission","Deceased"]
    what_to_show_l = st.sidebar.selectbox(
        "What to show left", to_show_l_list, index=2
    )
    what_to_show_r = None

    # what_to_show_r = st.sidebar.selectbox(
    #     "What to show right", to_show_r_list, index=1
    # )
    st.sidebar.write("NB: De weken lopen niet exact gelijk met de testen (ma-zo vs wo-di)")
    st.header ("Percentage positieve testen per leeftijdsgroep door de tijd heen")

    # Rerun when new weeks are added
    #regroup_df( ages_to_show_in_graph, kids_split_up)

    real_action ( ages_to_show_in_graph, what_to_show_l,what_to_show_r, kids_split_up, show_from, show_until)
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/grafiek_pos_testen_per_leeftijdscategorie_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)
    st.markdown('<br><br><br><br><br><br><br><br><br><br>', unsafe_allow_html=True)
    st.image(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
    )

    st.markdown(
        '<br><br>'
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<br><br><a href="https://www.linkedin.com/in/rcsmit" target="_blank">Contact me for custom dashboards and infographics</a>',
        unsafe_allow_html=True,
    )

def main():
    what_to_do = st.sidebar.selectbox(
        "WHAT TO DO", [ "Ages/percentage", "Per province"]
    )
    st.sidebar.write ("==============================")
    if what_to_do == "Ages/percentage":
        main_ages_percentage()
    else:
        perprovincieperleeftijd.main_per_province_per_leeftijd()



if __name__ == "__main__":
    main()
