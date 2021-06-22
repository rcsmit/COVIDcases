import pandas as pd
import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt
# contact matrix retrieved from
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material
# https://www.eurosurveillance.org/docserver/fulltext/eurosurveillance/26/8/20-00994_BACKER_Supplement2.pdf?expires=1622904589&id=id&accname=guest&checksum=D4271340B23924AA59899E444B283F63

def calculate_total(df, df_name, contact_type):
        #       0-4   5-9     10-19   20-29  30-39   40-49   50-59   60-69   70-79  80+
    #pop_ =      [857000, 899000 , 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [ 0.04907, 0.05148, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]

    all1 = df['All'].tolist()
    total,total2 = 0,0
    st.markdown (f"Gemiddeld aantal contacten  per persoon _{contact_type}_ _{df_name}_ (gewogen naar populatiefractie)")
    for n in range(len(all1)-1):
        total += (all1[n]*fraction[n])
    st.markdown (f"Van boven naar beneden  __{round(total,2)}__")

    all2 = df.loc[df.index == "All"].values.flatten().tolist()

    for n in range(len(all2)-1):
        total2 += (all2[n]*fraction[n])
    st.markdown (f"Van links naar rechts __{round(total2,2)}__")
    st.markdown (f"Gemiddeld van beide __{round(((total+total2)/2),2)}__")




def main():

        # average number of contacts perday
    st.header ("CONTACTMATRIXES")
    contact_type  = st.sidebar.selectbox("All, community or household", ["all", "community", "household"], index=0)
    df1  = st.sidebar.selectbox("First dataframe", ["2016/-17", "April2020", "June2020"], index=0)
    df2  = st.sidebar.selectbox("Second dataframe",["2016/-17", "April2020", "June2020"], index=1)
    #test 13:40  14:49
    df= pd.read_csv(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/contactmatrix.tsv",
            # "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\contactmatrix.tsv",
            comment="#",
            delimiter="\t",
            low_memory=False,
        )

    df = df.replace("[5,10)", "[05,10)")
    df = df.replace ("baseline", "2016/-17")
    df = df.rename(columns={'part_age':'participant_age'})
    df = df.rename(columns={'cont_age':'contact_age'})


    #contact_type = "community" #household"  # community  all
    df_first =  df[(df['survey'] == df1) & (df['contact_type'] == contact_type)]
    df_second = df[(df['survey'] == df2) & (df['contact_type'] == contact_type)]

    st.write ("Rijlabels = contact age / Kolomlabels = participant age")

    df_first_pivot =  df_first.pivot_table(index='contact_age', columns='participant_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader(f"Contactmatrix {df1}")
    st.write (df_first_pivot)
    calculate_total (df_first_pivot, df1, contact_type)

    df_second_pivot =  df_second.pivot_table(index='contact_age', columns='participant_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader(f"Contactmatrix {df2}")
    st.write (df_second_pivot)
    calculate_total (df_second_pivot, df2, contact_type)


    st.subheader (f"Verschil als ratio -- ({df2}/{df1}")
    df_difference_as_ratio =  df_second_pivot / df_first_pivot

    st.write (df_difference_as_ratio)
    fig, ax = plt.subplots()
    #max  = result.to_numpy().mean() + ( 1* result.to_numpy().std())
    max_value = st.sidebar.number_input("Max value heatmap", 0, None,2)

    sn.heatmap(df_difference_as_ratio, ax=ax,  vmax=max_value)
    st.write(fig)



    all_first = df_first_pivot['All'].tolist()
    all_second = df_second_pivot['All'].tolist()
    all_diff_ratio = df_difference_as_ratio['All'].tolist()
    del all_first[-1]
    del all_second[-1]
    del all_diff_ratio[-1]
    age_groups = ["0-4",  "5-9",  "10-19",  "20-29",  "30-39",  "40-49",  "50-59",  "60-69",  "70-79",  "80+"]
    relative_s_2_i = [ 1.000, 1.000, 3.051, 5.751, 3.538, 3.705, 4.365, 5.688, 5.324, 7.211]

    s = sum(
        round((all_diff_ratio[i] * relative_s_2_i[i]) / len(age_groups), 2)
        for i in range(len(age_groups) - 1)
    )

    # st.write (f"Relative reduction of contacs = {s}")

    st.subheader(f"Verschil als percentage -- ({df1}-{df2})/{df1} * 100" )
    df_difference_as_perc =   (df_first_pivot - df_second_pivot) / df_first_pivot*100
    st.write (df_difference_as_perc)

    fig2a = plt.figure(facecolor='w')
    ax = fig2a.add_subplot(111, axisbelow=True)
    ax.plot (age_groups, all_first, label= df1)
    ax.plot (age_groups, all_second, label = df2)

    plt.legend()
    plt.title("Average contacts per person per day")

    st.pyplot(fig2a)


    with st.sidebar.beta_expander('Data sources',  expanded=False):
        #st.write ("Retrieved from https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text")

        #st.write ("https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material")
        st.write ("Retrieved from https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.8.2000994#f4")
        st.write ("https://www.eurosurveillance.org/docserver/fulltext/eurosurveillance/26/8/20-00994_BACKER_Supplement2.pdf?expires=1622904589&id=id&accname=guest&checksum=D4271340B23924AA59899E444B283F63")

if __name__ == "__main__":
    main()
