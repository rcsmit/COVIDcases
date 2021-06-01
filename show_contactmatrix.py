import pandas as pd
import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt
# contact matrix retrieved from
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material

def calculate_total(df):
        #       0-4   5-9     10-19   20-29  30-39   40-49   50-59   60-69   70-79  80+
    #pop_ =      [857000, 899000 , 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [ 0.04907, 0.05148, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]

    all1 = df['All'].tolist()
    total,total2 = 0,0
    for n in range(0, len(all1)-1):
        total += (all1[n]*fraction[n])
    st.write (f"Gemiddeld aantal contacten per persoon (gewogen naar populatiefractie) - totaal van links naar rechts {round(total,2)}")

    all2 = df.loc[df.index == "All"].values.flatten().tolist()

    for n in range(0, len(all2)-1):
        total2 += (all2[n]*fraction[n])
    st.write (f"Gemiddeld aantal contacten per persoon (gewogen naar populatiefractie) - totaal van boven naar beneden {round(total2,2)}")



def main():

        # average number of contacts perday
    st.header ("CONTACTMATRIXES")
    st.write ("Retrieved from https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text")
    st.write ("https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material")
    st.write ("participant_age = participant age, contact_agee = contact age")
    contact_type  = st.sidebar.selectbox("All, community or household", ["all", "community", "household"], index=0)
    #test 13:40  14:49
    df= pd.read_csv(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/contactmatrix.tsv",
            # "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\contactmatrix.tsv",
            comment="#",
            delimiter="\t",
            low_memory=False,
        )
    # st.write (df)
    df = df.replace("[5,10)", "[05,10)")
    df = df.rename(columns={'part_age':'participant_age'})
    df = df.rename(columns={'cont_age':'contact_age'})

    #contact_type = "community" #household"  # community  all
    df_baseline =  df[(df['survey'] == "baseline"           ) & (df['contact_type'] == contact_type)]
    df_phys_dist = df[(df['survey'] == "physical distancing") & (df['contact_type'] == contact_type)]
    #print (df_baseline.dtypes)
    df_baseline_pivot =  df_baseline.pivot_table(index='participant_age', columns='contact_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader("Contactmatrix 2016/-17")
    st.write (df_baseline_pivot)
    calculate_total (df_baseline_pivot)

    df_phys_dist_pivot =  df_phys_dist.pivot_table(index='participant_age', columns='contact_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader("Contactmatrix april 2020")
    st.write (df_phys_dist_pivot)
    calculate_total (df_phys_dist_pivot)
    #list_baseline = df_baseline_pivot.values.tolist()
    #st.write (list_baseline)

    st.subheader ("Verschil als ratio -- (nieuw/oud")
    result =  df_phys_dist_pivot / df_baseline_pivot
    st.write (result)
    fig, ax = plt.subplots()
    #max  = result.to_numpy().mean() + ( 1* result.to_numpy().std())
    max_value = st.sidebar.number_input("Max value heatmap", 0, None,2)

    sn.heatmap(result, ax=ax,  vmax=max_value)
    st.write(fig)


    st.subheader("Verschil als percentage -- ((oud-nieuw)/oud)*100")
    result_perc =   (df_baseline_pivot - df_phys_dist_pivot) / df_baseline_pivot*100
    st.write (result_perc)

    all_baseline = df_baseline_pivot['All'].tolist()
    all_phys_dist = df_phys_dist_pivot['All'].tolist()
    del all_baseline[-1]
    del all_phys_dist[-1]
    age_groups = ["0-4",  "5-9",  "10-19",  "20-29",  "30-39",  "40-49",  "50-59",  "60-69",  "70-79",  "80+"]
    fig2a = plt.figure(facecolor='w')
    ax = fig2a.add_subplot(111, axisbelow=True)
    ax.plot (age_groups, all_baseline, label="2016/-17")
    ax.plot (age_groups, all_phys_dist, label = "april 2020")

    plt.legend()
    plt.title("Average contacts per person per day")

    st.pyplot(fig2a)
if __name__ == "__main__":
    main()
