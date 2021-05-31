import csv
import pandas as pd
import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt
# contact matrix retrieved from
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text
# https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material

def main():
        # average number of contacts perday
    st.header ("CONTACTMATRIXES")
    st.write ("Retrieved from https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.full-text")
    st.write ("https://www.medrxiv.org/content/10.1101/2020.05.18.20101501v1.supplementary-material")
    st.write ("part_age = participant age, cont_age = contact age")
    contact_type  = st.sidebar.selectbox("All, community or household", ["all", "community", "household"], index=0)

    df= pd.read_csv(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/contactmatrix.tsv",
            # "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\contactmatrix.tsv",
            comment="#",
            delimiter="\t",
            low_memory=False,
        )
    # st.write (df)
    df = df.replace("[5,10)", "[05,10)")

    #contact_type = "community" #household"  # community  all
    df_baseline =  df[(df['survey'] == "baseline"           ) & (df['contact_type'] == contact_type)]
    df_phys_dist = df[(df['survey'] == "physical distancing") & (df['contact_type'] == contact_type)]

    df_baseline_pivot =  df_baseline.pivot_table(index='part_age', columns='cont_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader("Contactmatrix 2016/-17")
    st.write (df_baseline_pivot)
    df__phys_dist_pivot =  df_phys_dist.pivot_table(index='part_age', columns='cont_age', values="m_est", margins = True, aggfunc=sum)
    st.subheader("Contactmatrix april 2020")
    st.write (df__phys_dist_pivot)

    #list_baseline = df_baseline_pivot.values.tolist()
    #st.write (list_baseline)

    st.subheader ("Verschil als ratio -- (nieuw/oud")
    result =  df__phys_dist_pivot / df_baseline_pivot
    st.write (result)
    fig, ax = plt.subplots()
    sn.heatmap(result, ax=ax)
    st.write(fig)


    st.subheader("Verschil als percentage -- ((oud-nieuw)/oud)*100")
    result_perc =   (df_baseline_pivot - df__phys_dist_pivot) / df_baseline_pivot*100
    st.write (result_perc)


if __name__ == "__main__":
    main()
