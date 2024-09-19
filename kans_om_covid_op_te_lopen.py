import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
# from matplotlib.backends.backend_agg import RendererAgg
from helpers import cell_background_helper
# _lock = RendererAgg.lock

def bereken_kans_periode (kans_jaar, periode):
    return (1-(kans_jaar/100))**periode

def main():
    incidentie = st.sidebar.number_input("incidentie per 100k/week",  0, 100000,  100)
    ratio = st.sidebar.number_input("verhoudingen besmettingen/cases", 0.0, 10.0, 3.0)
    vaccinatiebonus= st.sidebar.number_input("Vaccinatiebonus",  0.0, 1.0, 0.85)
    gedragsbonus = st.sidebar.number_input("Gedragsbonus/-malus (1=neutraal)",  0.0, 10.0, 0.75)
    clusterfactor = st.sidebar.number_input("cluster-/locatiefactor (1=neutraal)",  0.0, 10.0,  1.0)

    kans_besm_100k_jaar = ratio * 52 * incidentie / 1000
    st.sidebar.write (f"Kans om besmet te worden per jaar {kans_besm_100k_jaar} %")
    kans_jaar = kans_besm_100k_jaar * vaccinatiebonus * gedragsbonus * clusterfactor
    st.sidebar.write (f"Kans om besmet te worden per jaar met bonussen : {kans_jaar}%")
    x_ax, y_ax = [],[]

    try:
        col1,col2 = st.columns([1,3])
        b = "alfa"
    except:
        col1,col2 = st.columns([1,3])
        b = "beta"

    for x in range (1,30,1):
        y = round((100 - (bereken_kans_periode(kans_jaar,x)*100)),2)
        x_ax.append(x)
        y_ax.append(y)

    d = {'jaren': x_ax, 'kans':y_ax}

    df_legenda = pd.DataFrame(data=d)
    df_legenda.set_index('jaren', inplace=True)
    with col1:
        st.write (f"Kans om >=1 keer COVID op te lopen in x jaar:" )
        st.write (df_legenda) #.styler) #.format(None, na_rep="-").applymap(lambda x:  cell_background_helper(x,"kwartiel", 100,None)).set_precision(1))


    with col2:
        if 1==1:
        # with _lock:
            fig1x = plt.figure()
            ax = fig1x.add_subplot(111)
            plt.plot(x_ax,y_ax)
            plt.grid()
            plt.title ("Kans om COVID op te lopen in x jaar")
            st.pyplot(fig1x)

    toelichting = ("<b>Incidentie</b> - Aantal gemelde cases per 100.000 inwoners per week<br>"
                "<b>Verhoudingen besmettingen/cases</b> - Wat is de verhouding tussen gemelde cases en daadwerkelijke besmettingen?<br>"

                    "<b>Vaccinatiebonus</b> - <i>Default : 0.85</i>. Dat betekent niet dat er maar 15% bescherming is, maar dat je hierdoor 15% minder risico loopt dan de gemiddelde Nederlander (waarvan straks misschien 70% gevaccineerd is) (1=neutraal)<br>"

                    "<b>Gedragsbonus</b> - <i>Default : 0,75</i> oftewel ik zal 25% minder vaak besmet raken dan de gemiddelde Nederlander (1=neutraal)<br>"

                    "<b>Cluster-/locatiefactor</b> - <i>Default 1</i>. Vergroting/ verlaging van kans door clusters of locatie (1=neutraal)<br>")

    st.subheader( "Parameters")
    st.markdown(toelichting, unsafe_allow_html=True)

    st.write ("Info over de parameters en berekening: https://twitter.com/roelgrif/status/1433215517901344771")
    
    # st.write(b)



if __name__ == "__main__":
    main()
  


