import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

def bereken_kans_periode (kans_jaar, periode):
    return (1-(kans_jaar/100))**periode

def main():

    ratio = st.sidebar.number_input("verhoudingen besmettingen/cases", 0, None, 3.0)
    incidentie = st.sidebar.number_input("incidentie per 100k/week",  0, None,  100.0)
    vaccinatiebonus= st.sidebar.number_input("Vaccinatiebonus",  0, 1, 0.85)
    gedragsbonus = st.sidebar.number_input("Gedragsbonus/-malus",  0, 10, 0.75)
    clusterfactor = st.sidebar.number_input("cluster-/locatiefactor",  0, 10,  1.0)

    kans_besm_100k_jaar = ratio * 52 * incidentie / 1000
    st.sidebar.write (f"Kans om besmet te worden per jaar {kans_besm_100k_jaar} %")
    kans_jaar = kans_besm_100k_jaar * vaccinatiebonus * gedragsbonus * clusterfactor
    st.sidebar.write (f"Kans om besmet te worden per jaar met bonussen : {kans_jaar}%")
    x_ax, y_ax = [],[]

    col1,col2 = st.columns([1,3])
    with col1:
        st.write (f"Kans om het >=1 keer op te lopen in x jaar:" )
        y = round((100 - (bereken_kans_periode(kans_jaar,1)*100)),2)
        st.write (f"1 jaar - {y}")
        x_ax.append(1)
        y_ax.append(y)



        for x in range (5,50,5):
            y = round((100 - (bereken_kans_periode(kans_jaar,x)*100)),2)

            st.write (f" {x} jaar - {y}")
            x_ax.append(x)
            y_ax.append(y)

    with col2:

        with _lock:
            fig1x = plt.figure()
            ax = fig1x.add_subplot(111)
            plt.plot(x_ax,y_ax)
            plt.grid()
            plt.title ("Kans om COVID op te lopen in x jaar")
            st.pyplot(fig1x)
    st.write ("Geinspireerd door Roel Griffioen: https://twitter.com/roelgrif/status/1433215573912080388")
main()



