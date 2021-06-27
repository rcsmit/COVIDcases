import streamlit as st
import math


def rh2q(rh, t, p):
    # https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R#L45
    es = 6.112 * math.exp((17.67 * t) / (t + 243.5)) # Saturation Vapor Pressure
    e = es * (rh / 100) # vapor pressure
    q_ = (0.622 * e) / (p - (0.378 * e)) * 1000
    # In het dashboard: minimale relatieve luchtvochtigheid en de maximale temperatuur (per dag) van de 24 uurs waardes van De Bilt door het KNMI

    # More info:
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    # https://earthscience.stackexchange.com/questions/5076/how-to-calculate-specific-humidity-with-relative-humidity-temperature-and-pres
    # Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    return round(q_, 2)

def rh2ah(rh, t):
    # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
    return (6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t)

def main():
    st.header("Calculate specific and absolute humidity")
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        t = st.number_input("Temperature (Celcius)", None, None, 25)
    with col2:
        rh = st.number_input("Relative humidity (%)", None, None, 36)
    with col3:
        p = st.number_input("Pressure (mbar)", None, None, 1020)

    tekst = (f"<div style='background-color: lightblue;padding:20px;'>Specific humidity (q) = <b>{round(rh2q(rh, t, p ),1)}</b> g/kg<br><br>Absolute humidity  = <b>{round(rh2ah(rh, t),1)}</b> grams/m<sup>3</sup></div>")
    st.markdown(tekst, unsafe_allow_html=True)

    st.subheader("Formula for specific humidity")
    r'''$$es = 6.112 * e^{\frac{17.67 * t}{t + 243.5}} (e = 2,71828..)\\$$
	$$e = es * \frac{rh}{100}\\$$
	$$q = {\frac{0.622 * e}{p - (0.378 * e)}}*1000$$'''
    link =('<a href="https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R#L45" target="_blank">source</a>')
    st.markdown(link, unsafe_allow_html=True)

    st.subheader("Formula for absolute humidity")
    r'''$$ah = \frac{6.112 * e^{\frac{17.67 * t}{t + 243.5}} * rh * 2.1674}{273.15 + t} (e = 2,71828..)$$'''
    link2 =('<a href="https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/" target="_blank">source</a>')
    st.markdown(link2, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
