import streamlit as st

def rh2q(rh, t, p ):
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    #Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    es = 6.112 * math.exp((17.67 * t)/(t + 243.5))
    e = es * (rh / 100)
    q_ = (0.622 * e)/(p - (0.378 * e)) * 1000
    return round(q_,2)

def rh2ah(rh, t ):
    return (6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t)

def main():
    rh =      st.sidebar.number_input(
            "Relative humidity %", None, None, 36
        )
    t =     total_cases_0 = st.sidebar.number_input(
            "Temperature (Celcius)", None, None, 8000
        )
    p =     total_cases_0 = st.sidebar.number_input(
            "Presure", None, None, 1020
        )


    st.write (f"Specific humidity (q) = {rh2q(rh, t, p )}")
    st.write (f"Absolute humidity  = {rh2ah(rh, t)}")

    st.subheader ("Formula for specific humidity")
    st.write ("es = 6.112 * math.exp((17.67 * t)/(t + 243.5))<br>e = es * (rh / 100)<br>q = (0.622 * e)/(p - (0.378 * e)) * 1000")
    st.subheader("Formula for absolute humidity")
    st.write ("ah = 6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t)")

if __name__ == "__main__":

    main()


