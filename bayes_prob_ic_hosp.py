import streamlit as st


ziekenhuis = [[ 2105, 68, 144],
              [ 2105, 120, 808],
		      [ 1543, 79, 2156]]

icopnames = [[ 431,11,11],
            [ 674,29,169],
		    [ 265,9,178]]

leeftijdsgr = ["12-49", "50-59", ">=70"]
aantal_inw =[8406602, 4691222,2424970]
aantal_vax = [6118208, 4144429, 2222492]
aantal_inw_groter_dan_12 = 15522794
aantal_inw_tot = 17475414



def calculate(event, event_txt, follow_txt):
    totaal_aantal_event = 0
    for k in range(0,3):
        for l in range(0,3):
            totaal_aantal_event += event[k][l] 
    st.header(event_txt)
    
    for i in range(0,3):
       
        p_event = totaal_aantal_event / aantal_inw_tot
        leeftijdsgr_txt = leeftijdsgr[i]
        st.subheader (f"{event_txt} - {leeftijdsgr[i]}")
        p_leeftijdsgr = aantal_inw[i]/aantal_inw_tot

        p_vax_x_leeftijdsgr = aantal_vax[i]/ aantal_inw[i]
        p_non_vax_x_leeftijdsgr = (aantal_inw[i] - aantal_vax[i])/ aantal_inw[i]
        
        aantal_non_vax_event_leeftijdsgr = event[i][0]+event[i][1]
        aantal_vax_event_leeftijdsgr = event[i][2]
        
        totaal_leeftijdsgr_event = event[i][0]+event[i][1]+ +event[i][2]
        p_leeftijdsgr_x_event = totaal_leeftijdsgr_event / totaal_aantal_event
        st.write(f"P ({leeftijdsgr_txt}|{event_txt}) = {round(p_leeftijdsgr_x_event,3)}")
        
        if follow_txt == True:
            # just to follow the calculation in the original story
            p_vax_x_event_leeftijdsgr = aantal_vax_event_leeftijdsgr / totaal_aantal_event
            p_non_vax_x_event_leeftijdsgr = aantal_non_vax_event_leeftijdsgr / totaal_aantal_event
        
        else:
            # this is the right calculation
            p_vax_x_event_leeftijdsgr = aantal_vax_event_leeftijdsgr / totaal_leeftijdsgr_event
            p_non_vax_x_event_leeftijdsgr = aantal_non_vax_event_leeftijdsgr / totaal_leeftijdsgr_event
            
        ratio_of_probabilities_non_vax_vs_vax = p_non_vax_x_event_leeftijdsgr / p_vax_x_event_leeftijdsgr

        st.write(f"Pr(VAX|{event_txt},{leeftijdsgr_txt}) = {p_vax_x_event_leeftijdsgr}")
        st.write(f"Pr(nVAX|{event_txt},{leeftijdsgr_txt}) = {p_non_vax_x_event_leeftijdsgr}")
        st.write(f"Marginal ratio {ratio_of_probabilities_non_vax_vs_vax}")
        st.write(" ------ ")
        p_event_x_leeftijdsgr = p_leeftijdsgr_x_event * p_event / p_leeftijdsgr
        st.write(f"Pr({event_txt} | {leeftijdsgr_txt}) = {p_event_x_leeftijdsgr}")
        
        p_event_x_vax_leeftijdsgr = p_vax_x_event_leeftijdsgr * p_event_x_leeftijdsgr        / p_vax_x_leeftijdsgr
        st.write(" ------ ")
        st.write(f"Pr({event_txt} | VAX, {leeftijdsgr_txt}) = Pr(VAX | {event_txt}, {leeftijdsgr_txt}) * Pr({event_txt}|{leeftijdsgr_txt})  / Pr(VAX | {leeftijdsgr_txt})")
        
        st.write(f"{p_event_x_vax_leeftijdsgr} = {p_vax_x_event_leeftijdsgr} * {p_event_x_leeftijdsgr}        / {p_vax_x_leeftijdsgr}")
        
        st.write(" ------ ")
        p_event_x_non_vax_leeftijdsgr = p_non_vax_x_event_leeftijdsgr * p_event_x_leeftijdsgr         / p_non_vax_x_leeftijdsgr
        st.write(f"Pr({event_txt} | nVAX, {leeftijdsgr_txt}) = Pr(nVAX | {event_txt}, {leeftijdsgr_txt}) * Pr({event_txt}|{leeftijdsgr_txt})  / Pr(nVAX | {leeftijdsgr_txt})")
        st.write(f"{p_event_x_non_vax_leeftijdsgr} = {p_non_vax_x_event_leeftijdsgr} * {p_event_x_leeftijdsgr}         / {p_non_vax_x_leeftijdsgr}")
        
        st.write(f"Pr({event_txt} | nVAX,{leeftijdsgr_txt}) = {p_event_x_non_vax_leeftijdsgr}")
        



        ratio_event_x__leeftijdsgr = p_event_x_non_vax_leeftijdsgr / p_event_x_vax_leeftijdsgr


        st.write (f"Marginal ratio = {ratio_event_x__leeftijdsgr}")
st.header("Reproduction of chances")
st.write("Reproduction of https://medium.com/mlearning-ai/pr-non-vax-icu-pr-icu-non-vax-f7af477896ad")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Met de goede methode:")
    calculate(ziekenhuis, "HOSP", False)
    calculate(icopnames, "IC", False)

with col2:
    st.subheader("Met de oude methode:")
    calculate(ziekenhuis, "HOSP", True)
    calculate(icopnames, "IC", True)
