import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.title("Vaccin efficacity vs inverse odds")
    # st.write("A change of VE from 97% to 94 seems to be small. However with a VE of 97% an unvaccinated persion has 33x more the chance to get sick than a vaccinated person.
    # A ")

    title = "VE as index"
    # st.write(columnlist)
    x_,y_ = [],[]
    for i in range (10,1000):

        y = (1-(1/(i/10)))*100
        #y = -1/(i-1)
        x_.append(i/10)
        y_.append(y)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_, y= x_, mode='lines' ))
    fig.update_layout(
        xaxis_title="VE (%)",
        yaxis_title="aantal keer kans ongevacc. vs gevacc. / inv odds / factor"    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_, y= y_, mode='lines' ))
    fig2.update_layout(
        yaxis_title="VE (%)",
        xaxis_title="aantal keer kans ongevacc. vs gevacc. / inv odds/ factor"    )
    st.plotly_chart(fig2, use_container_width=True)


main()


