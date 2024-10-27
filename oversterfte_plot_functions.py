import pandas as pd
import cbsodata
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import get_rioolwater

# from streamlit import caching
import scipy.stats as stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels.api as sm

def plot_wrapper(
    df_boosters,
    df_herhaalprik,
    df_herfstprik,
    df_rioolwater,
    df_,
    df_corona,
    df_quantile,
    df_kobak,
    series_name,
    how,
    yaxis_to_zero,
    rightax,
    mergetype,
    sec_y,
):
    """wrapper for the plots

    Args:
        df_ : df_sterfte
        series_names (_type_): _description_
        how (_type_): _description_
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """

    def plot_graph_oversterfte(
        how,
        df,
        df_corona,
        df_boosters,
        df_herhaalprik,
        df_herfstprik,
        df_rioolwater,
        df_kobak,
        series_name,
        rightax,
        mergetype,
        sec_y,
    ):

        """_summary_

        Args:
            how (_type_): _description_
            df (_type_): _description_
            df_corona (_type_): _description_
            df_boosters (_type_): _description_
            df_herhaalprik (_type_): _description_
            series_name (_type_): _description_
            rightax (_type_): _description_
            mergetype (_type_): _description_
        """
        booster_cat = ["m_v_0_999", "m_v_0_64", "m_v_65_79", "m_v_80_999"]
      
        df_oversterfte = pd.merge(
            df, df_corona, left_on="periodenr", right_on="periodenr", how="outer"
        )

        if rightax == "boosters":
            df_oversterfte = pd.merge(
                df_oversterfte, df_boosters, on="periodenr", how=mergetype
            )
        if rightax == "herhaalprik":
            df_oversterfte = pd.merge(
                df_oversterfte, df_herhaalprik, on="periodenr", how=mergetype
            )
        if rightax == "herfstprik":
            df_oversterfte = pd.merge(
                df_oversterfte, df_herfstprik, on="periodenr", how=mergetype
            )
        if rightax == "rioolwater":
            df_oversterfte = pd.merge(
                df_oversterfte, df_rioolwater, on="periodenr", how=mergetype
            )
        if rightax == "kobak":
            df_oversterfte = pd.merge(
                df_oversterfte, df_kobak, on="periodenr", how=mergetype
            )

        df_oversterfte["over_onder_sterfte"] = 0
        df_oversterfte["meer_minder_sterfte"] = 0

        df_oversterfte["year_minus_high95"] = (
            df_oversterfte[series_name] - df_oversterfte["high95"]
        )
        df_oversterfte["year_minus_avg"] = (
            df_oversterfte[series_name] - df_oversterfte["avg"]
        )
        df_oversterfte["p_score"] = (
            df_oversterfte[series_name] - df_oversterfte["avg"]
        ) / df_oversterfte["avg"]
        df_oversterfte = rolling(df_oversterfte, "p_score")

        for i in range(len(df_oversterfte)):
            if df_oversterfte.loc[i, series_name] > df_oversterfte.loc[i, "high95"]:
                df_oversterfte.loc[i, "over_onder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "avg"]
                )  # ["high95"]
                df_oversterfte.loc[i, "meer_minder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "high95"]
                )
            elif df_oversterfte.loc[i, series_name] < df_oversterfte.loc[i, "low05"]:
                df_oversterfte.loc[i, "over_onder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "avg"]
                )  # ["low05"]
                df_oversterfte.loc[i, "meer_minder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "low05"]
                )
        
        
        
           
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df_oversterfte["periodenr"],
                y=df_oversterfte[how],
                # line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                line=dict(width=2, color="rgba(205, 61,62, 1)"),
                mode="lines",
                name=how,
            )
        )

        if how == "p_score":
            # the p-score is already plotted
            pass
        elif how == "year_minus_avg":
            show_avg = False
            if show_avg:
                grens = "avg"
                fig.add_trace(
                    go.Scatter(
                        name=grens,
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[grens],
                        mode="lines",
                        line=dict(width=1, color="rgba(205, 61,62, 1)"),
                    )
                )
        else:
            grens = "95%_interval"

            fig.add_trace(
                go.Scatter(
                    name="low",
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte["low05"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
                    fillcolor="rgba(68, 68, 68, 0.2)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="high",
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte["high95"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
                    fill="tonexty",
                )
            )

            # data = [high, low, fig_, sterfte ]
            fig.add_trace(
                go.Scatter(
                    name="Verwachte Sterfte",
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte["avg"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(204, 63, 61, .8)"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Sterfte",
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte[series_name],
                    mode="lines",
                    line=dict(width=1, color="rgba(204, 63, 61, 1)"),
                )
            )
        # rightax = "boosters" # "herhaalprik"
        if series_name in booster_cat or rightax == "rioolwater":
            if rightax == "boosters":

                b = "boosters_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="boosters",
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )
                corr = df_oversterfte[b].corr(df_oversterfte[how])
                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "herhaalprik":

                b = "herhaalprik_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="herhaalprik",
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "herfstprik":

                b = "herfstprik_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="herfstprik",
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "rioolwater":
                b = "rioolwater_sma"
                fig.add_trace(
                    go.Scatter(
                        name="rioolwater",
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "kobak":

                b = "excess deaths"
                fig.add_trace(
                    go.Scatter(
                        name="excess deaths(kobak)",
                        x=df_oversterfte["periodenr"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")

        # data.append(booster)

        title = how
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig.add_hline(y=0)

        fig.update_yaxes(rangemode="tozero")

        st.plotly_chart(fig, use_container_width=True)
        # /plot_graph_oversterfte

    def plot_lines(series_name, df_data):
        # fig = plt.figure()

        year_list = df_data["jaar"].unique().tolist()

        data = []

        for idx, year in enumerate(year_list):
            df = df_data[df_data["jaar"] == year].copy(
                deep=True
            )  
            if (
                year == 2020
                or year == 2021
                or year == 2022
                or year == 2023
                or year == 2024
            ):
                width = 3
                opacity = 1
            else:
                width = 0.7
                opacity = 0.3

            fig_ = go.Scatter(
                x=df["week"],
                y=df[series_name],
                line=dict(width=width),
                opacity=opacity,  # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                mode="lines",
                name=year,
                legendgroup=str(year),
            )

            data.append(fig_)

        title = f"Stefte - {series_name}"
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        # end of plot_lines

    def plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile):
    
        df_corona = df_corona[df_corona["periodenr"] !="2019_1"] #somewhere an error bc of 53 weeks in 53
       
        columnlist = ['avg_', 'low05', 'high95']
        for what_to_sma in columnlist:
            df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=7, center=True).mean().round(1)

        # df_quantile = df_quantile.sort_values(by=['jaar','week_'])
        
        fig = go.Figure()
        low05 = go.Scatter(
            name="low",
            x=df_quantile["periodenr"],
            y=df_quantile["low05"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.1)",
            fill="tonexty",
        )

        avg = go.Scatter(
            name="gemiddeld",
            x=df_quantile["periodenr"],
            y=df_quantile["avg_"],
            mode="lines",
            line=dict(width=0.75, color="rgba(68, 68, 68, 0.8)"),
        )

        sterfte = go.Scatter(
            name="Sterfte",
            x=df_corona["periodenr"],
            y=df_corona[series_name],
            mode="lines",
            line=dict(width=2, color="rgba(255, 0, 0, 0.8)"),
        )

        high95 = go.Scatter(
            name="high",
            x=df_quantile["periodenr"],
            y=df_quantile["high95"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.2)",
        )

        # data = [ q95, high95, q05,low05,avg, sterfte] #, value_in_year_2021 ]
        data = [high95, low05, avg, sterfte]  # , value_in_year_2021 ]
        title = f"Overleden x {series_name}"
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig = go.Figure(data=data, layout=layout)
        fig = layout_annotations_fig(fig)
    

        if yaxis_to_zero:
            fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        return df_quantile
        # end of plot quantiles

    st.subheader(series_name)
    if how == "quantiles":

        plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile)

    elif (
        (how == "year_minus_avg")
        or (how == "over_onder_sterfte")
        or (how == "meer_minder_sterfte")
        or (how == "p_score")
    ):
        plot_graph_oversterfte(
            how,
            df_quantile,
            df_corona,
            df_boosters,
            df_herhaalprik,
            df_herfstprik,
            df_rioolwater,
            df_kobak,
            series_name,
            rightax,
            mergetype,
            sec_y,
        )
    else:
        plot_lines(series_name, df_corona)



def layout_annotations_fig(fig):
    fig.update_layout(xaxis=dict(tickformat="%d-%m"))

    #             — eerste oversterftegolf: week 13 tot en met 18 van 2020 (eind maart–eind april 2020);
    # — tweede oversterftegolf: week 39 van 2020 tot en met week 3 van 2021 (eind
    # september 2020–januari 2021);
    # — derde oversterftegolf: week 33 tot en met week 52 van 2021 (half augustus 2021–eind
    # december 2021).
    # De hittegolf in 2020 betreft week 33 en week 34 (half augustus 2020).

    fig.add_vrect(
        x0="2020_13",
        x1="2020_18",
        annotation_text="Eerste golf",
        annotation_position="top left",
        fillcolor="pink",
        opacity=0.25,
        line_width=0,
    )
    fig.add_vrect(
        x0="2020_39",
        x1="2021_03",
        annotation_text="Tweede golf",
        annotation_position="top left",
        fillcolor="pink",
        opacity=0.25,
        line_width=0,
    )
    fig.add_vrect(
        x0="2021_33",
        x1="2021_52",
        annotation_text="Derde golf",
        annotation_position="top left",
        fillcolor="pink",
        opacity=0.25,
        line_width=0,
    )

    # hittegolven
    fig.add_vrect(
        x0="2020_33",
        x1="2020_34",
        annotation_text=" ",
        annotation_position="top left",
        fillcolor="yellow",
        opacity=0.35,
        line_width=0,
    )

    fig.add_vrect(
        x0="2022_32",
        x1="2022_33",
        annotation_text=" ",
        annotation_position="top left",
        fillcolor="yellow",
        opacity=0.35,
        line_width=0,
    )

    fig.add_vrect(
        x0="2023_23",
        x1="2023_24",
        annotation_text=" ",
        annotation_position="top left",
        fillcolor="yellow",
        opacity=0.35,
        line_width=0,
    )
    fig.add_vrect(
        x0="2023_36",
        x1="2023_37",
        annotation_text="Geel = Hitte golf",
        annotation_position="top left",
        fillcolor="yellow",
        opacity=0.35,
        line_width=0,
    )
    return fig

  
def plot_graph_rivm(df, series_naam, rivm):

    # Maak een interactieve plot met Plotly
    fig = go.Figure()

    # Voeg de werkelijke data toe
    fig.add_trace(
        go.Scatter(
            x=df["periodenr"],
            y=df[f"{series_naam}_y"],
            mode="lines",
            name="Werkelijke data cbs",
        )
    )

    # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df["periodenr"], y=df["voorspeld"], mode="lines", name="Voorspeld model"
        )
    )
    if rivm == True:
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["periodenr"],
                y=df["verw_waarde_rivm"],
                mode="lines",
                name="Voorspeld RIVM",
            )
        )
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["periodenr"],
                y=df["ondergrens_verwachting_rivm"],
                mode="lines",
                name="onder RIVM",
            )
        )  # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["periodenr"],
                y=df["bovengrens_verwachting_rivm"],
                mode="lines",
                name="boven RIVM",
            )
        )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df["periodenr"],
            y=df["upper_ci"],
            mode="lines",
            fill=None,
            line_color="lightgrey",
            name="Bovenste CI",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["periodenr"],
            y=df["lower_ci"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="lightgrey",
            name="Onderste CI",
        )
    )

    # Titel en labels toevoegen
    fig.update_layout(
        title="Voorspelling van Overledenen met 95% Betrouwbaarheidsinterval RIVM",
        xaxis_title="Tijd",
        yaxis_title="Aantal Overledenen",
    )

    st.plotly_chart(fig)

def show_difference_plot(df, date_field, show_official, year):
    # Maak een interactieve plot met Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["baseline_kobak"],
            mode="lines",
            name="Baseline Kobak",
        )
    )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["high_rivm"],
            mode="lines",
            fill=None,
            line_color="yellow",
            name="high rivm",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["low_rivm"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="yellow",
            name="low rivm",
        )
    )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["low_cbs_sma"],
            mode="lines",
            fill=None,
            line_color="lightgrey",
            name="low cbs",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["high_cbs_sma"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="lightgrey",
            name="high cbs",
        )
    )
    if show_official:
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["high_rivm_official"],
                mode="lines",
                fill=None,
                line_color="orange",
                name="high rivm official",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["low_rivm_official"],
                mode="lines",
                fill="tonexty",  # Vul het gebied tussen de lijnen
                line_color="orange",
                name="low rivm  official",
            )
        )

        # Voeg de betrouwbaarheidsinterval toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["low_cbs_official"],
                mode="lines",
                fill=None,
                line_color="lightblue",
                name="low cbs  official",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["high_cbs_official"],
                mode="lines",
                fill="tonexty",  # Vul het gebied tussen de lijnen
                line_color="lightblue",
                name="high cbs  official",
            )
        )
        # Voeg de voorspelde lijn toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["verw_rivm_official"],
                mode="lines",
                name="Baseline model rivm  official",
            )
        )
        # Voeg de voorspelde lijn toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["verw_cbs_official"],
                mode="lines",
                name="Baseline model cbs  official",
                line=dict(width=0.75)#, color="rgba(0, 255, 68, 0.8)")
            )
        )

        # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["verw_rivm"],
            mode="lines",
            name="Baseline model rivm",
        )
    )

    # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["verw_cbs_sma"],
            mode="lines",
            name="Baseline model cbs",
            line=dict(width=0.75, color="rgba(68, 68, 68, 0.8)"),
        )
    )

    # Voeg de voorspelde lijn RIVM toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["aantal_overlijdens"],
            mode="lines",
            name="Werkelijk overleden",
            line=dict(width=1.5, color="rgba(255, 0, 0, 1.0)"),
        )
    )
    # Titel en labels toevoegen
    fig.update_layout(
        title="Vergelijking CBS vs RIVM",
        xaxis_title="Tijd",
        yaxis_title="Aantal Overledenen",
    )

    st.plotly_chart(fig)

def show_plot_steigstra(df_spaghetti,series_name):
    
    # Plotting with Plotly
    fig = go.Figure()
    fig.add_vline(x=25, name="week 1", line=dict(color="gray", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="black", width=2))

    for year in df_spaghetti.columns[:-4]:
        fig.add_trace(
            go.Scatter(
                x=df_spaghetti.index,
                y=df_spaghetti[year],
                mode="lines",
                name=f"{year} - {year+1}",
            )
        )

    fig.add_trace(
        go.Scatter(
            name="low",
            x=df_spaghetti.index,
            y=df_spaghetti["low"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="high",
            x=df_spaghetti.index,
            y=df_spaghetti["high"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.1)"),
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.2)",
        )
    )

    # Update layout to customize x-axis labels
    fig.update_layout(
        title=f"Steigstra Plot of {series_name} over Different Years week 32 to week 31",
        xaxis_title="Weeks (32 to 31)",
        yaxis_title="Values",
        xaxis=dict(
            tickvals=df_spaghetti.index,  # Set the tick positions to the DataFrame index
            ticktext=df_spaghetti['week_real'].astype(str)  # Set the tick labels to 'week_real'
        )
    )

    st.plotly_chart(fig)