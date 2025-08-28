# app.py
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.api import OLS, add_constant
    

from sklearn.linear_model import LinearRegression
from datetime import datetime

try:
    st.set_page_config(page_title="Verwachte sterfte • RIVM-stijl", layout="wide")
except:
    pass


def main_chatgpt():
   
    # ---------- Helpers ----------
    def load_data(uploaded_file=None, fallback_path=None):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=";", dtype={"jaar": int, "week": int, "overleden": float})
        elif fallback_path and os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path, sep=";", dtype={"jaar": int, "week": int, "overleden": float})
        else:
            st.stop()

        cols = {c.lower(): c for c in df.columns}
        for must in ["jaar", "week", "overleden"]:
            assert must in cols, f"Kolom '{must}' niet gevonden"
        df = df.rename(columns={cols["jaar"]: "jaar", cols["week"]: "week", cols["overleden"]: "overleden"})

        df["date"] = pd.to_datetime(
            df["jaar"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
            errors="coerce"
        )
        df["season_year"] = np.where(df["week"] >= 27, df["jaar"] + 1, df["jaar"])
        df["month"] = df["date"].dt.month
        df = df.sort_values("date").reset_index(drop=True)
        df["t"] = np.arange(len(df))

        week_frac = (df["week"] - 1) / 52.0
        df["sin1"] = np.sin(2 * np.pi * week_frac)
        df["cos1"] = np.cos(2 * np.pi * week_frac)
        df["sin2"] = np.sin(4 * np.pi * week_frac)
        df["cos2"] = np.cos(4 * np.pi * week_frac)

        # Positie 1..52 in seizoensjaar: 27..52 → 1..26, 1..26 → 27..52
        df["season_week_idx"] = np.where(df["week"] >= 27, df["week"] - 26, df["week"] + 26)

        # Label voor as: W27..W52, W01..W26
        def weeklabel(w):
            return f"W{w:02d}" if w <= 26 else f"W{w}"
        df["season_week_label"] = df["week"].apply(lambda w: f"W{w:02d}")

        return df

    def train_mask_exclusions(train):
        q75_all = train["overleden"].quantile(0.75)
        mask_all = train["overleden"] <= q75_all
        is_jul_aug = train["month"].isin([7, 8])
        if is_jul_aug.any():
            q80_summer = train.loc[is_jul_aug, "overleden"].quantile(0.80)
            mask_summer = (~is_jul_aug) | (train["overleden"] <= q80_summer)
        else:
            mask_summer = pd.Series(True, index=train.index)
        return mask_all & mask_summer

    def fit_baseline(df, season_year_target: int, harmonics: int = 1):
        prev_years = list(range(season_year_target - 5, season_year_target))
        train = df[df["season_year"].isin(prev_years)].copy()
        if train.empty or train["overleden"].isna().all():
            raise ValueError("Onvoldoende trainingsdata")

        mask = train_mask_exclusions(train)
        train_cln = train.loc[mask].copy()

        X_cols = ["t", "sin1", "cos1"] if harmonics == 1 else ["t", "sin1", "cos1", "sin2", "cos2"]
        X_train = add_constant(train_cln[X_cols])
        y_train = train_cln["overleden"].values
        model = OLS(y_train, X_train).fit()

        # --- Belangrijk: maak alle weken van dit seizoensjaar expliciet ---
        season_weeks = list(range(27, 53)) + list(range(1, 27))
        # Kalenderjaar = season_year -1 voor weken 27-52, anders season_year
        rows = []
        for w in season_weeks:
            jaar = season_year_target - 1 if w >= 27 else season_year_target
            date = pd.to_datetime(f"{jaar}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
            rows.append({"jaar": jaar, "week": w, "season_year": season_year_target, "date": date})
        target = pd.DataFrame(rows)

        # t-index en harmonische termen opnieuw koppelen
        # Zoek dichtstbijzijnde index t uit df
        df_map = df.set_index(["jaar", "week"])
        target = target.join(df_map[["t"]], on=["jaar", "week"])
        # Als laatste jaar nog geen t heeft → zelf doortrekken
        if target["t"].isna().any():
            max_t = df["t"].max()
            miss_mask = target["t"].isna()
            target.loc[miss_mask, "t"] = np.arange(max_t + 1, max_t + 1 + miss_mask.sum())
        target["t"] = target["t"].astype(int)

        week_frac = (target["week"] - 1) / 52.0
        target["sin1"] = np.sin(2 * np.pi * week_frac)
        target["cos1"] = np.cos(2 * np.pi * week_frac)
        target["sin2"] = np.sin(4 * np.pi * week_frac)
        target["cos2"] = np.cos(4 * np.pi * week_frac)

        # Baseline voorspellen
        X_target = add_constant(target[X_cols])
        target["baseline"] = model.predict(X_target)

        resid_sd = np.sqrt(model.scale)
        target["lower"] = target["baseline"] - 2 * resid_sd
        target["upper"] = target["baseline"] + 2 * resid_sd

        # Werkelijke sterfte toevoegen (als bekend)
        actual_map = df.set_index(["jaar", "week"])["overleden"]
        target["overleden"] = actual_map.reindex(list(zip(target["jaar"], target["week"]))).values

        return {
            "target_df": target,
            "train_used": train_cln,
            "model_summary": model.summary().as_text(),
            "resid_sd": resid_sd,
        }


    def make_plot_single(target_df, season_year_target):
        def sort_key(row):
            w = int(row["week"])
            return (0 if w >= 27 else 1, w)
        target_df = target_df.copy()
        target_df["__order"] = target_df.apply(sort_key, axis=1)
        target_df = target_df.sort_values(["__order", "week"]).reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=target_df["date"], y=target_df["upper"], name="Base + 2×sd",
                                mode="lines", line=dict(width=0), hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=target_df["date"], y=target_df["lower"], name="Base − 2×sd",
                                mode="lines", line=dict(width=0), fill="tonexty",
                                hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=target_df["date"], y=target_df["baseline"], name="Baselijn",
                                mode="lines", hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=target_df["date"], y=target_df["overleden"], name="Werkelijk",
                                mode="lines+markers", hovertemplate="%{y:.0f}"))

        fig.update_layout(
            title=f"Verwachte sterfte • Seizoensjaar {season_year_target} (week 27 t/m 26)",
            xaxis_title="Week",
            yaxis_title="Overledenen",
            legend_title="Reeks",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    def make_plot_all_seasons(df, seasons, include_baselines=False, harmonics=1):
        fig = go.Figure()
        for sy in seasons:
            # Werkelijk
            sel = df[df["season_year"] == sy].copy()
            if sel.empty:
                continue
            sel = sel.sort_values("season_week_idx")
            fig.add_trace(go.Scatter(
                x=sel["season_week_idx"],
                y=sel["overleden"],
                name=f"{sy} werkelijk",
                mode="lines",
                hovertemplate=f"Seizoen {sy}"# • %{x} • %{y:.0f}"
            ))
            # Baselijn optioneel
            if include_baselines:
                try:
                    res = fit_baseline(df, sy, harmonics=harmonics)
                    tgt = res["target_df"].sort_values("season_week_idx")
                    fig.add_trace(go.Scatter(
                        x=tgt["season_week_idx"],
                        y=tgt["baseline"],
                        name=f"{sy} baselijn",
                        mode="lines",
                        line=dict(dash="dot"),
                        hovertemplate=f"Baselijn {sy} • %{x} • %{y:.0f}"
                    ))
                except Exception:
                    pass

        # Mooie X-as labels: W27..W52, W01..W26
        idx = list(range(1, 53))
        labels = [f"W{w}" if (w>=27 and w<=52) else f"W{w:02d}" for w in (list(range(27,53)) + list(range(1,27)))]
        fig.update_layout(
            title="Alle seizoenen 2021–2026 • Werkelijk" + (" + baselines" if include_baselines else ""),
            xaxis=dict(title="Week in seizoensjaar", tickmode="array", tickvals=idx, ticktext=labels),
            yaxis_title="Overledenen",
            legend_title="Reeks",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    #--- NIEUW: alles-in-1 tijdlijn ---
    def make_plot_timeline_all(df, seasons, show_bands=True, harmonics=1):
        
        fig = go.Figure()

        # Doelvenster: half 2019 t/m half 2026
        start_date = pd.Timestamp("2019-07-01")
        end_date = pd.Timestamp("2026-06-30")

        for sy in seasons:
            try:
                res = fit_baseline(df, sy, harmonics=harmonics)
            except Exception:
                continue

            tdf = res["target_df"].copy()
            tdf = tdf[(tdf["date"] >= start_date) & (tdf["date"] <= end_date)]
            if tdf.empty:
                continue

            # Banden per seizoen
            if show_bands:
                fig.add_trace(go.Scatter(
                    x=tdf["date"], y=tdf["upper"],
                    name=f"{sy} band boven",
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=f"{sy}",
                    showlegend=False,
                    hovertemplate="%{y:.0f}"
                ))
                fig.add_trace(go.Scatter(
                    x=tdf["date"], y=tdf["lower"],
                    name=f"{sy} band onder",
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    legendgroup=f"{sy}",
                    fillcolor=None,
                    showlegend=False,
                    hovertemplate="%{y:.0f}"
                ))

            # Baselijn
            fig.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["baseline"],
                name=f"{sy} baselijn",
                mode="lines",
                legendgroup=f"{sy}",
                showlegend=False,
                hovertemplate="%{y:.0f}"
            ))

            # Werkelijk
            fig.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["overleden"],
                name=f"{sy} werkelijk",
                mode="lines",
                legendgroup=f"{sy}",
                showlegend=False,
                hovertemplate="%{y:.0f}"
            ))

        fig.update_layout(
            title="Alle seizoenen op één tijdlijn • 2019-H2 t/m 2026-H1",
            xaxis_title="Datum",
            yaxis_title="Overledenen",
            hovermode="x unified",
            legend_title="Reeks",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        fig.update_xaxes(range=[start_date, end_date])
        return fig
    # ---------- UI ----------
    st.title("Verwachte sterfte • RIVM-stijl baselijn")
    st.caption("5 voorgaande seizoensjaren • trend + sinus/cosinus • pieken uitgesloten")

    st.info("We reproduceren de methode van het RIVM naar aanleiding van https://x.com/infopinie/status/1960744770810073247")
    default_path = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week_2014_2025.csv"
    default_path ="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week_2014_2025.csv"
    uploaded = st.file_uploader("Upload CSV (;-gescheiden) met kolommen: ID;jaar;week;overleden", type=["csv"])

    with st.expander("Opties"):
        use_harm2 = st.checkbox("Gebruik 2 harmonischen", value=True)
        show_model = st.checkbox("Toon modeldetails", value=False)
        show_train = st.checkbox("Toon trainingspunten (na uitsluiten)", value=False)
        include_baselines_all = st.checkbox("Toon baselines in 'Alle seizoenen'", value=False)

    df = load_data(uploaded_file=uploaded, fallback_path=default_path)

    min_season_available = int(df["season_year"].min() + 5)
    max_season_available = int(df["season_year"].max())

   
    tab1, tab2, tab3, tab4 = st.tabs(["Tijdlijn 2019H2–2026H1","Per seizoen", "Alle seizoenen 2021–2026", "GROK"])
    with tab1:
        st.subheader("Alle tab1-reeksen in één plot")
        show_bands_timeline = st.checkbox("Toon ±2×sd banden", value=True)
        # Half 2019 hoort bij seizoensjaar 2020
        seasons_timeline = [sy for sy in range(2020, 2027) if min_season_available <= sy <= max_season_available]
        if not seasons_timeline:
            st.info("Onvoldoende data om 2020 t/m 2026 te tonen.")
        else:
            fig3 = make_plot_timeline_all(
                df,
                seasons=seasons_timeline,
                show_bands=show_bands_timeline,
                harmonics=2 if use_harm2 else 1
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.info("https://chatgpt.com/share/68b0d9ad-fdb0-8004-b943-886a964a8baa")
    with tab2:
        season_choice = st.slider(
            "Kies seizoensjaar",
            min_value=min_season_available,
            max_value=max_season_available,
            value=max_season_available,
            step=1
        )
        try:
            result = fit_baseline(df, season_choice, harmonics=2 if use_harm2 else 1)
            target_df = result["target_df"]
            fig1 = make_plot_single(target_df, season_choice)
            st.plotly_chart(fig1, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Residuele sd", f"{result['resid_sd']:.1f}")
            col2.metric("Aantal trainingsweken", f"{len(result['train_used'])}")
            col3.metric("Aantal doelweken", f"{len(target_df)}")

            if show_train:
                st.dataframe(result["train_used"][["date", "jaar", "week", "overleden"]].reset_index(drop=True))

            csv_buf = io.StringIO()
            target_df[["date", "jaar", "week", "overleden", "baseline", "lower", "upper"]].to_csv(csv_buf, index=False)
            st.download_button(
                "Download resultaten CSV",
                data=csv_buf.getvalue(),
                file_name=f"baseline_{season_choice}.csv",
                mime="text/csv"
            )

            if show_model:
                st.text(result["model_summary"])
        except Exception as e:
            st.error(str(e))

    with tab3:
        # Half 2020 → start bij seizoen 2021 (week 27 2020 t/m week 26 2021)
        # Half 2026 → eindigt bij seizoen 2026
        seasons_range = [sy for sy in range(2021, 2027) if min_season_available <= sy <= max_season_available]
        if not seasons_range:
            st.info("Onvoldoende data voor 2021–2026.")
        else:
            fig2 = make_plot_all_seasons(
                df,
                seasons=seasons_range,
                include_baselines=include_baselines_all,
                harmonics=2 if use_harm2 else 1
            )
            st.plotly_chart(fig2, use_container_width=True)
    with tab4:
        main_grok()
    
    st.info("RIVM Grafiek: https://www.rivm.nl/monitoring-sterftecijfers-nederland")
    
def main_grok():


    # Path to the CSV file
    # csv_path = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week_2014_2025.csv"
    csv_path ="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week_2014_2025.csv"
    
    df = pd.read_csv(csv_path, sep=';')
    df = df[['jaar', 'week', 'overleden']]
    df.sort_values(['jaar', 'week'], inplace=True)
    df['time'] = np.arange(len(df))
    df['week_rad'] = 2 * np.pi * df['week'] / 52
    df['sin_week'] = np.sin(df['week_rad'])
    df['cos_week'] = np.cos(df['week_rad'])
    df['date'] = pd.to_datetime(df['jaar'].astype(str) + '-' + df['week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')

    def calculate_expected_mortality(pred_year):
        past_years = range(pred_year - 5, pred_year)
        past_data = df[df['jaar'].isin(past_years)].copy()
        if len(past_data) == 0:
            return None, None, None, None, None, None
        summer_weeks = range(27, 36)
        q_all = past_data['overleden'].quantile(0.75)
        summer_data = past_data[past_data['week'].isin(summer_weeks)]
        q_summer = summer_data['overleden'].quantile(0.80) if not summer_data.empty else np.inf
        is_peak_all = past_data['overleden'] > q_all
        is_summer = past_data['week'].isin(summer_weeks)
        is_peak_summer = past_data['overleden'] > q_summer
        to_exclude = is_peak_all | (is_summer & is_peak_summer)
        filtered_data = past_data[~to_exclude]
        if len(filtered_data) < 10:
            return None, None, None, None, None, None
        features = ['time', 'sin_week', 'cos_week']
        X = filtered_data[features]
        y = filtered_data['overleden']
        model = LinearRegression().fit(X, y)
        pred_train = model.predict(X)
        residuals = y - pred_train
        rmse = np.sqrt(np.mean(residuals ** 2))
        future_weeks = list(range(27, 53)) + list(range(1, 27))
        future_years = [pred_year] * 26 + [pred_year + 1] * 26
        future_df = pd.DataFrame({'jaar': future_years, 'week': future_weeks})
        future_df['time'] = past_data['time'].max() + 1 + np.arange(len(future_df))
        future_df['week_rad'] = 2 * np.pi * future_df['week'] / 52
        future_df['sin_week'] = np.sin(future_df['week_rad'])
        future_df['cos_week'] = np.cos(future_df['week_rad'])
        future_df['date'] = pd.to_datetime(future_df['jaar'].astype(str) + '-' + future_df['week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
        X_future = future_df[features]
        predicted = model.predict(X_future)
        upper = predicted + 2 * rmse
        lower = np.maximum(predicted - 2 * rmse, 0)
        observed_mask = ((df['jaar'] == pred_year) & (df['week'] >= 27)) | ((df['jaar'] == pred_year + 1) & (df['week'] <= 26))
        observed_df = df[observed_mask].copy()
        observed = observed_df['overleden'].values if not observed_df.empty else None
        observed_dates = observed_df['date'].values if not observed_df.empty else None
        return future_df['date'].values, predicted, lower, upper, observed, observed_dates

    st.title('Verwachte Sterfte (Week 27, 2019 - Week 26, 2026)')
    fig = go.Figure()
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    available_years = range(2019, 2026)
    last_year = max(available_years)
    for idx, pred_year in enumerate(available_years):
        if pred_year < min(df['jaar']) + 5:
            continue
        dates, baseline, lower, upper, observed, observed_dates = calculate_expected_mortality(pred_year)
        if baseline is None:
            continue
        start_date = pd.to_datetime('2019-07-01')
        end_date = pd.to_datetime('2026-06-30')
        mask = (dates >= start_date) & (dates <= end_date)
        dates = dates[mask]
        baseline = baseline[mask]
        lower = lower[mask]
        upper = upper[mask]
        if observed is not None:
            observed_mask = (observed_dates >= start_date) & (observed_dates <= end_date)
            observed_dates = observed_dates[observed_mask]
            observed = observed[observed_mask]
        color = colors[idx % len(colors)]
        if pred_year == last_year:
            # Show full baseline and bands for the last year
            fig.add_trace(go.Scatter(x=dates, y=upper, fill=None, mode='lines', line=dict(color='rgba(0,100,80,0.2)')))
            fig.add_trace(go.Scatter(x=dates, y=lower, fill='tonexty', mode='lines', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(0,100,80,0.2)')))
            fig.add_trace(go.Scatter(x=dates, y=baseline, mode='lines', line=dict(color=color)))
            if observed is not None:
                fig.add_trace(go.Scatter(x=observed_dates, y=observed, mode='lines', line=dict(color=color, dash='dash')))
        else:
            # For other years, limit to observed dates if available
            if observed is not None and len(observed_dates) > 0:
                # Align baseline, upper, lower to observed length (assuming observed_dates subset of dates)
                aligned_indices = [np.where(dates == od)[0][0] for od in observed_dates if od in dates]
                aligned_baseline = baseline[aligned_indices]
                aligned_lower = lower[aligned_indices]
                aligned_upper = upper[aligned_indices]
                fig.add_trace(go.Scatter(x=observed_dates, y=aligned_upper, fill=None, mode='lines', line=dict(color='rgba(0,100,80,0.2)')))
                fig.add_trace(go.Scatter(x=observed_dates, y=aligned_lower, fill='tonexty', mode='lines', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(0,100,80,0.2)')))
                fig.add_trace(go.Scatter(x=observed_dates, y=aligned_baseline, mode='lines', line=dict(color=color)))
                fig.add_trace(go.Scatter(x=observed_dates, y=observed, mode='lines', line=dict(color=color, dash='dash')))
    fig.update_layout(
        title='Verwachte Sterfte (Week 27, 2019 - Week 26, 2026)',
        xaxis_title='Datum',
        yaxis_title='Aantal Overlijdens',
        hovermode='x unified',
        showlegend=False,
        xaxis=dict(range=[pd.to_datetime('2019-07-01'), pd.to_datetime('2026-06-30')])
    )
    st.plotly_chart(fig)
    st.info("https://grok.com/share/bGVnYWN5LWNvcHk%3D_af78586a-002a-460b-a3a8-1fcf99456ae2")

def main():
    main_chatgpt()
   
if __name__ == "__main__":
    main()