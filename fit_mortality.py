from fit_to_data_streamlit import *
from mortality_yearly_per_capita import get_sterfte, get_bevolking, interface_opdeling
import streamlit as st
from scipy.optimize import curve_fit
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import cbsodata
import datetime

# ---------------------------------------------------------------------------
# Dynamic year helpers — everything derives from this so no hardcoding needed
# ---------------------------------------------------------------------------
_NOW = datetime.datetime.now()
CURRENT_YEAR: int = _NOW.year
# A "complete" year means all 12 months have passed, i.e. strictly before now.
# If we're partway through the current year, it is not yet complete.
LAST_COMPLETE_YEAR: int = CURRENT_YEAR - 1 if _NOW.month < 12 else CURRENT_YEAR
# Upper bound for CBS filter: exclude years that haven't started yet
MAX_DATA_YEAR: int = CURRENT_YEAR  # CBS may publish partial current-year data


@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata():
    """Get and manipulate data of the deaths. From CBS 70895ned"""

    def manipulate_data_df(data):
        """Filters out week 0 and 53 and makes a category column (eg. 'M_V_0_999')"""
        data["weeknr"] = (
            data["jaar"].astype(str) + "_" + data["week"].astype(str).str.zfill(2)
        )
        data["sex"] = data["Geslacht"].replace(["Totaal mannen en vrouwen"], "T")
        data["sexe"] = data["Geslacht"].replace(["Mannen"], "M")
        data["sexe"] = data["Geslacht"].replace(["Vrouwen"], "F")
        data["age"] = data["LeeftijdOp31December"].replace(["Totaal leeftijd"], "TOTAL")
        data["age"] = data["LeeftijdOp31December"].replace(["0 tot 65 jaar"], "Y0_64")
        data["age"] = data["LeeftijdOp31December"].replace(["65 tot 80 jaar"], "Y65_79")
        data["age"] = data["LeeftijdOp31December"].replace(["80 jaar of ouder"], "Y80_999")
        return data

    data_ruw = pd.DataFrame(cbsodata.get_data("70895ned"))
    data_ruw[["jaar", "week"]] = data_ruw.Perioden.str.split(" week ", expand=True)
    data_ruw = manipulate_data_df(data_ruw)
    data_ruw["jaar"] = data_ruw["jaar"].astype(int)
  
    data_bevolking = pd.DataFrame(cbsodata.get_data("03759ned"))



# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def logistic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """a + ((b-a)/(1+((x/c)**d)))"""
    return a + ((b - a) / (1 + ((x / c) ** d)))


def gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))


def first_derivative_gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """First derivative of the Gompertz function."""
    return a * b * c * np.exp(b * (-1 * np.exp(-c * x)) - c * x)

def gaussian(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a * exp(-((x - b)^2) / c)"""
    return a * np.exp(-((x - b) ** 2) / c)

def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """a + b * x"""
    return a + (b * x)

def exponential_2(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """a * ((1 + b)^x)"""
    return a * ((1 + b) ** x)

# ---------------------------------------------------------------------------
# Normalisation / initial-parameter helpers
# ---------------------------------------------------------------------------

def x_norm(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    x0 = float(np.mean(x))
    scale = 10.0
    return (x - x0) / scale, x0, scale


def expo_p0(x_fit: np.ndarray, y: np.ndarray) -> list[float]:
    c0 = float(np.percentile(y, 10))
    y_shift = np.clip(y - c0, 1e-6, None)
    b0, loga = np.polyfit(x_fit, np.log(y_shift), 1)
    a0 = float(np.exp(loga))
    return [a0, float(b0), c0]

def expo_bounds(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    rng = max(1.0, y_max - y_min)
    mrg = max(1.0, 0.05 * rng)
    eps = 1e-9
    lower = np.array([0.0, -5.0, y_min - mrg], dtype=float)
    upper = np.array([np.inf, 5.0, y_max + mrg], dtype=float)
    upper = np.maximum(upper, lower + eps)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

#@st.cache_data()
def get_data(opdeling) -> pd.DataFrame:
    """Fetch mortality data using get_sterfte with age group breakdown."""
    return get_sterfte(opdeling, "NL")


@st.cache_data()
def get_doodsoorzaken_cbs():
    return pd.DataFrame(cbsodata.get_data("7052_95"))

#@st.cache_data()
def get_doodsoorzaken(opdeling) -> pd.DataFrame:
 
    data = get_doodsoorzaken_cbs()
    data.to_csv("doodsoorzaken.csv")  
    print ("Doodsoorzaken opgeslagen")
    # Melt all cause-of-death columns
    df = data.melt(
        id_vars=['ID', 'Geslacht', 'Leeftijd', 'Perioden'], 
        value_vars=data.columns.difference(['ID', 'Geslacht', 'Leeftijd', 'Perioden']), 
        var_name='doodsoorzaak', 
        value_name='OBS_VALUE'
    )
    
    # Rename geslacht values
    df['Geslacht'] = df['Geslacht'].replace({
        'Mannen': 'M',
        'Vrouwen': 'F',
        'Totaal mannen en vrouwen': 'T'
    })
    df = df.rename(columns={'Geslacht': 'geslacht'})

    import re

    def format_age_group(leeftijd):
        """Convert CBS age label to Ylow-high format."""
        if leeftijd == 'Totaal alle leeftijden':
            return 'Total'
        if leeftijd == '0 jaar':
            return 'Y0-4'
        if leeftijd in ('90 tot 95 jaar', '95 jaar of ouder'):
            return 'Y90-120'
        pattern = r'(\d+) tot (\d+) jaar'
        match = re.match(pattern, leeftijd)
        if match:
            low = int(match.group(1))
            high = int(match.group(2)) - 1
            return f"Y{low}-{high}"
        return leeftijd  # fallback: leave as-is

    df['Leeftijd'] = df['Leeftijd'].apply(format_age_group)
    df = df.rename(columns={'Leeftijd': 'age_group'})

    def extract_age_ranges(age):
        """Extract (age_low, age_high) from age group string like Y80-120."""
        if age == 'Total':
            return 999, 999
        if age == 'UNK':
            return 9999, 9999
        if age == 'Y_LT5':
            return 0, 4
        try:
            parts = age.lstrip('Y').split('-')
            return int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            return 9999, 9999  # malformed → excluded from custom groups

    df['age_low'], df['age_high'] = zip(*df['age_group'].apply(extract_age_ranges))

    # Group by relevant columns (drop ID which is irrelevant after melt)
    df = df.groupby(
        ['geslacht', 'age_group', 'age_low', 'age_high', 'Perioden', 'doodsoorzaak'],
        as_index=False
    )['OBS_VALUE'].sum()

    df = df.rename(columns={'Perioden': 'jaar'})
    df['jaar'] = df['jaar'].astype(int)
    df['age_sex'] = df['age_group'] + '_' + df['geslacht']

    # Build custom age groups from opdeling
    def add_custom_age_group_deaths(df_, min_age, max_age):
        """Sum deaths across all raw age groups within [min_age, max_age]."""
        df_filtered = df_[
            (df_['age_low'] >= min_age) & 
            (df_['age_high'] <= max_age) &
            (df_['age_low'] != 999) &   # exclude Total
            (df_['age_low'] != 9999)    # exclude UNK/malformed
        ]
        totals = df_filtered.groupby(
            ['jaar', 'geslacht', 'doodsoorzaak'],
            observed=False
        )['OBS_VALUE'].sum().reset_index()
        totals['age_group'] = f'Y{min_age}-{max_age}'
        totals['age_sex'] = totals['age_group'] + '_' + totals['geslacht']
        totals['age_low'] = min_age
        totals['age_high'] = max_age
        return totals

    df_custom_age_groups = pd.concat(
        [add_custom_age_group_deaths(df, i[0], i[1]) for i in opdeling],
        ignore_index=True
    )

    # FIX: only keep original rows NOT covered by a custom group
    # so there is no double-counting when e.g. Y80-120 and Y80-84/Y85-89/Y90-120 coexist
    custom_age_group_labels = {f'Y{i[0]}-{i[1]}' for i in opdeling}
    df_originals_only = df[~df['age_group'].isin(custom_age_group_labels)]
    df = pd.concat([df_custom_age_groups, df_originals_only], ignore_index=True)

    # Sanity check: no duplicate age_group per jaar/geslacht/doodsoorzaak
    dupe_check = df.groupby(['jaar', 'geslacht', 'age_group', 'doodsoorzaak']).size()
    if (dupe_check > 1).any():
        pass
        # st.warning("⚠️ Duplicate age groups detected after concat — check opdeling logic")

    # Merge with population data
    df_bevolking = get_bevolking("NL", opdeling)
    df_eind = pd.merge(df, df_bevolking, on=['geslacht', 'age_group', 'jaar'], how='left')

    # Clean up
    df_eind = df_eind[df_eind['aantal'].notna()]
    df_eind = df_eind[df_eind['OBS_VALUE'].notna()]
    df_eind['per100k'] = round(df_eind['OBS_VALUE'] / df_eind['aantal'] * 100000, 1)

    return df_eind

@st.cache_data()
def get_doodsoorzaken_oud(opdeling) -> pd.DataFrame:
    data = get_doodsoorzaken_cbs()

    df = data.melt(
        id_vars=["ID", "Geslacht", "Leeftijd", "Perioden"],
        value_vars=data.columns.difference(["ID", "Geslacht", "Leeftijd", "Perioden"]),
        var_name="doodsoorzaak",
        value_name="OBS_VALUE",
    )

    df["Geslacht"] = df["Geslacht"].replace(
        {"Mannen": "M", "Vrouwen": "F", "Totaal mannen en vrouwen": "T"}
    )
    df = df.rename(columns={"Geslacht": "Sexe"})

    import re

    df["Leeftijd"] = df["Leeftijd"].replace(
        {
            "Totaal alle leeftijden": "Total",
            "0 jaar": "Y0-4",
            "90 tot 95 jaar": "Y90-120",
            "95 jaar of ouder": "Y90-120",
        }
    )

    def format_age_group(leeftijd: str) -> str:
        match = re.match(r"(\d+) tot (\d+) jaar", leeftijd)
        if match:
            low_age = int(match.group(1))
            high_age = int(match.group(2)) - 1
            return f"Y{low_age}-{high_age}"
        return leeftijd

    df["Leeftijd"] = df["Leeftijd"].apply(format_age_group)
    df = df.rename(columns={"Leeftijd": "age_group"})

    df = df.groupby(
        ["Sexe", "age_group", "Perioden", "doodsoorzaak"], as_index=False
    )["OBS_VALUE"].sum()
    df = df.rename(columns={"Perioden": "jaar", "Sexe": "geslacht"})
    df["jaar"] = df["jaar"].astype(int)

    df_bevolking = get_bevolking("NL", opdeling)

    def extract_age_ranges(age: str) -> tuple[int, int]:
        if age == "Total":
            return 999, 999
        if age == "UNK":
            return 9999, 9999
        if age == "Y_LT5":
            return 0, 4
        if age == "Y_90-120":
            return 90, 120
        parts = age[1:].split("-")
        return int(parts[0]), int(parts[1])

    df["age_low"], df["age_high"] = zip(*df["age_group"].apply(extract_age_ranges))
    df["age_sex"] = df["age_group"] + "_" + df["geslacht"]

    def add_custom_age_group_deaths(df_: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
        df_filtered = df_[(df_["age_low"] >= min_age) & (df_["age_high"] <= max_age)]
        totals = df_filtered.groupby(
            ["jaar", "geslacht", "doodsoorzaak"], observed=False
        )["OBS_VALUE"].sum().reset_index()
        totals["age_group"] = f"Y{min_age}-{max_age}"
        totals["age_sex"] = totals["age_group"] + "_" + totals["geslacht"]
        return totals

    df_custom_age_groups = pd.concat(
        [add_custom_age_group_deaths(df, i[0], i[1]) for i in opdeling],
        ignore_index=True,
    )
    df = pd.concat([df_custom_age_groups, df], ignore_index=True)

    df_eind = pd.merge(df, df_bevolking, on=["geslacht", "age_group", "jaar"], how="left")
    df_eind = df_eind[df_eind["aantal"].notna()]
    df_eind = df_eind[df_eind["OBS_VALUE"].notna()]

    # Keep all complete years — exclude any future year beyond the last complete one
    df_eind = df_eind[df_eind["jaar"] <= LAST_COMPLETE_YEAR]

    df_eind["per100k"] = round(df_eind["OBS_VALUE"] / df_eind["aantal"] * 100000, 1)
    return df_eind


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def main_(
    df: pd.DataFrame,
    value_field: str,
    age_group: str,
    sexe: str,
    START_YEAR: int,
    verbose: bool,
    secondary_choice_: list[str],
    show_confidence_intervals: bool,
    doordsoorzaak_keuze: str,
    what_to_plot: list[str],
    scaled: bool,
) -> tuple[float, float]:
    """Main analysis: secondary + linear curve fitting, projections, and plotting."""

    df_before_2020, df_2020_and_up = prepare_data(df, age_group, sexe, START_YEAR)
    x_ = df_before_2020["jaar"]
    y_ = df_before_2020[value_field]

    trendline, extended_years, trendline_extended = fit_and_predict(df_before_2020, x_, y_)

    if value_field == "OBS_VALUE":
        df_before_2020["predicted_deaths"] = trendline
        df_extended = pd.merge(
            df_2020_and_up,
            pd.DataFrame({"jaar": extended_years, "predicted_deaths": trendline_extended}),
            on="jaar",
        )
    else:
        df_before_2020["predicted_per100k"] = trendline
        df_extended = pd.merge(
            df_2020_and_up,
            pd.DataFrame({"jaar": extended_years, "predicted_per100k": trendline_extended}),
            on="jaar",
        )

    df_diff = pd.concat([df_before_2020, df_extended], ignore_index=True)
    df_diff = df_diff.sort_values(by="jaar").reset_index(drop=True)

    if value_field == "per100k":
        df_diff["predicted_deaths"] = (
            df_diff["predicted_per100k"] * df_diff["aantal"] / 100000
        )

    df_diff = do_calculations_df_diff_lineair(df_diff)
    result_str = []

    function_info = {
        "quadratic": {
            "func": quadratic,
            "p0": [1, 1, 1],
            "equation": "a*x^2 + b*x + c",
            "params": ["a", "b", "c"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "exponential": {
            "func": exponential,
            "p0": lambda x, y: expo_p0(x, y),
            "equation": "a*exp(b*x) + c",
            "params": ["a", "b", "c"],
            "bounds": lambda x, y: expo_bounds(y),
            "use_norm": True,
        },
        "gompertz": {
            "func": gompertz,
            "p0": [1, 1, 1],
            "equation": "a * exp(-b * exp(-c * x))",
            "params": ["a", "b", "c"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "first_derivative_gompertz": {
            "func": first_derivative_gompertz,
            "p0": [1, 1, 1],
            "equation": "a * b * c * exp(b * (-1 * exp(-c * x)) - c * x)",
            "params": ["a", "b", "c"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "gaussian": {
            "func": gaussian,
            "p0": lambda x: [1, np.mean(x), np.std(x)],
            "equation": "a * exp(-((x - b)^2) / c)",
            "params": ["a", "b", "c"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "linear": {
            "func": linear,
            "p0": [1, 1],
            "equation": "a + b*x",
            "params": ["a", "b"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "exponential_2": {
            "func": exponential_2,
            "p0": [1, 1],
            "equation": "a * ((1 + b)^x)",
            "params": ["a", "b"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
        "logistic": {
            "func": logistic,
            "p0": [1, 1, 1, 1],
            "equation": "a + ((b-a)/(1+((x/c)**d)))",
            "params": ["a", "b", "c", "d"],
            "bounds": lambda x, y: (-np.inf, np.inf),
            "use_norm": False,
        },
    }

    for secondary_choice in secondary_choice_:
        try:
            mask = np.isfinite(x_) & np.isfinite(y_)
            x_clean = x_[mask]
            y_clean = y_[mask]

            if secondary_choice not in function_info:
                st.warning(f"Error in secondary choice {secondary_choice}.")
                st.stop()

            info = function_info[secondary_choice]

            if info.get("use_norm", False):
                x_fit, x0, xscale = x_norm(x_clean)
            else:
                x_fit = x_clean

            p0_raw = info["p0"]
            if callable(p0_raw):
                try:
                    p0 = p0_raw(x_fit, y_clean)
                except TypeError:
                    p0 = p0_raw(x_fit)
            else:
                p0 = p0_raw

            bounds_raw = info.get("bounds", lambda x, y: (-np.inf, np.inf))
            bounds = bounds_raw(x_fit, y_clean)

            if isinstance(bounds, tuple) and len(bounds) == 2:
                lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
                if not np.isscalar(lo):
                    eps = 1e-9
                    p0 = np.clip(np.asarray(p0, float), lo + eps, hi - eps)

            pars, cov = curve_fit(
                f=info["func"],
                xdata=x_fit,
                ydata=y_clean,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )

            param_str = ", ".join(
                f"{param} = {value:.4f}" for param, value in zip(info["params"], pars)
            )
            result_str.append(f"*{secondary_choice}* - {info['equation']} | {param_str}")
            df_diff = do_calculations_df_diff_secondary_choice(pars, cov, df_diff, secondary_choice)

        except Exception as error:
            print(f"No fitting possible for {secondary_choice} - {error}")

    if verbose:
        show_result_str = False
        if (value_field == "OBS_VALUE" and "OBS_VALUE" in what_to_plot) or (
            value_field == "per100k" and "per100k" in what_to_plot
        ):
            if not scaled:
                plot_fitting_on_value_field(
                    value_field, df_before_2020, df_2020_and_up,
                    trendline, extended_years, trendline_extended,
                    df_diff, age_group, sexe, secondary_choice_, doordsoorzaak_keuze,
                )
            else:
                plot_fitting_on_value_field_scaled(
                    value_field, df_before_2020, df_2020_and_up,
                    trendline, extended_years, trendline_extended,
                    df_diff, age_group, sexe, secondary_choice_, doordsoorzaak_keuze,
                )
            show_result_str = True

        if value_field == "per100k":
            if "number_of_people" in what_to_plot or "100k_to_population" in what_to_plot:
                st.subheader("**From per 100k transformation back to Absolute Numbers**")
            if "number_of_people" in what_to_plot:
                plot_group_size(df_diff, age_group, sexe, doordsoorzaak_keuze)
                show_result_str = True
            if "100k_to_population" in what_to_plot:
                plot_transformed_to_absolute(
                    df_before_2020, df_2020_and_up, df_diff,
                    age_group, sexe, secondary_choice_, doordsoorzaak_keuze,
                )
                show_result_str = True

        if show_result_str:
            for r in result_str:
                st.write(r)
        else:
            verbose = False

        excess_mortality_lineair, excess_mortality_secondary_ = show_excess_mortality(
            value_field, df_diff, verbose, secondary_choice_
        )

    return excess_mortality_lineair, excess_mortality_secondary_


def show_excess_mortality(
    value_field: str,
    df_diff: pd.DataFrame,
    verbose: bool,
    secondary_choice_: list[str],
) -> tuple[float, list]:
    """Display excess mortality figures for all complete years from 2020 onward."""

    # Use all complete years from 2020 up to and including LAST_COMPLETE_YEAR
    post_2020_mask = df_diff["jaar"].between(2020, LAST_COMPLETE_YEAR)

    excess_mortality_lineair = round(df_diff[post_2020_mask]["oversterfte"].sum())
    excess_mortality_secondary_ = []

    for secondary_choice in secondary_choice_:
        try:
            if value_field == "per100k":
                col = f"oversterfte_expon_{secondary_choice}"
            else:
                col = f"oversterfte_expon_totals_{secondary_choice}"
            excess_mortality_secondary = round(df_diff[post_2020_mask][col].sum())
        except Exception:
            excess_mortality_secondary = None

        if verbose and excess_mortality_secondary is not None:
            excess_per_year = round(
                excess_mortality_secondary / max(1, LAST_COMPLETE_YEAR - 2019)
            )
            st.write(
                f"{value_field} - Excess mortality {secondary_choice} "
                f"{excess_mortality_secondary} | {excess_per_year} per year "
                f"(2020–{LAST_COMPLETE_YEAR})"
            )

        excess_mortality_secondary_.append(excess_mortality_secondary)

    return excess_mortality_lineair, excess_mortality_secondary_


def do_calculations_df_diff_lineair(df_diff: pd.DataFrame) -> pd.DataFrame:
    df_diff["oversterfte"] = round(df_diff["OBS_VALUE"] - df_diff["predicted_deaths"])
    df_diff["aantal"] = round(df_diff["aantal"])
    df_diff["percentage"] = round(
        ((df_diff["OBS_VALUE"] - df_diff["predicted_deaths"]) / df_diff["predicted_deaths"]) * 100,
        1,
    )
    return df_diff


def do_calculations_df_diff_secondary_choice(
    pars: np.ndarray,
    pcov: np.ndarray,
    df_diff: pd.DataFrame,
    secondary_choice: str,
) -> pd.DataFrame:
    perr = 0
    n_std = 0.0

    function_map = {
        "exponential": exponential,
        "quadratic": quadratic,
        "gompertz": gompertz,
        "first_derivative_gompertz": first_derivative_gompertz,
        "gaussian": gaussian,
        "linear": linear,
        "exponential_2": exponential_2,
        "logistic": logistic,
    }

    if secondary_choice not in function_map:
        st.write(f"Error in secondary choice |{secondary_choice}|")
        st.stop()

    func = function_map[secondary_choice]
    df_diff[f"fitted_curve_{secondary_choice}"] = func(df_diff["jaar"], *pars)
    df_diff[f"y_fit_upper_{secondary_choice}"] = func(df_diff["jaar"], *(pars + n_std * perr))
    df_diff[f"y_fit_lower_{secondary_choice}"] = func(df_diff["jaar"], *(pars - n_std * perr))
    df_diff[f"fitted_curve_transf_absolut_{secondary_choice}"] = (
        df_diff[f"fitted_curve_{secondary_choice}"] * df_diff["aantal"] / 100000
    )
    df_diff[f"oversterfte_expon_totals_{secondary_choice}"] = (
        df_diff["OBS_VALUE"] - df_diff[f"fitted_curve_{secondary_choice}"]
    )
    df_diff[f"oversterfte_expon_{secondary_choice}"] = round(
        df_diff["OBS_VALUE"] - df_diff[f"fitted_curve_transf_absolut_{secondary_choice}"]
    )
    df_diff = df_diff.sort_values(by="jaar").reset_index(drop=True)
    return df_diff


def fit_and_predict(
    df_before_2020: pd.DataFrame,
    x_: pd.Series,
    y_: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear OLS trend and predict up to (and including) LAST_COMPLETE_YEAR."""
    X = sm.add_constant(x_)
    model = sm.OLS(y_, X).fit()
    trendline = model.predict(X)

    # Extend projection through the last complete year
    extended_years = np.arange(df_before_2020["jaar"].min(), LAST_COMPLETE_YEAR + 1)
    extended_X = sm.add_constant(extended_years)
    trendline_extended = model.predict(extended_X)

    return trendline, extended_years, trendline_extended


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _pre2020_mask(df_diff: pd.DataFrame) -> pd.DataFrame:
    """Return only rows before 2020 — used for R² to avoid slicing by count."""
    return df_diff[df_diff["jaar"] < 2020]


def plot_fitting_on_value_field(
    value_field: str,
    df_before_2020: pd.DataFrame,
    df_2020_and_up: pd.DataFrame,
    trendline: np.ndarray,
    extended_years: np.ndarray,
    trendline_extended: np.ndarray,
    df_diff: pd.DataFrame,
    age_group: str,
    sexe: str,
    secondary_choice_: list[str],
    doordsoorzaak_keuze: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_before_2020["jaar"], y=df_before_2020[value_field],
        name="before 2020", mode="markers", marker=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=df_2020_and_up["jaar"], y=df_2020_and_up[value_field],
        name="2020 and up", mode="markers", marker=dict(color="red"),
    ))

    df_filtered = _pre2020_mask(df_diff)
    title = f"{age_group} - {sexe} | {value_field} | {doordsoorzaak_keuze}"
    colors = ["orange", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

    for i, secondary_choice in enumerate(secondary_choice_):
        try:
            r2_b = round(
                r2_score(df_filtered[value_field], df_filtered[f"fitted_curve_{secondary_choice}"]),
                4,
            )
            title += f"<br>r2 {secondary_choice}: {r2_b}"
            fig.add_trace(go.Scatter(
                x=df_diff["jaar"], y=df_diff[f"fitted_curve_{secondary_choice}"],
                mode="lines", line=dict(color=colors[i]),
                name=f"Fitted {secondary_choice} Curve",
            ))
        except Exception:
            pass

    fig.update_layout(title=title, xaxis_title="Year", yaxis_title=value_field)
    st.plotly_chart(fig)


def plot_fitting_on_value_field_scaled(
    value_field: str,
    df_before_2020: pd.DataFrame,
    df_2020_and_up: pd.DataFrame,
    trendline: np.ndarray,
    extended_years: np.ndarray,
    trendline_extended: np.ndarray,
    df_diff: pd.DataFrame,
    age_group: str,
    sexe: str,
    secondary_choice_: list[str],
    doordsoorzaak_keuze: str,
) -> None:
    value_curve = df_diff[value_field]
    colors = ["orange", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

    for i, secondary_choice in enumerate(secondary_choice_):
        fig = go.Figure()
        try:
            fitted_curve = df_diff[f"fitted_curve_{secondary_choice}"]
            values = (value_curve - fitted_curve) / fitted_curve
            years = df_diff["jaar"]

            positive_years = [years[j] for j in range(len(values)) if values[j] >= 0]
            positive_values = [v for v in values if v >= 0]
            negative_years = [years[j] for j in range(len(values)) if values[j] < 0]
            negative_values = [v for v in values if v < 0]

            fig.add_trace(go.Bar(
                x=positive_years, y=positive_values,
                name="Positive Values", marker=dict(color="blue"),
            ))
            fig.add_trace(go.Bar(
                x=negative_years, y=negative_values,
                name="Negative Values", marker=dict(color="red"),
            ))
        except KeyError:
            pass

        title = f"{age_group} - {sexe} | {value_field} | {doordsoorzaak_keuze} | {secondary_choice}"
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Relative Value (Base = Trendline)",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig)


def plot_transformed_to_absolute(
    df_before_2020: pd.DataFrame,
    df_2020_and_up: pd.DataFrame,
    df_diff: pd.DataFrame,
    age_group: str,
    sexe: str,
    secondary_choice_: list[str],
    doordsoorzaak_keuze: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_before_2020["jaar"], y=df_before_2020["OBS_VALUE"],
        name="before 2020", mode="markers", marker=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=df_2020_and_up["jaar"], y=df_2020_and_up["OBS_VALUE"],
        name="2020 and up", mode="markers", marker=dict(color="red"),
    ))
    fig.add_trace(go.Scatter(
        x=df_diff["jaar"], y=df_diff["predicted_deaths"],
        name="trendline OLS", mode="lines", marker=dict(color="green"),
    ))

    df_filtered = _pre2020_mask(df_diff)
    title = (
        f"{age_group} - {sexe} | {doordsoorzaak_keuze} | "
        "Deaths Transformed from relative back to absolute numbers"
    )
    r2 = round(r2_score(df_filtered["OBS_VALUE"], df_filtered["predicted_deaths"]), 4)
    title += f" | r2 OLS: {r2}"
    colors = ["orange", "purple", "cyan", "magenta", "yellow", "brown", "pink"]

    for i, secondary_choice in enumerate(secondary_choice_):
        try:
            df_diff[f"fitted_aantal_{secondary_choice}"] = (
                df_diff[f"fitted_curve_{secondary_choice}"] * df_diff["aantal"] / 100000
            )
            fig.add_trace(go.Scatter(
                x=df_diff["jaar"], y=df_diff[f"fitted_aantal_{secondary_choice}"],
                mode="lines", line=dict(color=colors[i]),
                name=f"Fitted {secondary_choice} Curve",
            ))
            r2_b = round(
                r2_score(df_diff["OBS_VALUE"], df_diff[f"fitted_aantal_{secondary_choice}"]),
                4,
            )
            title += f" | r2 {secondary_choice}: {r2_b}"
        except Exception as error:
            print(f"{secondary_choice}: graphline failed | {error}")

    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Deaths")
    st.plotly_chart(fig)


def plot_group_size(
    df_diff: pd.DataFrame, age_group: str, sexe: str, doordsoorzaak_keuze: str
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_diff["jaar"], y=df_diff["aantal"],
        name="Population", marker=dict(color="blue"),
    ))
    fig.update_layout(
        title=f"{age_group} - {sexe} | Number of people in the population",
        xaxis_title="Year",
        yaxis_title="Count",
    )
    st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame, age_group: str, sexe: str, START_YEAR: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["age_group"] == age_group]
    df = df[df["geslacht"] == sexe]
    df_before_2020 = df[(df["jaar"] >= START_YEAR) & (df["jaar"] < 2020)]
    df_2020_and_up = df[df["jaar"] >= 2020]
    
    return df_before_2020, df_2020_and_up


def calculate_results(
    df: pd.DataFrame,
    age_groups_selected_: list[str],
    start_years: list[int],
    sexe: str,
    verbose: bool,
    secondary_choice_: list[str],
    show_confidence_intervals: bool,
    doordsoorzaak_keuze: str,
    what_to_plot: list[str],
    scaled: bool,
) -> pd.DataFrame:
    """Calculate excess mortality for each age group, value field, and start year."""
    if not isinstance(age_groups_selected_, list):
        age_groups_selected_ = [age_groups_selected_]

    counter = 0
    total = 2 * len(age_groups_selected_) * len(start_years)
    results = []

    for value_field in ["OBS_VALUE", "per100k"]:
        for age_group in age_groups_selected_:
            for START_YEAR in start_years:
                print(f"{counter + 1}/{total} | {value_field=} - {age_group=} {START_YEAR=}")
                excess_mortality_lineair, excess_mortality_secondary_ = main_(
                    df, value_field, age_group, sexe, START_YEAR,
                    verbose, secondary_choice_, show_confidence_intervals,
                    doordsoorzaak_keuze, what_to_plot, scaled,
                )
                results.append({
                    "start_year": START_YEAR,
                    "model": "lineair",
                    "value_field": value_field,
                    "age_group": age_group,
                    "excess_mortality": excess_mortality_lineair,
                })
                for secondary_choice, excess_mortality_secondary in zip(
                    secondary_choice_, excess_mortality_secondary_
                ):
                    results.append({
                        "start_year": START_YEAR,
                        "model": secondary_choice,
                        "value_field": value_field,
                        "age_group": age_group,
                        "excess_mortality": excess_mortality_secondary,
                    })
                counter += 1

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.subheader("Mortality Analysis Using Secondary Curve Fitting and Trendline Projection")
    st.info(f"""
        This app analyses mortality data using historical trends up to {LAST_COMPLETE_YEAR}.
        It fits linear and secondary curves to pre-2020 data, then projects them forward
        to estimate excess mortality from 2020 through {LAST_COMPLETE_YEAR}.

        Inspired by https://twitter.com/HansV_16/status/1838178383835848708
    """)
    st.info("https://rene-smit.com/low-excess-mortality-observed-using-quadratic-fitting-in-mortality-trend-analysis/")

    opdeling = [[0, 120], [0, 64], [65, 79], [80, 120]] + interface_opdeling()
    df_doodsoorzaken = get_doodsoorzaken(opdeling)

    doodsoorzaken = ["ALLE DOODSOORZAKEN"] + df_doodsoorzaken["doodsoorzaak"].unique().tolist()
    preset = st.sidebar.selectbox("Preset", ["None", "accuut_hartinfarct"])

    if preset == "None":
        doodsoorz_def = 0
        sec_choice_def = ["linear", "quadratic"]
        age_groups_def = 1
        what_to_plot_def = ["OBS_VALUE", "per100k", "number_of_people", "100k_to_population"]
    else:
        doodsoorz_def = 83
        sec_choice_def = ["linear"]
        age_groups_def = 0
        what_to_plot_def = ["per100k"]

    doodsoorzaak_keuze = st.sidebar.selectbox("Doodsoorzaak", doodsoorzaken, doodsoorz_def)
    if doodsoorzaak_keuze == "ALLE DOODSOORZAKEN":
        df = get_data(opdeling)
        doodsoorzaak_keuze = ""
    else:
        df = df_doodsoorzaken[df_doodsoorzaken["doodsoorzaak"] == doodsoorzaak_keuze]
   
    age_groups_ = df["age_group"].unique().tolist()
    priority_groups = [f"Y{start:02d}-{end:02d}" for start, end in opdeling]

    def get_starting_age(age_group: str) -> int:
        if age_group == "TOTAL":
            return 0
        try:
            return int(age_group[1:3])
        except Exception:
            return int(age_group[1:2])

    age_groups_sorted = sorted(age_groups_, key=get_starting_age)
    age_groups = ["ALLE LEEFTIJDEN IN EEN LOOP"] + age_groups_sorted

    sexe = st.sidebar.selectbox("Sexe [T|M|F]", ["T", "M", "F"], 0)
    possible_columns = [
        ["model", "value_field", "start_year"],
        ["model", "start_year", "value_field"],
        ["value_field", "model", "start_year"],
        ["value_field", "start_year", "model"],
        ["start_year", "model", "value_field"],
        ["start_year", "value_field", "model"],
    ]
    columns = st.sidebar.selectbox("Column hierarchie", possible_columns, 0)

    age_groups_selected = st.sidebar.multiselect("age group", age_groups, ["Y0-120"])
    start_years = [st.sidebar.number_input("Fitting from year", 1950, 2019, 2010)]
    verbose = True
    secondary_choice_ = st.sidebar.multiselect(
        "Secondary choice",
        ["linear", "exponential", "quadratic", "gompertz", "first_derivative_gompertz",
         "gaussian", "exponential_2", "logistic"],
        sec_choice_def,
    )
    if len(secondary_choice_) == 0:
        st.warning("Choose at least one secondary choice")
        st.stop()

    what_to_plot = st.sidebar.multiselect(
        "What to plot",
        ["OBS_VALUE", "per100k", "number_of_people", "100k_to_population"],
        what_to_plot_def,
    )
    show_confidence_intervals = st.sidebar.checkbox("Show confidence intervals", value=False)

    if "OBS_VALUE" in what_to_plot or "per100k" in what_to_plot:
        scaled = st.sidebar.checkbox("Show relative values", value=False)
    else:
        scaled = False

    if not age_groups_selected:
        st.error("Select agegroup(s) to continue.")
        st.stop()

    if age_groups_selected[0] == "ALLE LEEFTIJDEN IN EEN LOOP":
        for age_groups_selected_x in age_groups_:
            st.subheader(age_groups_selected_x)
            df_results = calculate_results(
                df, age_groups_selected_x, start_years, sexe, verbose,
                secondary_choice_, show_confidence_intervals,
                doodsoorzaak_keuze, what_to_plot, scaled,
            )
    else:
        for age_groups_selected_x in age_groups_selected:
            st.subheader(age_groups_selected_x)
            df_results = calculate_results(
                df, [age_groups_selected_x], start_years, sexe, verbose,
                secondary_choice_, show_confidence_intervals,
                doodsoorzaak_keuze, what_to_plot, scaled,
            )
            df_results.to_csv(f"data_{age_groups_selected_x}.csv")

    df_pivot = df_results.pivot_table(
        index="age_group",
        columns=columns,
        values="excess_mortality",
    )
    for col in df_pivot.columns:
        df_pivot[col] = df_pivot[col].astype(str)

    st.subheader(f"Excess Mortality Comparison (2020–{LAST_COMPLETE_YEAR})")
    st.dataframe(df_pivot)
    st.info("Doodsoorzaken: https://www.cbs.nl/nl-nl/cijfers/detail/7052_95")
    st.info("Bevolkingsgrootte: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.info("Sterfte: https://ec.europa.eu/eurostat/databrowser/view/DEMO_R_MWK_05__custom_20841811/default/table")

if __name__ == "__main__":
    import os

    os.system("cls")
    print(f"--------------{datetime.datetime.now()}----x---------------------")
    main()