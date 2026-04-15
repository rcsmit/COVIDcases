# covid_catalogue.py
# ---------------------------------------------------------------------------
# Single source of truth for the COVID script catalogue and category definitions.
# Both covid_menu_streamlit.py and covid_welcome.py import from here.
#
# Usage:
#   from covid_catalogue import options, CATEGORIES, CHOICE_TO_CAT_I, LETTER_TO_CAT_I
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Script catalogue  [display_label, module_name, description]
# ---------------------------------------------------------------------------
options = [
    ["[0] welcome",                          "covid_welcome",                                           "Landing page and overview of all COVID scripts"],
    ["[1] Covid dashboard",                  "covid_dashboard_rcsmit",                            "Main COVID dashboard for the Netherlands"],
    ["[2] Plot hosp/IC per age",             "plot_hosp_ic_streamlit",                            "Hospital and IC admissions per age group"],
    ["[3] False positive rate covid test",   "calculate_false_positive_rate_covid_test_streamlit","False positive rate calculator for COVID tests"],
    ["[4] Number of cases interactive",      "number_of_cases_interactive",                       "Interactive case count visualization"],
    ["[5] IFR from prevalence",              "calculate_ifr_from_prevalence_streamlit",           "Infection fatality rate derived from seroprevalence data"],
    ["[6] Fit to data",                      "fit_to_data_streamlit",                             "Curve fitting to COVID case data"],
    ["[7] SEIR hobbeland",                   "SEIR_hobbeland",                                    "Classic SEIR epidemic model (hobbeland variant)"],
    ["[8] Show contactmatrix",               "show_contactmatrix",                                "Age-structured contact matrix visualizer"],
    ["[9] R getal per provincie",            "r_getal_per_provincie",                             "Reproduction number R per Dutch province"],
    ["[10] Cases from susceptibles",         "cases_from_susp_streamlit",                         "Case trajectory from susceptible pool dynamics"],
    ["[11] Fit to data OWID animated",       "fit_to_data_owid_streamlit_animated",               "Animated curve fitting to Our World in Data"],
    ["[12] Calculate R per country OWID",    "calculate_r_per_country_owid_streamlit",            "R number per country using OWID data"],
    ["[13] Covid dashboard OWID/Google",     "covid_dashboard_owid",                              "COVID dashboard with OWID, Google, and Waze mobility data"],
    ["[14] Dag verschillen per leeftijd",    "dag_verschillen_casus_landelijk",                   "Daily case differences per age group"],
    ["[15] Abs./rel. humidity from RH",      "rh2q",                                              "Calculate specific/absolute humidity from relative humidity"],
    ["[16] R getal per leeftijdscategorie",  "r_number_by_age",                                   "Reproduction number R per age category"],
    ["[17] Show rioolwaardes",               "show_rioolwater",                                   "Sewage water COVID signal visualization"],
    ["[18] SIR model met leeftijdsgroepen",  "SIR_age_structured_streamlit",                      "Age-structured SIR epidemic model"],
    ["[19] Pos testen per leeftijdscat.",    "grafiek_pos_testen_per_leeftijdscategorie_streamlit","Positive test rate per age category"],
    ["[20] Per provincie per leeftijd",      "perprovincieperleeftijd",                           "Cases per province per age group"],
    ["[21] Kans om COVID op te lopen",       "kans_om_covid_op_te_lopen",                         "Probability of contracting COVID"],
    ["[22] Data per gemeente",               "vacc_inkomen_cases",                                "Vaccination, income, and cases per municipality"],
    ["[23] VE Israel",                       "israel_zijlstra",                                   "Vaccine effectiveness — Israel (Zijlstra method)"],
    ["[24] Hosp/death NL",                   "cases_hospital_decased_NL",                         "Dutch hospitalizations and deaths over time"],
    ["[25] VE Nederland",                    "VE_nederland_",                                     "Vaccine effectiveness Netherlands"],
    ["[26] Scatterplots QoG OWID",           "qog_owid",                                          "Quality of Government vs OWID COVID scatterplots"],
    ["[27] VE & CI calculations",            "VE_CI_calculations",                                "Vaccine effectiveness and confidence interval calculations"],
    ["[28] VE scenario calculator",          "VE_scenario_calculator",                            "Vaccine effectiveness under different scenario assumptions"],
    ["[29] VE vs inv. odds",                 "VE_vs_inv_odds",                                    "Vaccine effectiveness vs inverse odds analysis"],
    ["[30] Fit to data Levitt",              "fit_to_data_owid_levitt_streamlit_animated",        "Animated Levitt-style curve fitting (OWID)"],
    ["[31] Aerosol concentration in room",   "aerosol_in_room_streamlit",                         "Aerosol concentration in room by @hk_nien"],
    ["[32] Compare two variants",            "compare_two_variants",                              "Side-by-side comparison of two COVID variants"],
    ["[33] Scatterplot OWID",                "scatterplots_owid",                                 "OWID country-level scatterplot explorer"],
    ["[34] Playing with R0",                 "playing_with_R0",                                   "Interactive R₀ sensitivity explorer"],
    ["[35] Calculate Se & Sp Rapidtest",     "calculate_se_sp_rapidtest_streamlit",               "Sensitivity and specificity calculator for rapid tests"],
    ["[36] Oversterfte gemeente",            "oversterfte_gemeente",                              "Excess mortality per Dutch municipality"],
    ["[37] Sterfte patronen",                "sterfte_2000_2024",                                 "Dutch mortality patterns 2000–2024"],
    ["[38] Bayes Lines tools",               "bayes_lines_tools",                                 "Bayesian line-fitting tools for mortality data"],
    ["[39] Oversterfte (CBS Odata)",         "oversterfte_compleet",                              "Excess mortality using CBS open data API"],
    ["[40] Bayes berekeningen IC/ziekenhuis","bayes_prob_ic_hosp",                                "Bayesian probability calculations for IC and hospital data"],
    ["[41] Disabled by Long covid",          "disabled_by_longcovid",                             "Disability burden estimates from Long COVID"],
    ["[42] Oversterfte 5yr Eurostats",       "oversterfte_eurostats_maand",                       "Monthly excess mortality — 5-year baseline, Eurostat"],
    ["[43] Doodsoorzaken Sankey",            "mortality_causes",                                  "Sankey diagram of causes of death"],
    ["[44] Rioolwaarde vs ziekenhuis",       "rioolwater_vs_ziekenhuis",                          "Sewage signal vs hospital admissions"],
    ["[45] Rioolwaarde vs overleden CBS",    "overledenen_rioolwaardes",                          "Sewage signal vs deaths (CBS data)"],
    ["[46] Mortality yearly per capita",     "mortality_yearly_per_capita",                       "Annual mortality per capita trend analysis"],
    ["[47] Deltavax",                        "deltavax",                                          "Delta variant and vaccination interaction model"],
    ["[48] Verwachte sterfte",               "verwachte_sterfte",                                 "Expected mortality baseline calculation"],
    ["[49] Logistic regression",             "logistic_regression",                               "Logistic regression on COVID outcome data"],
    ["[50] Calculate baselines (Poisson)",   "calculate_baselines",                               "Poisson-based mortality baseline calculator"],
    ["[51] AG table mortality",              "agtable_mortality",                                 "AG actuarial table mortality analysis"],
    ["[52] Find baseline length",            "find_baseline_length",                              "Sensitivity analysis on baseline window length"],
    ["[53] Mortality/week/100k",             "mortality_weekly_per_age_per_capita",               "Weekly mortality per age per 100,000 population"],
    ["[54] Herhaalprik",                     "herhaalprik",                                       "Booster vaccination uptake and effect analysis"],
    ["[55] Fit Mortality/causes death",      "fit_mortality",                                     "Curve fitting on cause-of-death mortality data"],
    ["[56] Bayes Mortality Vaccination",     "bayes_vaccination",                                 "Bayesian analysis of mortality and vaccination"],
    ["[57] Sterfte/rioolw./vaccins",         "correlatie_sterfte_rioolwater_vaccins",             "Correlation: mortality, sewage signal, vaccinations"],
    ["[58] SIR model agent based",           "SIR_agent_based_vector",                            "Vectorized agent-based SIR epidemic model"],
    ["[59] Rioolwater vs covidsterfte",      "rioolwaarde_vs_covidsterfte",                       "Sewage COVID signal vs COVID deaths"],
    ["[60] Oversterfte predict per levensjaar","oversterfte_predict_per_levensjaar",              "Excess mortality prediction per year of life"],
    ["[61] RIVM model",                      "rivm_model",                                        "Replicated RIVM epidemic model"],
    ["[62] Oversterfte GAM",                 "oversterfte_predict_per_levensjaar_nieuw",          "GAM-based excess mortality prediction"],
    ["[63] CBS-Oversterfte",                 "cbs_oversterfte",                                   "Excess mortality analysis with CBS data — back to the drawing board"],
]

# ---------------------------------------------------------------------------
# Categories  (letter_key, display_name, [choice_indices], hex_color)
#
# Rules:
#   - letter_key must be unique across all entries.
#   - Each script index may appear in at most one category.
#   - Scripts absent from every category are auto-collected into "Z".
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("A", "🏠  Home",                       [0],                                                          "#6C8EBF"),
    ("B", "🦠  Epidemic models",            [7, 8, 10, 18, 34, 58, 61],                                  "#E74C3C"),
    ("C", "📊  Cases & R number",           [1, 2, 4, 9, 12, 13, 14, 16, 19, 20],                        "#E67E22"),
    ("D", "🧪  Testing & Serology",         [3, 5, 15, 21, 31, 35],                                      "#F39C12"),
    ("E", "💉  Vaccines & Effectiveness",   [22, 23, 24, 25, 27, 28, 29, 47, 54, 56],                    "#27AE60"),
    ("F", "📈  Curve fitting",              [6, 11, 30],                                                  "#2980B9"),
    ("G", "⚰️  Mortality & Excess deaths",  [36, 37, 39, 41, 42, 43, 45, 46, 48, 50, 51, 52, 53, 55, 57, 59, 60, 62, 63], "#8E44AD"),
    ("H", "🚿  Sewage water",               [17, 44],                                                     "#16A085"),
    ("I", "🌍  International comparisons",  [26, 32, 33],                                                 "#2C3E50"),
    ("J", "📐  Statistics & Methods",       [49, 38, 40],                                                 "#7F8C8D"),
]


# ---------------------------------------------------------------------------
# Safety check 1: duplicate category letter keys
# Raises ValueError at import time so errors surface in both Streamlit and CLI.
# ---------------------------------------------------------------------------
def _check_no_duplicate_letters() -> None:
    seen: dict[str, str] = {}
    errors: list[str] = []
    for letter, cat_name, _idxs, _color in CATEGORIES:
        key = letter.upper()
        if key in seen:
            errors.append(f"  letter '{key}' used by '{seen[key]}' and '{cat_name}'")
        else:
            seen[key] = cat_name
    if errors:
        raise ValueError(
            "covid_catalogue.py — duplicate category letters:\n" + "\n".join(errors)
        )


_check_no_duplicate_letters()


# ---------------------------------------------------------------------------
# Safety check 2: script index appears in more than one category
# ---------------------------------------------------------------------------
def _check_no_duplicate_scripts() -> None:
    seen: dict[int, str] = {}
    errors: list[str] = []
    for _letter, cat_name, idxs, _color in CATEGORIES:
        for idx in idxs:
            if idx in seen:
                label = options[idx][0] if 0 <= idx < len(options) else str(idx)
                errors.append(
                    f"  [{idx}] '{label}' in '{seen[idx]}' and '{cat_name}'"
                )
            else:
                seen[idx] = cat_name
    if errors:
        raise ValueError(
            "covid_catalogue.py — scripts listed in multiple categories:\n" + "\n".join(errors)
        )


_check_no_duplicate_scripts()


# ---------------------------------------------------------------------------
# Auto-collect scripts not assigned to any category → category Z
# ---------------------------------------------------------------------------
_all_categorised: set[int] = {
    idx
    for _letter, _name, idxs, _color in CATEGORIES
    for idx in idxs
}
_uncategorised: list[int] = [i for i in range(len(options)) if i not in _all_categorised]

if _uncategorised:
    CATEGORIES.append(
        ("Z", "❓  Uncategorised", _uncategorised, "#95A5A6")
    )


# ---------------------------------------------------------------------------
# Derived lookups — built once at import time, re-exported for convenience
# ---------------------------------------------------------------------------
# choice_index → index in CATEGORIES
CHOICE_TO_CAT_I: dict[int, int] = {}
# "B" → index in CATEGORIES
LETTER_TO_CAT_I: dict[str, int] = {}

for _cat_i, (_letter, _name, _idxs, _color) in enumerate(CATEGORIES):
    LETTER_TO_CAT_I[_letter.upper()] = _cat_i
    for _idx in _idxs:
        CHOICE_TO_CAT_I[_idx] = _cat_i


def give_options_categories() -> tuple:
    """Convenience accessor kept for backwards compatibility."""
    return options, CATEGORIES
