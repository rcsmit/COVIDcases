# 🦠 René Smit — COVID Scripts

> A collection of 63+ interactive Python/Streamlit apps covering epidemic modelling, vaccine effectiveness, excess mortality, Dutch open data, and more.

🔗 **Live app:** [rcsmit-covidcases.streamlit.app](https://rcsmit-covidcases.streamlit.app)


### 🦠 Epidemic models
| # | Script | Description |
|---|--------|-------------|
| 7 | [SEIR hobbeland](https://rcsmit-covidcases.streamlit.app/?choice=7) | Classic SEIR epidemic model (hobbeland variant) |
| 8 | [Show contactmatrix](https://rcsmit-covidcases.streamlit.app/?choice=8) | Age-structured contact matrix visualizer |
| 10 | [Cases from susceptibles](https://rcsmit-covidcases.streamlit.app/?choice=10) | Case trajectory from susceptible pool dynamics |
| 18 | [SIR model met leeftijdsgroepen](https://rcsmit-covidcases.streamlit.app/?choice=18) | Age-structured SIR epidemic model |
| 34 | [Playing with R0](https://rcsmit-covidcases.streamlit.app/?choice=34) | Interactive R₀ sensitivity explorer |
| 58 | [SIR model agent based](https://rcsmit-covidcases.streamlit.app/?choice=58) | Vectorized agent-based SIR epidemic model |
| 61 | [RIVM model](https://rcsmit-covidcases.streamlit.app/?choice=61) | Replicated RIVM epidemic model |

### 📊 Cases & R number
| # | Script | Description |
|---|--------|-------------|
| 1 | [Covid dashboard](https://rcsmit-covidcases.streamlit.app/?choice=1) | Main COVID dashboard for the Netherlands |
| 2 | [Plot hosp/IC per age](https://rcsmit-covidcases.streamlit.app/?choice=2) | Hospital and IC admissions per age group |
| 4 | [Number of cases interactive](https://rcsmit-covidcases.streamlit.app/?choice=4) | Interactive case count visualization |
| 9 | [R getal per provincie](https://rcsmit-covidcases.streamlit.app/?choice=9) | Reproduction number R per Dutch province |
| 12 | [Calculate R per country OWID](https://rcsmit-covidcases.streamlit.app/?choice=12) | R number per country using OWID data |
| 13 | [Covid dashboard OWID/Google](https://rcsmit-covidcases.streamlit.app/?choice=13) | COVID dashboard with OWID, Google, and Waze mobility data |
| 14 | [Dag verschillen per leeftijd](https://rcsmit-covidcases.streamlit.app/?choice=14) | Daily case differences per age group |
| 16 | [R getal per leeftijdscategorie](https://rcsmit-covidcases.streamlit.app/?choice=16) | Reproduction number R per age category |
| 19 | [Pos testen per leeftijdscat.](https://rcsmit-covidcases.streamlit.app/?choice=19) | Positive test rate per age category |
| 20 | [Per provincie per leeftijd](https://rcsmit-covidcases.streamlit.app/?choice=20) | Cases per province per age group |

**[1] Covid dashboard**
![covid dashboard](https://user-images.githubusercontent.com/1609141/112730553-8b1cf680-8f32-11eb-83f6-1569f5114678.png)

**[2] Plot hosp/IC per age**
![plot hosp IC](https://user-images.githubusercontent.com/1609141/118802804-e02a1880-b8a2-11eb-8772-cc495bf7bca8.png)

**[4] Number of cases interactive**
![number of cases](https://user-images.githubusercontent.com/1609141/112731094-945b9280-8f35-11eb-8c3d-a99e5f48487d.png)

**[19] Pos testen per leeftijdscat.**
![pos testen per leeftijdscategorie](https://user-images.githubusercontent.com/1609141/112730260-e0f09f00-8f30-11eb-9bff-a835c2f965f7.png)

### 🧪 Testing & Serology
| # | Script | Description |
|---|--------|-------------|
| 3 | [False positive rate covid test](https://rcsmit-covidcases.streamlit.app/?choice=3) | False positive rate calculator for COVID tests |
| 5 | [IFR from prevalence](https://rcsmit-covidcases.streamlit.app/?choice=5) | Infection fatality rate derived from seroprevalence data |
| 15 | [Abs./rel. humidity from RH](https://rcsmit-covidcases.streamlit.app/?choice=15) | Calculate specific/absolute humidity from relative humidity |
| 21 | [Kans om COVID op te lopen](https://rcsmit-covidcases.streamlit.app/?choice=21) | Probability of contracting COVID |
| 31 | [Aerosol concentration in room](https://rcsmit-covidcases.streamlit.app/?choice=31) | Aerosol concentration in room by @hk_nien |
| 35 | [Calculate Se & Sp Rapidtest](https://rcsmit-covidcases.streamlit.app/?choice=35) | Sensitivity and specificity calculator for rapid tests |

### 💉 Vaccines & Effectiveness
| # | Script | Description |
|---|--------|-------------|
| 22 | [Data per gemeente](https://rcsmit-covidcases.streamlit.app/?choice=22) | Vaccination, income, and cases per municipality |
| 23 | [VE Israel](https://rcsmit-covidcases.streamlit.app/?choice=23) | Vaccine effectiveness — Israel (Zijlstra method) |
| 24 | [Hosp/death NL](https://rcsmit-covidcases.streamlit.app/?choice=24) | Dutch hospitalizations and deaths over time |
| 25 | [VE Nederland](https://rcsmit-covidcases.streamlit.app/?choice=25) | Vaccine effectiveness Netherlands |
| 27 | [VE & CI calculations](https://rcsmit-covidcases.streamlit.app/?choice=27) | Vaccine effectiveness and confidence interval calculations |
| 28 | [VE scenario calculator](https://rcsmit-covidcases.streamlit.app/?choice=28) | Vaccine effectiveness under different scenario assumptions |
| 29 | [VE vs inv. odds](https://rcsmit-covidcases.streamlit.app/?choice=29) | Vaccine effectiveness vs inverse odds analysis |
| 47 | [Deltavax](https://rcsmit-covidcases.streamlit.app/?choice=47) | Delta variant and vaccination interaction model |
| 54 | [Herhaalprik](https://rcsmit-covidcases.streamlit.app/?choice=54) | Booster vaccination uptake and effect analysis |
| 56 | [Bayes Mortality Vaccination](https://rcsmit-covidcases.streamlit.app/?choice=56) | Bayesian analysis of mortality and vaccination |

### 📈 Curve fitting
| # | Script | Description |
|---|--------|-------------|
| 6 | [Fit to data](https://rcsmit-covidcases.streamlit.app/?choice=6) | Curve fitting to COVID case data |
| 11 | [Fit to data OWID animated](https://rcsmit-covidcases.streamlit.app/?choice=11) | Animated curve fitting to Our World in Data |
| 30 | [Fit to data Levitt](https://rcsmit-covidcases.streamlit.app/?choice=30) | Animated Levitt-style curve fitting (OWID) |

### ⚰️ Mortality & Excess deaths
| # | Script | Description |
|---|--------|-------------|
| 36 | [Oversterfte gemeente](https://rcsmit-covidcases.streamlit.app/?choice=36) | Excess mortality per Dutch municipality |
| 37 | [Sterfte patronen](https://rcsmit-covidcases.streamlit.app/?choice=37) | Dutch mortality patterns 2000–2024 |
| 38 | [Bayes Lines tools](https://rcsmit-covidcases.streamlit.app/?choice=38) | Bayesian line-fitting tools for mortality data |
| 39 | [Oversterfte (CBS Odata)](https://rcsmit-covidcases.streamlit.app/?choice=39) | Excess mortality using CBS open data API |
| 41 | [Disabled by Long covid](https://rcsmit-covidcases.streamlit.app/?choice=41) | Disability burden estimates from Long COVID |
| 42 | [Oversterfte 5yr Eurostats](https://rcsmit-covidcases.streamlit.app/?choice=42) | Monthly excess mortality — 5-year baseline, Eurostat |
| 43 | [Doodsoorzaken Sankey](https://rcsmit-covidcases.streamlit.app/?choice=43) | Sankey diagram of causes of death |
| 45 | [Rioolwaarde vs overleden CBS](https://rcsmit-covidcases.streamlit.app/?choice=45) | Sewage signal vs deaths (CBS data) |
| 46 | [Mortality yearly per capita](https://rcsmit-covidcases.streamlit.app/?choice=46) | Annual mortality per capita trend analysis |
| 48 | [Verwachte sterfte](https://rcsmit-covidcases.streamlit.app/?choice=48) | Expected mortality baseline calculation |
| 50 | [Calculate baselines (Poisson)](https://rcsmit-covidcases.streamlit.app/?choice=50) | Poisson-based mortality baseline calculator |
| 51 | [AG table mortality](https://rcsmit-covidcases.streamlit.app/?choice=51) | AG actuarial table mortality analysis |
| 52 | [Find baseline length](https://rcsmit-covidcases.streamlit.app/?choice=52) | Sensitivity analysis on baseline window length |
| 53 | [Mortality/week/100k](https://rcsmit-covidcases.streamlit.app/?choice=53) | Weekly mortality per age per 100,000 population |
| 55 | [Fit Mortality/causes death](https://rcsmit-covidcases.streamlit.app/?choice=55) | Curve fitting on cause-of-death mortality data |
| 57 | [Sterfte/rioolw./vaccins](https://rcsmit-covidcases.streamlit.app/?choice=57) | Correlation: mortality, sewage signal, vaccinations |
| 60 | [Oversterfte predict per levensjaar](https://rcsmit-covidcases.streamlit.app/?choice=60) | Excess mortality prediction per year of life |
| 62 | [Oversterfte GAM](https://rcsmit-covidcases.streamlit.app/?choice=62) | GAM-based excess mortality prediction |
| 63 | [CBS-Oversterfte](https://rcsmit-covidcases.streamlit.app/?choice=63) | Excess mortality analysis with CBS data — back to the drawing board |

### 🚿 Sewage water
| # | Script | Description |
|---|--------|-------------|
| 17 | [Show rioolwaardes](https://rcsmit-covidcases.streamlit.app/?choice=17) | Sewage water COVID signal visualization |
| 44 | [Rioolwaarde vs ziekenhuis](https://rcsmit-covidcases.streamlit.app/?choice=44) | Sewage signal vs hospital admissions |
| 59 | [Rioolwater vs covidsterfte](https://rcsmit-covidcases.streamlit.app/?choice=59) | Sewage COVID signal vs COVID deaths |

### 🌍 International comparisons
| # | Script | Description |
|---|--------|-------------|
| 26 | [Scatterplots QoG OWID](https://rcsmit-covidcases.streamlit.app/?choice=26) | Quality of Government vs OWID COVID scatterplots |
| 32 | [Compare two variants](https://rcsmit-covidcases.streamlit.app/?choice=32) | Side-by-side comparison of two COVID variants |
| 33 | [Scatterplot OWID](https://rcsmit-covidcases.streamlit.app/?choice=33) | OWID country-level scatterplot explorer |

### 📐 Statistics & Methods
| # | Script | Description |
|---|--------|-------------|
| 40 | [Bayes berekeningen IC/ziekenhuis](https://rcsmit-covidcases.streamlit.app/?choice=40) | Bayesian probability calculations for IC and hospital data |
| 49 | [Logistic regression](https://rcsmit-covidcases.streamlit.app/?choice=49) | Logistic regression on COVID outcome data |

---

## 🚀 Running locally

```bash
git clone https://github.com/rcsmit/COVIDcases.git
cd COVIDcases
pip install -r requirements.txt
streamlit run covid_menu_streamlit.py
```

### Deep-link to a specific script
```
?choice=39          # opens script #39 (Oversterfte CBS Odata)
?cat=G              # opens the Mortality & Excess deaths category
?choice=39&cat=G    # both in sync
```

---

## 🛠️ Tech stack

- **Python** · Streamlit · Plotly · pandas · NumPy
- **Data sources:** CBS Odata, RIVM, Eurostat, Our World in Data, OWID
- **Statistics:** SciPy, statsmodels, PyMC / Bayesian methods
- **Simulation:** SEIR/SIR models, Monte Carlo, agent-based models

---

## 👤 About

I'm René Smit — a Dutch multidisciplinary freelancer with a background in data analysis and a long-standing interest in epidemiology, public health data, and critical appraisal of scientific claims.
These scripts were built during and after the COVID-19 pandemic as a way to understand and visualize what was happening.

🌐 [rene-smit.com](https://rene-smit.com) · 📊 [rcsmit.streamlit.app](https://rcsmit.streamlit.app) · 🐙 [github.com/rcsmit](https://github.com/rcsmit)

---

*MIT License · Contributions and feedback welcome via Issues or PRs*
