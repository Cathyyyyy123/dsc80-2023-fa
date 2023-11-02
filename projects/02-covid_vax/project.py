# project.py


import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

# If this import errors, run `pip install plotly` in your Terminal with your conda environment activated.
import plotly.express as px



# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def count_monotonic(arr):
    difference = np.diff(arr)
    return np.sum(difference < 0)

def monotonic_violations_by_country(vacs):    
    vacs_mono = vacs.groupby('Country_Region')[['Doses_admin', 'People_at_least_one_dose']].agg(count_monotonic)
    result = vacs_mono.rename(columns={'Doses_admin': 'Doses_admin_monotonic', 'People_at_least_one_dose': 'People_at_least_one_dose_monotonic'})
    return result


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def percentile(arr):
    return np.percentile(arr, 97)

def robust_totals(vacs):
    robust = vacs.groupby('Country_Region')[['Doses_admin', 'People_at_least_one_dose']].agg(percentile)
    return robust

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def split_percentage(x):
    return float(x.split('%')[0])


def split_area(x):
    return int(x.split('Km')[0].replace('_', '').replace(',', ''))


def split_density(x):
    return float(x.split('/')[0])


def fix_dtypes(pops_raw):
    pop = pops_raw.copy()
    wout_per = pop['World Percentage'].apply(split_percentage)
    wout_per = wout_per/100
    pop['World Percentage'] = wout_per
    pop['Population in 2023'] = (pop['Population in 2023']*1000).astype(int)
    pop['Area (Km²)'] = pop['Area (Km²)'].apply(split_area)
    pop['Density (P/Km²)'] = pop['Density (P/Km²)'].apply(split_density)
    return pop


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pops):
    tots_copy = tots.copy()
    pops_copy = pops.copy()
    index_col = tots_copy.index.name if tots_copy.index.name else 'index'
    countries = tots_copy.reset_index()[index_col]
    countries_in_tots = set(countries)
    countries_in_pops = set(pops_copy['Country (or dependency)'])
    final_countries = countries_in_tots - countries_in_pops
    return final_countries

    
def fix_names(pops):
    fixed_pops = pops.copy()
    name_mapping = {
        'Myanmar': 'Burma',
        'Cape Verde': 'Cabo Verde',
        'Republic of the Congo': 'Congo (Brazzaville)',
        'DR Congo': 'Congo (Kinshasa)',
        'Ivory Coast': "Cote d'Ivoire",
        'Czech Republic': 'Czechia',
        'South Korea': 'Korea, South',
        'United States': 'US',
        'Palestine': 'West Bank and Gaza'
    }
    fixed_pops['Country (or dependency)'] = fixed_pops['Country (or dependency)'].replace(name_mapping)
    
    return fixed_pops


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def draw_choropleth(tots, pops_fixed):
    tots_copy  = tots.copy()
    tots_copy = tots_copy.reset_index()
    pops_fixed_copy = pops_fixed.copy()
    tots_copy.rename(columns={'Country_Region': 'Country (or dependency)'}, inplace=True)
    merged = tots_copy.merge(pops_fixed_copy, on='Country (or dependency)')
    merged['Doses Per Person'] = merged['Doses_admin'] / merged['Population in 2023']
    merged
    fig = px.choropleth(merged, 
                            locations="ISO", 
                            color="Doses Per Person",
                            hover_name="Country (or dependency)",
                            title="COVID Vaccine Doses Per Person, Globally",
                            color_continuous_scale=px.colors.sequential.Reds,
                            labels={'Doses Per Person': 'Doses Per Person'}
                        )
        
    fig.update_layout(title_font_family="Arial", title_font_size=24)
    return fig


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_israel_data(df):
    data = df.copy()
    data['Age'].replace('-', np.nan, inplace=True)
    data['Age'] = data['Age'].astype(float)
    data['Vaccinated'] = data['Vaccinated'].astype(bool)
    data['Severe Sickness'] = data['Severe Sickness'].astype(bool)
    return data


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def mcar_permutation_tests(df, n_permutations=100):
    stats = []
    for _ in range(n_permutations):
        shuffled = df.copy()
        shuffled['Vaccinated'] = np.random.permutation(shuffled['Vaccinated'])
        shuffled = shuffled.assign(missing_age = shuffled['Age'].isna())
        grouped = shuffled.groupby('Vaccinated')['missing_age'].mean().diff().abs().iloc[-1]
        stats.append(grouped)
    stats = np.array(stats)
    
    stats2 = []
    for _ in range(n_permutations):
        shuffled = df.copy()
        shuffled['Severe Sickness'] = np.random.permutation(shuffled['Severe Sickness'])
        shuffled = shuffled.assign(missing_age = shuffled['Age'].isna())
        grouped = shuffled.groupby('Severe Sickness')['missing_age'].mean().diff().abs().iloc[-1]
        stats2.append(grouped)
    stats2 = np.array(stats2)

    return (stats, stats2)


def missingness_type():
    return 1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def effectiveness(df):
    vac = df[df['Vaccinated']]
    unvac = df[~df['Vaccinated']]
    pv = vac[vac['Severe Sickness']].shape[0] / vac.shape[0]
    pu = unvac[unvac['Severe Sickness']].shape[0] / unvac.shape[0]
    effect = 1-pv/pu
    return effect


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


AGE_GROUPS = [
    '12-15',
    '16-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80-89',
    '90-'
]

def stratified_effectiveness(df):
    result = []
    
    df = df.copy()
    g1 = df[(12 <= df['Age']) & (df['Age'] < 15)]
    g2 = df[(16 <= df['Age']) & (df['Age'] < 19)]
    g3 = df[(20 <= df['Age']) & (df['Age'] < 29)]
    g4 = df[(30 <= df['Age']) & (df['Age'] < 39)]
    g5 = df[(40 <= df['Age']) & (df['Age'] < 49)]
    g6 = df[(50 <= df['Age']) & (df['Age'] < 59)]
    g7 = df[(60 <= df['Age']) & (df['Age'] < 69)]
    g8 = df[(70 <= df['Age']) & (df['Age'] < 79)]
    g9 = df[(80 <= df['Age']) & (df['Age'] < 89)]
    g10 = df[(90 <= df['Age'])]

    result.append(effectiveness(g1))
    result.append(effectiveness(g2))
    result.append(effectiveness(g3))
    result.append(effectiveness(g4))
    result.append(effectiveness(g5))
    result.append(effectiveness(g6))
    result.append(effectiveness(g7))
    result.append(effectiveness(g8))
    result.append(effectiveness(g9))
    result.append(effectiveness(g10))
    return pd.Series(result, index=AGE_GROUPS)


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def effectiveness_calculator(
    *,
    young_vaccinated_prop,
    old_vaccinated_prop,
    young_risk_vaccinated,
    young_risk_unvaccinated,
    old_risk_vaccinated,
    old_risk_unvaccinated
):
    result = {}
    overall_risk_vaccinated = young_vaccinated_prop * young_risk_vaccinated + old_vaccinated_prop * old_risk_vaccinated
    overall_risk_unvaccinated = (1- young_vaccinated_prop) * young_risk_unvaccinated + (1 - old_vaccinated_prop) * old_risk_unvaccinated
    pv = overall_risk_vaccinated / (young_vaccinated_prop + old_vaccinated_prop)
    pu = overall_risk_unvaccinated / ((1 - young_vaccinated_prop) + (1 - old_vaccinated_prop))
    result['Overall'] = 1 - (pv / pu)
    result['Young'] = 1 - (young_risk_vaccinated / young_risk_unvaccinated)
    result['Old'] = 1 - (old_risk_vaccinated / old_risk_unvaccinated)
    return result


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    result = {
        'young_vaccinated_prop': 0.01,
        'old_vaccinated_prop': 0.99,
        'young_risk_vaccinated': 0.01,
        'young_risk_unvaccinated': 0.07,
        'old_risk_vaccinated': 0.09,
        'old_risk_unvaccinated': 0.5
    }

    return result
