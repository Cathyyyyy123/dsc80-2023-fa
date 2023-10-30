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


def fix_dtypes(pops_raw):
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pops):
    ...

    
def fix_names(pops):
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def draw_choropleth(tots, pops_fixed):
    ...


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
    ...
    
    
def missingness_type():
    ...


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
    ...


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
    overall_risk_unvaccinated = young_vaccinated_prop * young_risk_unvaccinated + old_vaccinated_prop * old_risk_unvaccinated
    result['Overall'] = 1 - (overall_risk_vaccinated / overall_risk_unvaccinated)
    result['Young'] = 1 - (young_risk_vaccinated / young_risk_unvaccinated)
    result['Old'] = 1 - (old_risk_vaccinated / old_risk_unvaccinated)
    return result


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    ...
