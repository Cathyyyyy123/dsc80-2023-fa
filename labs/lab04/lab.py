# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    date_format = '%Y-%m-%d'
    clean_login = login.assign(Time = pd.to_datetime(login['Time'], format=date_format))
    clean_login['Time'].dt.hour
    clean_login['Hour'] = clean_login['Time'].dt.hour
    clean_login.loc[(clean_login['Hour'] >= 16) & (clean_login['Hour'] < 20), 'Count'] = 1
    return clean_login.groupby('Login Id')['Count'].sum().reset_index().set_index('Login Id').rename(columns={'Count': 'Time'})


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    date_format = '%Y-%m-%d'
    today = pd.to_datetime('2023-1-31')
    clean_login = login.assign(Time = pd.to_datetime(login['Time'], format=date_format))
    clean_login['Time']
    membership_days = clean_login.groupby('Login Id')['Time'].min().apply(lambda x: (today - x).days+1)
    num_login = clean_login.groupby('Login Id').size()
    return num_login/membership_days


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def suits_null_hyp():
    return [1, 2]

def simulate_suits_null():
    data = [0, 1]
    probabilities = [0.98, 0.02]
    sample = np.random.choice(data, size=250, p=probabilities)
    return sample

def estimate_suits_p_val(N):
    results = []
    for i in range(N):
            sample = simulate_suits_null()
            sample_mean = sample.mean()
            results.append(sample_mean)
    observed = 10/250
    p_val = np.mean([1 if sample_mean >= observed else 0 for sample_mean in results])
    return p_val



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypoth():
    return [1, 4]

def car_alt_hypoth():
    return [2, 6]

def car_test_stat():
    return [1, 2, 4]

def car_p_value():
    return 5


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    return heroes['Eye color'].str.lower().str.contains('blue') & heroes['Hair color'].str.lower().str.contains('blond')

def superheroes_observed_stat(heroes):
    num_bb = bhbe_col(heroes)
    num_bb_good = (heroes[num_bb]['Alignment'] == 'good').sum()
    prop_bb_good = num_bb_good/num_bb.sum()
    return prop_bb_good

def simulate_bhbe_null(n):
    new_array = np.random.multinomial(100, [0.68, 0.32], size=n)[:, 0]/100
    return new_array

def superheroes_calc_pval():
    return [0.0, 'Reject']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    return abs(data.groupby('Factory')[col].mean()[0]-data.groupby('Factory')[col].mean()[1])


def simulate_null(data, col='orange'):
    new_shuffled = data.assign(Shuffled_factory=np.random.permutation(data['Factory']))
    return abs(new_shuffled.groupby('Shuffled_factory')[col].mean()[0]-new_shuffled.groupby('Shuffled_factory')[col].mean()[1])

def pval_color(data, col='orange'):
    n_repetitions = 1000
    differences = []
    for _ in range(n_repetitions):
        diff = simulate_null(data, col)
        differences.append(diff)
    return np.mean(differences == diff_of_means(data, col))


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [('yellow', 0.0),
 ('orange', 0.004),
 ('red', 0.006),
 ('green', 0.01),
 ('purple', 0.013)]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------



def same_color_distribution():
    return (0.008, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
