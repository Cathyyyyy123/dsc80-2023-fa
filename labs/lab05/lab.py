# lab.py


import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'MAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'NMAR', 'MAR', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def first_round():
    return [0.146, 'NR']


def second_round():
    return [0.039, 'R', 'D']


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    heights_copy = heights.copy()
    heights_copy
    pvalues = []
    columns = heights.filter(like='child_').columns
    for col in columns:
        heights_copy[f'{col}_missing'] = heights_copy[col].isna()
        pvalues.append(ks_2samp(heights_copy.query(f'{col}_missing')['father'], heights_copy.query('not 'f'{col}_missing')['father']).pvalue)
    return pd.Series(pvalues, index=columns)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def mean_impute(ser):
    return ser.fillna(ser.mean())


def cond_single_imputation(new_heights):
    new_heights['father_quartile'] = pd.qcut(new_heights['father'], 4, labels=False)
    heights_mar_cond = new_heights.groupby('father_quartile')['child'].transform(mean_impute).to_frame()
    return heights_mar_cond['child']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    child = child.dropna()
    hist_prob, bin_edges = np.histogram(child, bins=10, density=True)
    bin_widths = np.diff(bin_edges)
    hist_prob *= bin_widths
    cdf_bins = np.cumsum(hist_prob)
    imputed_values = np.zeros(N)
    for i in range(N):
        random_num = np.random.rand()
        bin_index = np.searchsorted(cdf_bins, random_num)
        if bin_index == len(cdf_bins):
            bin_index -= 1
        random_value_in_bin = np.random.uniform(bin_edges[bin_index], bin_edges[bin_index + 1])
        imputed_values[i] = random_value_in_bin
    return imputed_values


def impute_height_quant(child):
    values = quantitative_distribution(child, child.isna().sum())
    child.loc[child.isnull()] = values
    return child


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    list1 = [1, 2, 2, 1]
    list2 = ['https://hackertyper.com/robots.txt', 'https://libgen.com/robots.txt']
    return list1, list2
