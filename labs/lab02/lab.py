# lab.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def data_load(scores_fp):
    scores = pd.read_csv(scores_fp,
                         usecols=['name', 'tries', 'highest_score', 'sex'],
                         )
    scores.drop(columns=['sex'], inplace=True)
    scores.rename(columns={'name': 'firstname', 'tries': 'attempts'}, inplace=True)
    scores.set_index('firstname', inplace=True)
    return scores 


def check_pass(scores):
    if scores['attempts'] > 1 and scores['highest_score'] < 60:
        return 'No'
    if scores['attempts'] > 4 and scores['highest_score'] < 70:
            return 'No'
    if scores['attempts'] > 6 and scores['highest_score'] < 90:
        return 'No'
    if scores['attempts'] > 8:
        return 'No'
    return 'Yes'


def pass_fail(scores):
    scores['pass'] = scores.apply(check_pass, axis=1)
    return scores


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def med_score(scores):
    return scores[scores['pass'] == 'Yes']['highest_score'].median()

def highest_score_name(scores):
    highest_score = scores.sort_values('highest_score', ascending=False).iloc[0]['highest_score']
    name = scores[scores['highest_score'] == highest_score].index
    return (highest_score, name)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def idx_dup():
    return 6


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def trick_me():
    return 3


def trick_bool():
    return [4, 10, 13]



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def change(x):
    if np.isnan(x):
        return 'MISSING'
    else:
        return x
    
def correct_replacement(df_with_nans):
    df_with_nans_copy = df_with_nans.copy()
    return df_with_nans_copy.applymap(change)
    
def missing_ser():
    return 2
    
def fill_ser(df_with_nans):
    for i in df_with_nans.columns:
        df_with_nans.loc[df_with_nans[i].isna(), i] = 'MISSING'
    return


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def population_stats(df):
    num_nonnull = df.count()
    prop_nonnull = num_nonnull / len(df)
    num_distinct = df.nunique()
    prop_distinct = num_distinct / num_nonnull
    stats_df = pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })
    return stats_df


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def most_common(df, N=10):
    result_df = pd.DataFrame(index=range(N))
    for i in df.columns:
        common = df[i].value_counts().head(N)
        if len(common.index) == N:
            result_df[i + '_values'] = common.index
            result_df[i + '_counts'] = common.values
        else:
            result_df[i + '_values'].loc[:len(common.index)] = common.index
            result_df[i + '_counts'].loc[:len(common.index)] = common.values
            result_df[i + '_values'].iloc[len(common.index):] = np.NaN
            result_df[i + '_counts'].iloc[len(common.index):] = np.NaN
    return result_df


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    num_power = powers.sum(axis=1)
    name = powers.assign(new_num = num_power).set_index('hero_names')['new_num'].idxmax()
    
    fly = powers[powers['Flight'] == True]
    result = fly.drop(columns=['Flight']).set_index('hero_names')
    second_common = result.sum().idxmax()
    
    num_power = powers.sum(axis=1)
    powers.assign(num_power = num_power)
    powers_copy = powers.assign(num_power = num_power)[powers.assign(num_power = num_power)['num_power'] == 1]
    common_power = powers_copy.set_index('hero_names').drop(columns=['num_power']).sum().idxmax()
    return [name, second_common,common_power]


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace('-', np.NaN)


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def super_hero_stats():
    return ['Onslaught', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']
