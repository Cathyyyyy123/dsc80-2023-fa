# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_universities(df):
    df2 = df.copy()
    cleaned_instution = df2['institution'].str.replace('\n', ', ')
    df2 = df2.assign(institution=cleaned_instution)
    df2['broad_impact'] = df2['broad_impact'].astype(int)
    rank = df2['national_rank'].str.split(',', expand=True)
    replacement = {
        'USA': 'United States',
        'UK': 'United Kingdom',
        'Czechia': 'Czech Republic',
    }
    nation = rank[0].str.strip().replace(replacement)
    national_rank_cleaned = rank[1].str.strip().apply(lambda x: int(x))
    df2['nation'] = nation
    df2['national_rank_cleaned'] = national_rank_cleaned
    df2.drop(columns=['national_rank'], inplace=True)
    univ_public = df2[df2['control'].str.split(' ', expand=True)[0] == 'Public']
    univer_r1 = univ_public[(univ_public['city'] != np.nan) & (univ_public['state'] != np.nan)]
    univer_r1['institution']
    df2['is_r1_public'] = df2['institution'].isin(univer_r1['institution'])
    return df2

def university_info(cleaned):
    grouped = cleaned.groupby('state').filter(lambda x: len(x) >= 3)
    state_lowest_mean = grouped.groupby('state')['score'].mean().idxmin()
    top_100 = cleaned[cleaned['world_rank'] <= 100]
    top_fac = top_100[top_100['quality_of_faculty'] <= 100]
    proportion = len(top_fac) / len(top_100)
    pivot = cleaned.pivot_table(
        index='state',
        values='is_r1_public',
        aggfunc='mean',
    )
    states__50_percent_private = (1 - pivot['is_r1_public']) >= 0.5
    num_states_private = int(states__50_percent_private.sum())
    nationaltop = cleaned[cleaned['national_rank_cleaned'] == 1]
    univer = nationaltop[['world_rank', 'institution']].sort_values(by='world_rank', ascending=False).iloc[0]['institution']
    return [state_lowest_mean, proportion, num_states_private, univer]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

def normalize_score(row, mean_scores, std_scores):
        nation = row['nation']
        score = row['score']
        mean = mean_scores.loc[nation]
        std = std_scores.loc[nation]
        if pd.isna(score) or std_scores[nation] == 0:
            return np.NaN
        return (score - mean) / std

def std_scores_by_nation(cleaned):
    cleaned_copy = cleaned[['institution', 'nation', 'score']].copy()
    mean_scores = cleaned_copy.groupby('nation')['score'].mean()
    std_scores = cleaned_copy.groupby('nation')['score'].std(ddof=0)
    cleaned_copy['score'] = cleaned_copy.apply(normalize_score, axis=1, args=(mean_scores, std_scores))
    return cleaned_copy
    

def su_and_spread():
    return [2, 'United States']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    dirname = Path(dirname)
    list_survey = dirname.iterdir()
    dfs = []
    for item in list_survey:
        if item.suffix == '.csv':
            survey_case = pd.read_csv(item)
            survey_case.columns = survey_case.columns.str.lower()
            if survey_case.columns.str.contains('_').any():
                survey_case.columns = survey_case.columns.str.replace('_', ' ')
            survey_case = survey_case[['first name', 'last name', 'current company', 'job title', 'email', 'university']]
            dfs.append(survey_case)
    df = pd.concat(dfs, ignore_index=True)
    return df

def com_stats(df):
    ohio_nurse = df[(df['university'].str.contains('Ohio')) & (df['job title'].str.contains('Nurse'))]
    prop = len(ohio_nurse)/df['university'].str.contains('Ohio').sum()

    df['job title'] = df['job title'].fillna('')
    num_job = df[df['job title'].str.endswith('Engineer')]['job title'].nunique()

    name_index = df['job title'].str.len().idxmax()
    longest_name = df.loc[name_index, 'job title']

    num_manger = df['job title'].str.lower().str.contains('manager').sum()
    return [prop, num_job, longest_name, num_manger]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    survey_all = Path(dirname).iterdir()
    dfs = []
    for item in survey_all:
        survey_case = pd.read_csv(item)
        dfs.append(survey_case)
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.merge(dfs[i], how='left', on='id')
    df.set_index('id', inplace=True)
    return df


def check_credit(df):
    df_copy = df.drop(columns=['name']).copy()
    for colunms in df_copy.columns:
        df_copy.loc[pd.notna(df[colunms]), colunms] = 1
    df_copy.fillna(0, inplace=True)
    num_ques = df_copy.sum(axis=1)
    prop = df_copy.sum()/df_copy.shape[0]
    df_copy.loc[num_ques >= df_copy.shape[1] / 2, 'ec'] = 5
    count = 0
    df_copy.fillna(0, inplace=True)
    for value in prop:
        if value >= 0.8:
            count += 1
    if count == 1:
        df_copy['ec'] += 1
    elif count >= 2:
        df_copy['ec'] += 2
    else:
        df_copy['ec'] += 0
    df_copy = df_copy[['ec']]
    return df_copy.merge(df, on='id')[['name', 'ec']]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    return pets.merge(procedure_history, on='PetID', how='left').groupby('ProcedureType')['PetID'].count().idxmax()

def pet_name_by_owner(owners, pets):
    owners_copy = owners.rename(columns={'Name': 'First name'})
    pets_copy = pets.rename(columns={'Name': 'Pet name'})
    result = owners_copy[['OwnerID', 'First name']].merge(pets_copy[['OwnerID', 'Pet name']], on='OwnerID', how='left').groupby('OwnerID')['Pet name'].agg(list)
    pet_list = result.apply(lambda x: x[0] if len(x) == 1 else x)
    pet_list = pd.DataFrame(pet_list)
    final_result = pet_list.merge(owners_copy[['OwnerID', 'First name']], on='OwnerID', how='left')
    final_result.set_index('First name', inplace=True)
    return final_result.drop(columns=['OwnerID'])['Pet name']


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    owner_pet = owners[['OwnerID', 'City']].merge(pets[['OwnerID', 'PetID']], on='OwnerID', how='left')
    owner_pet_code = owner_pet.merge(procedure_history[['PetID', 'ProcedureSubCode', 'ProcedureType']], on='PetID', how='inner')
    proce_price = owner_pet_code.merge(procedure_detail[['ProcedureSubCode', 'Price', 'ProcedureType']], on=['ProcedureSubCode', 'ProcedureType'], how='left')
    return proce_price[['City', 'Price']].groupby('City')['Price'].sum()


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def average_seller(sales):
    average = sales.pivot_table(
        index='Name',
        values='Total',
        aggfunc='mean',
    )
    average.rename(columns={'Total': 'Average Sales'}, inplace=True)
    return average

def product_name(sales):
    result = sales.pivot_table(
        index='Name', 
        columns='Product', 
        values='Total', 
        aggfunc='sum'
    )
    return result

def count_product(sales):
    result = sales.pivot_table(
        index=['Product', 'Name'], 
        columns='Date', 
        values='Total', 
        aggfunc='count'
    )
    result.fillna(0, inplace=True)
    return result

def total_by_month(sales):
    sales['month'] = pd.to_datetime(sales['Date']).dt.strftime('%B')
    result = sales.pivot_table(
        index=['Name', 'Product'],
        columns='month',
        values='Total',
        aggfunc='sum'
    )
    result.fillna(0, inplace=True)
    return result
