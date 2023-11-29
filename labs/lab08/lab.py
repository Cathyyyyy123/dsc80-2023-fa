# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer
import itertools

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def quality_cut(str_in):
    if str_in == 'Fair':
        return 0
    elif str_in == 'Good':
        return 1
    elif str_in == 'Very Good':
        return 2
    elif str_in == 'Premium':
        return 3
    else:
        return 4
  

def color(str_in):
    if str_in == 'J':
        return 0
    elif str_in == 'I':
        return 1
    elif str_in == 'H':
        return 2
    elif str_in == 'G':
        return 3
    elif str_in == 'F':
        return 4
    elif str_in == 'E':
        return 5
    else:
        return 6
  

def clarity(str_in):
    if str_in == 'I1':
        return 0
    elif str_in == 'SI2':
        return 1
    elif str_in == 'SI1':
        return 2
    elif str_in == 'VS2':
        return 3
    elif str_in == 'VS1':
        return 4
    elif str_in == 'VVS2':
        return 5
    elif str_in == 'VVS1':
        return 6
    else:
        return 7
  

def create_ordinal(df):
    diamonds_copy = df[['cut', 'color', 'clarity']]
    diamonds_copy['ordinal_cut'] = diamonds_copy['cut'].apply(quality_cut)
    diamonds_copy['ordinal_color'] = diamonds_copy['color'].apply(color)
    diamonds_copy['ordinal_clarity'] = diamonds_copy['clarity'].apply(clarity)
    diamonds_copy = diamonds_copy[['ordinal_cut', 'ordinal_color', 'ordinal_clarity']]
    return diamonds_copy


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def nominal_col(dfout, df, col):
    for val in df[col].unique():
        dfout[f'one_hot_{col}_{val}'] = (df[col] == val).astype(int)
    return dfout


def create_one_hot(df):
    new_df = pd.DataFrame()
    for col in ['cut', 'color', 'clarity']:
        new_df = nominal_col(new_df, df, col)
    return new_df


def proportion_col(dfin, df, col):
    counts = df[col].value_counts(normalize=True)
    dfin[f'proportion_' + col] = df[col].map(counts)
    return dfin


def create_proportions(df):
    result = pd.DataFrame()
    for col in ['cut', 'color', 'clarity']:
        result = proportion_col(result, df, col)
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    quantitative_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    quantitative_columns.remove('price')
    pairs = list(itertools.combinations(quantitative_columns, 2))
    result = pd.DataFrame()
    for (col1, col2) in pairs:
        result[f'{col1} * {col2}'] = df[col1] * df[col2]
    return result


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def comparing_performance():
    # create a model per variable => (variable, R^2, RMSE) table
    return [0.8493305264354858, 1548.5331930613002, 'x', 'carat * x', 'ordinal_color', 1434.8400089047382]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        binary = Binarizer(threshold=1)
        return binary.transform(data[['carat']])
    
    # Question 6.2
    def transform_to_quantile(self, data):
        quantTrans = QuantileTransformer(n_quantiles=100)
        quantTrans.fit(self.data[['carat']])
        return quantTrans.transform(data[['carat']])
    
    # Question 6.3
    def transform_to_depth_pct(self, data):
        def depth_pct(array):
            return 100 * 2 * array[:, 2] / (array[:, 0] + array[:, 1])
        depth = FunctionTransformer(depth_pct)
        return depth.transform(np.array(data[['x', 'y', 'z']]))
