# lab.py


import pandas as pd
import numpy as np
import os
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    log_transformer = FunctionTransformer(np.log, validate=True)
    pl = Pipeline(steps=[
        ('log_scaler', log_transformer),
        ('regressor', LinearRegression())
    ])
    pl.fit(data[['c2']], data['y'])
    return (pl, pl.predict(data[['c2']]))


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multi_type_pipeline(data):
    log_transformer = FunctionTransformer(np.log, validate=True)
    preproc = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['c1']),
            ('c2_log', log_transformer, ['c2']),
            ('ohe_group', OneHotEncoder(), ['group'])
        ],
        remainder='passthrough'
    )
    pl = Pipeline([
        ('preprocessor', preproc), 
        ('lin-reg', LinearRegression())
    ])
    pl.fit(data.drop('y', axis=1), data['y'])
    return (pl, pl.predict(data.drop('y', axis=1)))


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


# Imports
from sklearn.base import BaseEstimator, TransformerMixin

class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # X might not be a pandas DataFrame (e.g. a np.array)
        df = pd.DataFrame(X)

        # Compute and store the means/standard-deviations for each column (e.g. 'c1' and 'c2'),
        # for each group (e.g. 'A', 'B', 'C').
        # (Our solution uses a dictionary)
        self.grps_ = df.groupby(df.columns[0]).agg(['mean', 'std']).to_dict()

        return self

    def transform(self, X, y=None):

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Hint: Define a helper function here!
        def standardize(row, col):
            group = row[df.columns[0]]
            value = row[col]
            mean, std = self.grps_[(col, 'mean')][group], self.grps_[(col, 'std')][group]
            if std == 0:
                return 0
            else:
                return (value - mean) / std
        
        df = pd.DataFrame(X)
        for col in df.columns[1:]:
            df[col] = df.apply(standardize, axis=1, args=(col,))
        df.drop(df.columns[0], axis=1, inplace=True)
        return df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def eval_toy_model():
    return [(2.7551086974518104, 0.39558507345910776), (2.3148336164355263, 0.5733249315673331), (2.3157339477823844, 0.5729929650348398)]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tree_reg_perf(galton):
    # Add your imports here
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    X = galton.drop('childHeight', axis=1)
    y = galton['childHeight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    errors = {'train_err': [], 'test_err': []}
    for depth in range(1, 21):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=1)
        # Train the model
        tree.fit(X_train, y_train)
        # Predict and calculate the RMSE for training set
        train_pred = tree.predict(X_train)
        train_err = np.sqrt(mean_squared_error(y_train, train_pred))
        errors['train_err'].append(train_err)
        # Predict and calculate the RMSE for test set
        test_pred = tree.predict(X_test)
        test_err = np.sqrt(mean_squared_error(y_test, test_pred))
        errors['test_err'].append(test_err)
    results_df = pd.DataFrame(errors, index=range(1, 21))
    return results_df


def knn_reg_perf(galton):
    # Add your imports here
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    X = galton.drop('childHeight', axis=1)
    y = galton['childHeight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    errors = {'train_err': [], 'test_err': []}
    # Loop over tree depths from 1 to 20
    for k in range(1, 21):
        knn = KNeighborsRegressor(n_neighbors=k)
        # Train the model
        knn.fit(X_train, y_train)

        # Predict and calculate the RMSE for training set
        train_pred = knn.predict(X_train)
        train_err = np.sqrt(mean_squared_error(y_train, train_pred))
        errors['train_err'].append(train_err)

        # Predict and calculate the RMSE for test set
        test_pred = knn.predict(X_test)
        test_err = np.sqrt(mean_squared_error(y_test, test_pred))
        errors['test_err'].append(test_err)

    # Create a DataFrame from the errors dictionary
    results_df = pd.DataFrame(errors, index=range(1, 21))
    return results_df


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------

def titanic_model(titanic):
    # Add your import(s) here
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    import re

    def extract_title(name):
        return re.search(r'([A-Za-z]+)\.', name).group(1)

    titanic['Title'] = titanic['Name'].apply(extract_title)

    preprocessor = ColumnTransformer(
        transformers=[
            ('pclass', OneHotEncoder(), ['Pclass']),
            ('AgeStdbytitle', StdScalerByGroup(), ['Pclass', 'Age']),
            ('sex', OneHotEncoder(), ['Sex']),
            ('FareStdbytitle', StdScalerByGroup(), ['Pclass', 'Fare'])
        ],
        remainder='drop'
    )

    pl = Pipeline([
        ('preprocessing', preprocessor),
        ('classification', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    pl.fit(titanic.drop('Survived', axis=1), titanic['Survived'])
    return pl