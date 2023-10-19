# lab.py


import os
import io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    nums_sorted = sorted(nums)
    if len(nums_sorted) % 2 == 0:
        median = (nums_sorted[len(nums_sorted) // 2] + nums_sorted[(len(nums_sorted) // 2) - 1]) / 2
    else:
        median = nums_sorted[len(nums) // 2]
    mean = sum(nums) / len(nums)
    return median <= mean


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def same_diff_ints(ints):
    if len(ints) == 0:
        return False
    for i in range(len(ints)):
        for j in range(i + 1, len(ints)):
            if abs(ints[i]-ints[j]) == j-i:
                return True
    return False


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    result = ''
    for i in range(n):
        var = s[:i+1]
        result = var + result
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    result = []
    largest = max(ints) + n
    length = len(str(largest))
    for i in range(len(ints)):
        exploded_str = ''
        result_str = ''
        for j in range(n):
            exploded_str = str(ints[i] - (j+1)).zfill(length)
            result_str = exploded_str + ' ' + result_str
        for k in range(n + 1):
            exploded_str = str(ints[i] + (k)).zfill(length)
            result_str = result_str + exploded_str + ' '
        result.append(result_str.strip())
    return result


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def last_chars(fh):
    lis = []
    s = ''
    for line in fh:
        lis = line[:-1]
        s = s + lis[-1]
    return s


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def add_root(A):
    position = np.arange(A.shape[0])
    A = A + position ** 0.5
    print(A)
    return A

def where_square(A):
    sqrt_vals = np.round(np.sqrt(A))
    return A == np.square(sqrt_vals)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    rates = (A[1:] - A[:-1]) / A[:-1]
    return np.round(rates, 2)

def with_leftover(A):
    B = 20 % A
    leftover = np.cumsum(B)
    day = np.where(leftover >= A)[0]
    if len(day) == 0:
        return -1
    else:
        return day[0]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    num_players = salary.shape[0]
    num_teams = salary['Team'].unique().shape[0]
    total_salary = salary['Salary'].sum()
    highest_salary = salary.loc[salary['Salary'].idxmax(), 'Player']
    avg_los = salary[salary['Team']=='Los Angeles Lakers']['Salary'].mean().round(2)
    fifth_lowest_player = salary.nsmallest(5, 'Salary').iloc[-1]
    fifth_lowest = f"{fifth_lowest_player['Player']}, {fifth_lowest_player['Team']}"
    salary = salary.assign(Last_Name = salary['Player'].str.split().str[-1].str.replace(r'\W', ''))
    duplicates = salary['Last_Name'].duplicated().any()
    team_of_highest_paid = salary.loc[salary['Salary'].idxmax(), 'Team']
    total_highest = salary[salary['Salary']== team_of_highest_paid]['Salary'].sum()
    result = pd.Series({
        'num_players': num_players,
        'num_teams': num_teams,
        'total_salary': total_salary,
        'highest_salary': highest_salary,
        'avg_lal': avg_los,
        'fifth_lowest': fifth_lowest,
        'duplicates': duplicates,
        'total_highest': total_highest
    })
    return result


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')
        parsed_data = []
        for line in lines[1:]:
            line = line.replace(',,', ',')
            line = line.strip()
            first, rest_of_line = line.split(',', 1)
            first = first.strip('"')
            last, rest_of_line = rest_of_line.split(',', 1)
            last = last.strip('"')
            weight, rest_of_line = rest_of_line.split(',', 1)
            weight = weight.strip('"')
            height, geo = rest_of_line.split(',', 1)
            height = height.strip('"')
            geo = geo.strip('"')
            weight = float(weight)
            height = float(height)
            parsed_data.append([first, last, weight, height, geo])
        df = pd.DataFrame(parsed_data, columns=headers)
        return df
