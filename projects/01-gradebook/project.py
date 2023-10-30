# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    d = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }
    for i in grades.columns:
        if i.startswith('lab') and i[:5] not in d['lab']:
            d['lab'].append(i[:5])
        elif i.startswith('project') and i[:9] not in d['project']:
            d['project'].append(i[:9])
        elif i.startswith('Midterm') and i[:7] not in d['midterm']:
            d['midterm'].append(i[:7])
        elif i.startswith('Final') and i[:5] not in d['final']:
            d['final'].append(i[:5])
        elif i.startswith('disc') and i[:12] not in d['disc']:
            d['disc'].append(i[:12])
        elif 'checkpoint' in i and i[:22] not in d['checkpoint']:
            d['checkpoint'].append(i[:22])
    return d


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    projects = get_assignment_names(grades)['project']
    
    project_scores = pd.DataFrame()

    for project in projects:
        earned_points = grades[project].fillna(0)
        
        if f"{project}_free_response" in grades.columns:
            earned_points += grades[f"{project}_free_response"].fillna(0)

        max_points = grades[f"{project} - Max Points"].fillna(0)
        if f"{project}_free_response - Max Points" in grades.columns:
            max_points += grades[f"{project}_free_response - Max Points"].fillna(0)

        max_points.replace(0, pd.NA, inplace=True)

        project_score = earned_points / max_points

        project_scores[project] = project_score

    total_earned = project_scores.sum(axis=1)
    total_max_points = len(projects)
    total_scores = total_earned / total_max_points

    project_scores['Total'] = total_scores

    return project_scores['Total']

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def late_second(time):
    num = 0
    for i in time:
        hour = i.split(':')[0]
        minute = i.split(':')[1]
        second = i.split(':')[2]
        total = int(hour) * 3600 + int(minute) * 60 + int(second)
        if 0 < total <= 6*60*60:
            num += 1
    return num

def last_minute_submissions(grades):    
    late = {}
    labs = get_assignment_names(grades)['lab']
    for lab in labs:
        late[lab] = late_second(grades[f"{lab} - Lateness (H:M:S)"])

    return pd.Series(late)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def late_week(time):
  num = []
  for i in time:
    week = 0
    hour = i.split(':')[0]
    minute = i.split(':')[1]
    second = i.split(':')[2]
    total = int(hour) * 3600 + int(minute) * 60 + int(second)
    if total < 6*60*60:
        week = 0
        num.append(week)
    elif 0 < total <= 60*60*24*7:
        week = 1
        num.append(week)
    elif 60*60*24*7 < total < 60*60*24*7*2:
        week = 2
        num.append(week)
    else:
        week = 3
        num.append(week)
  return num

def lateness_penalty(col):
    num = late_week(col)
    for i in range(len(col)):
        if num[i] == 0:
            num[i] = 1.0
        elif num[i] == 1:
            num[i] = 0.9
        elif num[i] == 2:
            num[i] = 0.7
        else:
            num[i] = 0.4
    return pd.Series(num)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    labs = get_assignment_names(grades)['lab']
    lab_grade = grades[labs]
    lab_grade = lab_grade.fillna(0)
    final = pd.DataFrame()
    for lab in labs:
        penalty = lateness_penalty(grades[f"{lab} - Lateness (H:M:S)"])
        adjusted_grade = lab_grade[lab] * penalty
        lowest = adjusted_grade.min()
        final[lab] = (adjusted_grade - lowest) / (grades[f"{lab} - Max Points"] - lowest)
    return final


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    result = pd.Series()
    lowest = processed.min(axis=1)
    result = (processed.sum(axis=1) - lowest) / (processed.shape[1] - 1)
    return result


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    names = get_assignment_names(grades)
    project_point = projects_total(grades)
    lab_point = lab_total(process_labs(grades))
    mid = grades['Midterm'].fillna(0)
    mid_point = mid / grades['Midterm - Max Points']
    final = grades['Final'].fillna(0)
    final_point = final / grades['Final - Max Points']
    dis = names['disc'][0]
    dis_max = grades[f'{dis} - Max Points'][0]
    di = grades[names['disc']].fillna(0)
    disc_point = di.sum(axis=1) / (dis_max * len(names['disc']))
    che = names['checkpoint'][0]
    ch = grades[names['checkpoint']].fillna(0)
    che_max = grades[f'{che} - Max Points'][0]
    che_point = ch.sum(axis=1) / (che_max * len(names['checkpoint']))
    total = project_point * 0.3 + lab_point * 0.2 + mid_point * 0.15 + final_point * 0.3 + disc_point * 0.025 + che_point * 0.025
    return total


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def final_grades(total):
    final = []
    for i in total:
        if i >= 0.9:
            final.append('A')
        elif 0.8 <= i < 0.9:
            final.append('B')
        elif 0.7 <= i < 0.8:
            final.append('C')
        elif 0.6 <= i < 0.7:
            final.append('D')
        else:
            final.append('F')
    return pd.Series(final)
    
def letter_proportions(total):
    final = final_grades(total)
    result = {}
    letter = ['A', 'B', 'C', 'D', 'F']
    for x in letter:
        count = np.count_nonzero(final == x)
        result[x] = count / len(final)
    result = pd.Series(result)
    return result.sort_values(ascending=False)


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    result = {}
    final_breakdown = final_breakdown.fillna(0)
    result['PID'] = final_breakdown['PID']
    col = []
    for x in final_breakdown.columns[1:]:
        question = x.split(' ')[1]
        for y in question_numbers:
            if int(question) == y:
                col.append(x)
    score = final_breakdown[col].sum(axis=1)

    total = []
    for x in col:
        points = x.split('(')[1]
        point = points.split(' ')[0]
        total.append(float(point))
    total_points = sum(total)

    result['Raw Redemption Score'] = score / total_points

    return pd.DataFrame(result)
    
def combine_grades(grades, raw_redemption_scores):
    return grades.merge(raw_redemption_scores, on='PID', how='left')


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def z_score(ser):
    mean = ser.mean()
    std = ser.std(ddof=0)
    return (ser - mean) / std
    
def add_post_redemption(grades_combined):
    result = grades_combined.copy()
    mid_score = grades_combined['Midterm'].fillna(0)
    mid_max = grades_combined['Midterm - Max Points'].iloc[0]
    mid_prop = mid_score / mid_max
    
    raw_z = z_score(grades_combined['Raw Redemption Score'])
    mid_z = z_score(mid_prop)

    updated_scores = mid_prop.copy()

    for i in range(len(raw_z)):
        if raw_z[i] > mid_z[i]:
            updated_scores[i] = raw_z[i]
        else:
            updated_scores[i] = mid_z[i]
    mid_mean = mid_prop.mean()
    mid_std = mid_prop.std(ddof=0)

    result['Midterm Score Pre-Redemption'] = mid_prop
    result['Midterm Score Post-Redemption'] = updated_scores * mid_std + mid_mean

    return result


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    after = add_post_redemption(grades_combined)
    total = total_points(grades_combined)
    mid = grades_combined['Midterm'].fillna(0)
    mid_point = mid / grades_combined['Midterm - Max Points']
    total_wo_mid = total - mid_point * 0.15
    total_after = total_wo_mid + after['Midterm Score Post-Redemption'] * 0.15
    return total_after
    
def proportion_improved(grades_combined):
    final_after = total_points_post_redemption(grades_combined)
    final_letter = final_grades(final_after)
    final_before = total_points(grades_combined)
    final_letter_before = final_grades(final_before)

    count = np.count_nonzero(final_letter != final_letter_before)
    return count / len(final_letter)