"""
Created on Wed Sep 13 12:34:05 2023

@author: Emma Louise Blair (s214680)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load filtered_countries.csv file
data = pd.read_csv('C:/Users/s214680/OneDrive - Danmarks Tekniske Universitet/3. Semester/02450 Introduktion til machine learning og data mining/Project/project1_02450/Data/2023-09-08_jennifer_filtered_complete.csv')
data

# Extracting attribute columns
age_col = data['age']
edu_num_col = data['education-num']
hpw_col = data['hours-per-week']

# Compute values (taken from ex3_2_1.py)
mean_age = age_col.mean()
std_age = age_col.std(ddof=1)
median_age = np.median(age_col)
range_age = age_col.max()-age_col.min()

mean_edu_num = edu_num_col.mean()
std_edu_num = edu_num_col.std(ddof=1)
median_edu_num = np.median(edu_num_col)
range_edu_num = edu_num_col.max()-edu_num_col.min()

mean_hpw = hpw_col.mean()
std_hpw = hpw_col.std(ddof=1)
median_hpw = np.median(hpw_col)
range_hpw = hpw_col.max()-hpw_col.min()

# Display results
#print('Vector:', age_col)
print('Age:')
print('Mean:', mean_age)
print('Standard Deviation:', std_age)
print('Median:', median_age)
print('Range:', range_age)
print('')
print('Education number:')
print('Mean:', mean_edu_num)
print('Standard Deviation:', std_edu_num)
print('Median:', median_edu_num)
print('Range:', range_edu_num)
print('')
print('Hours-per-week:')
print('Mean:', mean_hpw)
print('Standard Deviation:', std_hpw)
print('Median:', median_hpw)
print('Range:', range_hpw)
print('')
print('Done basic summary statistics of attributes')

print('Hours-per-week maximum: ', hpw_col.max())
print('Hours-per-week minimum: ', hpw_col.min())

print('Age maximum: ', age_col.max())
print('Age minimum: ', age_col.min())




