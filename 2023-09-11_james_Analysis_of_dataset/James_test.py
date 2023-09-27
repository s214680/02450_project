# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# %%
data = pd.read_csv('/Users/luchengliang/02450_project/2023-09-07_jennifer_data_filtration/2023-09-08_jennifer_filtered_by_country.csv')
print(data)

# %%
value_counts = data['age'].value_counts()
value_counts.plot(kind = 'bar')

plt.title("The distribution for the people with same age")
plt.xlabel("Age")
plt.ylabel("Counts")
plt.show()

# %%
age_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

age_labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
              '90-99']

data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

age_group_counts = data['age_group'].value_counts().reset_index()
age_group_counts.columns = ['Age Group', 'Counts']

#seaborn plot
plt.figure(figsize=(8, 6)) 
sns.barplot(x='Age Group', y='Counts', data=age_group_counts, palette='viridis')

# Customize the plot (optional)
plt.title("The distribution for the people with the same age")
plt.xlabel("Age")
plt.ylabel("Counts")

# Show the plot
plt.show()

# %%
education_num_counts = data['education-num'].value_counts().reset_index()
education_num_counts.columns = ['Education number', 'Counts']

# Create a bar plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x='Education number', y='Counts', data=education_num_counts, palette='viridis')

# Customize the plot (optional)
plt.title("The distribution for the people with the same education number")
plt.xlabel("Education number")
plt.ylabel("Counts")

# Show the plot
plt.show()

# %%
hours_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

hours_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
              '90-99']

hours_group = pd.cut(data['hours-per-week'], bins=hours_bins, labels=hours_labels, right=False)

hours_group_counts = hours_group.value_counts().reset_index()
hours_group_counts.columns = ['Hours Group', 'Counts']

#seaborn plot
plt.figure(figsize=(8, 6)) 
sns.barplot(x='Hours Group', y='Counts', data=hours_group_counts, palette='viridis')

# Customize the plot (optional)
plt.title("The distribution for the number of people working hours per week")
plt.xlabel("Hours per Week")
plt.ylabel("Counts")

# Show the plot
plt.show()

# %%
count_workclass = Counter(data['workclass']).most_common()
count_workclass_key = [key for key, _ in count_workclass]
count_workclass_value = [item for _, item in count_workclass]
#print(count_workclass_key)
#print(count_workclass_value)

#seaborn plot
plt.figure(figsize=(8, 6)) 
work_plot = sns.barplot(x=count_workclass_key, y=count_workclass_value, palette='viridis')

work_plot.set_xticklabels(work_plot.get_xticklabels(), rotation=45,
                        horizontalalignment='right')

# Customize the plot (optional)
plt.title("The distribution for the number of people with the same work class")
plt.xlabel("Workclass", )
plt.ylabel("Counts")

# Show the plot
plt.show()


# %%
count_occupation = Counter(data['occupation']).most_common()
count_occupation_key = [key for key, _ in count_occupation]
count_occupation_value = [item for _, item in count_occupation]
#print(count_workclass_key)
#print(count_workclass_value)

#seaborn plot
plt.figure(figsize=(8, 6)) 
occupation_plot = sns.barplot(x=count_occupation_key, y=count_occupation_value, palette='viridis')

occupation_plot.set_xticklabels(occupation_plot.get_xticklabels(), rotation=45,
                        horizontalalignment='right')

# Customize the plot (optional)
plt.title("The distribution for the number of people with the same occupation")
plt.xlabel("Occupation")
plt.ylabel("Counts")

# Show the plot
plt.show()


