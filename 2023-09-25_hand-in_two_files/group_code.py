"""
Below is the python code used to do the initial data filtration:
"""
"""
File_name: 2023-09-07_jennifer_step1_filer_countries.py
Author: Jennifer Fortuny I Zhan
Date: Thursday, 2023-09-07

This file contains the python code I used to extract the countries with the top 10 gdp per capita from the original dataset.

"""
# Make sure to work in an enviornment were I have pandas installed.
import pandas as pd

# We will first load the entire data set into a variable:
# Make sure I am in the same working directory as the data.
data = pd.read_csv('adult.data', header=None)

# Given that our file has no column names, I will add them by referencing the adult.names file.
# After view in the list using less on the terminal, I notice that the last few lines of the file are about the attributes.
# I know that the first attribute is "age", and all the lines after it, until the last line of the file, are attributes.
# To begin, I will find out the line number for "age: ", and use this to help extract the other attribute names later.
# First I open the file containing the attribute name in read mode, and I read its lines into a list.
with open('adult.names', 'r') as infile:
    lines = infile.readlines()

# Then, I look through this list from the end, to find where "age: " appears:
for index, line in enumerate(reversed(lines), start=1):
    if "age: " in line:
        last_n_lines_we_need = index
        break

print(f" 'age: ', and then all the other attributes, begin to show from the last {last_n_lines_we_need}th line of the file.")
# The result shows that age, and the other attributes, begin to apear from the last 14th line of the file.
# Now I will join these 14 lines into a list.
# The lines containing the attributes are from line -14 until the end of the file, I make this into a variable:
# However, I noticed I forgot a line that apeared 2 lines before the line containing age: which was also part of the columns in our data.
# This is the coloumn which indicates income, and it takes on the values <=50K and >50K.
# So I will increase the number for the last_n_lines_we_need by 1.
# And also add one to the line we begin our search from, because there is an empty line between <=50K, >50K and age:.
attribute_lines = lines[-(last_n_lines_we_need + 1) + 1:]

# Now, the attributes always come with a :, like in "age: ".
# So I look for that in each item on our list
# When the : is present, I split that item and take what ends up at the 0th index.
# This could be the word that comes between :
attributes = []
for line in attribute_lines:
    attribute_name = line.split(":")[0]
    attributes.append(attribute_name.strip())

# We also need to add a name for the column that indicates income.
# This column is describe as the first attribute in the adult.name file, but is the last column in the adult.data file.
# So I will add it to the end of our attributes list.
attributes.append('income')

# Now we have the attribute names in a list.
# We already loaded out data at the begining.
# Now we add the column names we just found to the data:
data.columns = attributes
print(data.columns)

# List of countries with top 10 GDP percapital we found previously.
# First I want to check how the countries are written in the file itself, to prevent mismatching.
uniq_country_names = [country.strip() for country in data['native-country'].unique()]
print(uniq_country_names)
# I create a list of the countries we are interested in, matching the country name format in the file:
top_10_countries = [
    'Luxembourg', 'Ireland', 'Norway', 'Switzerland', 'Singapore', 
    'Qatar', 'United-States', 'Iceland', 'Denmark', 'Australia'
]
# Before filtering the data, I want to make sure that these 10 countries exist in our original data.
for country in top_10_countries:
    if country not in uniq_country_names:
        print(f"'{country}' doesn't appear in the original data.")
    else:
        print(f"'{country}' is in our data")
# The results showed that most countries we were interested in were not in our list
# Therefore I will decide on a new set of countries to focus on.
# I decided to look at only South-East Asian countries that exist in the original data.
# They are of investigative interest because many South-East Asian economies are developing economies experiencing growth.
south_east_asia = ["Cambodia", "Laos", "Philippines", "Vietnam", "Thailand"]

# Finally, I filter our original data by these five countries.
# I noticed that the entries in the native-country column of the original data are surrounded by white space.
# Therefore I will remove these white spaces first.
data['native-country'] = data['native-country'].str.strip()

# The variable below only has the rows of data about people that have one of our five countries in their native-country column.
filtered_by_country = data[data['native-country'].isin(south_east_asia)]
print(filtered_by_country)

# I save this output as a csv file for further processing and sharing.
filtered_by_country.to_csv('2023-09-08_jennifer_filtered_by_country.csv', index=False)
# The csv file it produced contains 320 rows, and 15 columns.
# Next, I will reduce the number of columns to 6.

"""
Below is the python code used to complete step_2 of the data filtration, i.e. reducing the number of columns to 6.
"""
"""
File_name: 2023-09-08_jennifer_step2_filter_columns.py
Author: Jennifer Fortuny I Zhan
Date: Thursday, 2023-09-08
Main edits: 2023-09-15, added "income" as a column".

This file contains the python code I used to filter 6 out of 15 columns fron the filtered_by_country.csv file.

"""
# I will only retain these coloumns: age, education-num, hour-per-week, workclass, and occupation.
# The first three columns are continuous attributes, the last two are categorical.
# I use less on the terminal to get a look at the data we are wokring on:
# we have all the headers in place, and no index column.

# As before, I make sure to work in an enviornment with the pandas package:
import pandas as pd

# First I load my filtered_countries.csv file
data = pd.read_csv('2023-09-08_jennifer_filtered_by_country.csv')

# Now I specify the columns that I want to keep:
keep_attributes = data[['age', 'education-num', 'hours-per-week', 'workclass', 'occupation', 'income']]

# Finally, I save this filtered data as a csv file for further processing and sharing.
keep_attributes.to_csv('2023-09-08_jennifer_filtered_complete.csv', index=False)

# Below is some information about this dataset:
rows, columns = keep_attributes.shape
datapoints = rows * columns
print(f"We have {rows} rows and {columns} columns, giving us {datapoints} datapoints.\n")
print(f"The {columns} columns, i.e. attributes, are:")
print(", ".join(keep_attributes.columns))

"""
Below is the python code used to check the basic summary statistics
"""
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

"""
Below is the code used to carryout preliminary data vidualisation:
"""
# Author James Cheng-Liang Lu (s220034)

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


"""
Below is the python code used to carry out the data visualisation section of the report
"""
# Author Jennifer Fortuny I Zhan (s230705)

import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
DATA_PATH = '/Users/jenniferfortuny/02450_project/2023-09-15_jennifer_pca_section/2023-09-08_jennifer_filtered_complete_copy.csv'

def load_data(path):
    return pd.read_csv(path)

def plot_histogram_boxplot(data, continuous_attributes):
    """Plot a histogram with boxplot for each continous attributes"""
    for column in continuous_attributes:
        fig = plt.figure(figsize=(10, 6))
        
        # Create grid
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Histogram on the top axis (ax1)
        sns.histplot(data[column], kde=True, ax=ax1)
        ax1.set_title(f'Histogram with Boxplot for {column}')
        ax1.set_xlabel('')
        ax1.set_ylabel('Frequency')

        # Box plot on the bottom axis (ax2)
        sns.boxplot(x=data[column], ax=ax2)
        ax2.set_xlabel(column)

        # I find and label any outliers on the box plot with its own value
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))][column]
        for outlier in outliers:
            ax2.text(outlier, -0.18, f'{outlier: .0f}', ha='center', va='top', fontsize=8, color='blue')
            # -0.02 places my text at y = -0.02, i.e. below the dot on the box plot.
        plt.tight_layout()
        ax2.set_ylabel(column)
        plt.show()

# Create data visualisation
def plot_histogram(data, categorical_attributes):
    """Plot a histogram for each categorical attribute."""
    for column in categorical_attributes:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        # I angle the x-axis labels a bit to show all the words clearly
        plt.xticks(rotation=20, ha='right', fontsize=10)
        plt.tight_layout()
        plt.show()

# Q-Q plots for continuous attributes
def plot_qq(data, continous_attributes):
    """Plot a Q-Q plot for a given"""
    for column in continous_attributes:
        # Since I am also using statsmodels now, in addition to matplotlib.
        # Here I create a figure and axis just for the Q-Q plot.
        plt.figure(figsize=(10, 6))
        sm.qqplot(data[column].dropna(), line='45', fit=True)
        plt.title(f'Q-Q Plot for {column}')
        plt.show()

# Correlation heatmap
def plot_correlation_heatmap(data, continous_attributes):
    """Plot a heatmap for the correlations of the continous attributes"""
    # First I calculate the correlation matrix:
    corr = data[continous_attributes].corr()
    # Then I make a heatmap:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.yticks(rotation=0)
    plt.xticks(rotation=15)
    plt.show()

# PCA analysis
def perform_pca(data_standarized):
    """Perform PCA analysis on the standarized continuous data"""
    # PCA applied without specifying the number of components.
    pca = PCA()
    principal_components_full = pca.fit_transform(data_standarized)
    
    # Now, I look into the explained variance.
    # I plot the explained variance ratio, to see how much variance each component explains.
    # This helps me determine a good number of PCs
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # I check for the first n components that explain 95% or more of the variance.
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    
    # I annotate the PCAs on the plot
    # It seems like the first 20 to 22 PCs explain over 95% of the variance in the data.
    for i in range(len(cumulative_variance)):
        plt.annotate(f"PC{i+1}: {cumulative_variance[i] * 100:.2f}%", (i, cumulative_variance[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')
    plt.xlabel('Number of Compnents')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance as Number of Components Increases')
    plt.grid(True)
    plt.show()

    return pca, principal_components_full

def plot_3d_scatter_for_pca(pca_coordinates):
    """Plot a 3D scatter plot of the first 3 principal components."""
    # Create a 3D scatter plot for the continuous attributes.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Isolate the coordinates of the first 3PCs
    x = pca_coordinates[:, 0]
    y = pca_coordinates[:, 1]
    z = pca_coordinates[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Scatter plot of the PCs for continuous data')

    plt.show()

# MCA analysis
def perform_mca(categorical_encoded):
    """Perform PCA analysis on the encoded categorical data"""
    # I begin by initialising MCA with the prince module and fit the encoded categorical data:
    # To begin, I don't not specify the number of components.
    mca = prince.MCA()
    mca = mca.fit(categorical_encoded)

    # Now I  transform the categorical data
    mca_coordinates = mca.transform(categorical_encoded)

    # Now, I take a look at MCA's explained inertia for each component,
    # which is similar to PCA's explained variance.
    # I start by getting the eigenvalues.
    eigenvalues = mca.eigenvalues_

    # Calculate the total inertia:
    total_inertia = sum(eigenvalues)

    # Now get the proportion of explained inertia:
    # Now I plot the explained inertia:
    explained_inertia = [eig/total_inertia for eig in eigenvalues]
    plt.figure(figsize=(10, 6))
    plt.plot(explained_inertia, marker='o', linestyle='--', color='b')
    # Annotating the points
    for i, inertia in enumerate(explained_inertia):
        plt.annotate(f"PC{i+1}: {inertia*100:.2f}%", (i, inertia), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Inertia')
    plt.title('Explained Inertia as Number of Components Increases')
    plt.grid(True)
    plt.show()

    # I create a 2D scatter plot for the categorical values
    # First I extract coordinates for the first two components:
    x = mca_coordinates[0]
    y = mca_coordinates[1]

    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, edgecolor="k", color="blue", alpha=0.6)
    plt.title("MCA Scatter Plot for Individuals")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

    # Display
    plt.show()

    return mca

# Load the dataset
data = load_data(DATA_PATH)

# Splitting attributes
continuous_attributes = data.columns[:3]
categorical_attributes = data.columns[3:]

# One-hot encoding for categorical attributes
categorical_encoded = pd.get_dummies(data, columns=categorical_attributes, drop_first=True)

# Create data visualisations to detect outliers
"""
1. Check for outliers. 
To check if the attributes are normally distributes,
I begin by reflectiong on the results of the histograms:
continous attributes:
     age: tail to the right, so right sqewed distribution.
     edu-num: looks like a bimodial distribution.
     hours-per-week: looks like an extreme plot with on highly frequent value at 35-40.
 categorical attributes:
    workclass: extreme with "Private" at the highest frequency.
    occupation: some outliers, most seem to be at the similar frequency, no clear trend.
"""

plot_histogram_boxplot(data, continuous_attributes)
plot_histogram(data, categorical_attributes)

"""
I use Q-Q plots to determine if they attributes have a formal normal distribution.
If the data are mostly on the y=x line in the Q-Q plot, then we can assume there is a normal distribution.
I plot the continous variables' Q-Q plots:
"""

# Q-Q plots for continuous attributes
plot_qq(data, continuous_attributes)

# Correlation heatmap
plot_correlation_heatmap(data, continuous_attributes)

"""
2.Carry-out the PCA analysis:

If your attributes have different scales you,
should include the step where the data is standardizes
by the standard deviation prior to the PCA analysis.

1. The amount of variation explained as a function of the number of PCA components included.
2. The principal direction of the considered PCA components (either find a way to plot them or interpret them in terms of the features).
3. The data projected onto the considered principal components.

"""
# Standarize continuous data
continuous_attributes_standarized = StandardScaler().fit_transform(data[continuous_attributes].dropna())

"""
Now I wil carry out the PCA analysis.
I start by running the PCA without initially setting a number of components.
This helps me understand the total explained variance for each component.
It is also helpful in determining the number of PCs that would capture most of the dataset's variance.
"""
# PCA analysis, with 3D scatter plot
pca, pca_coordinates = perform_pca(continuous_attributes_standarized)
plot_3d_scatter_for_pca(pca_coordinates)

"""
Given that we are working with a large number of potential PCs
I carry-out some dimensional reduction, 
and only use three PCs for the PCA analysis and visualisation.
I also try to retain as much information about the data as possible.
21 PCs is a lot of PCs. This is most likely the result of having used one-hot encoding for our categorical data.

Given this situation. I would like to apply the PCA only to the continous part,
and the MCA on the categorical part.
"""

# MCA analysis
perform_mca(categorical_encoded)
"""
Noting the result from this MCA analysis on the categorical data,
where 2 PCs explain all the variation.
"""
