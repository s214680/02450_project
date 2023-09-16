"""
File_name: 2023-09-07_2023-09-15_jennifer_pca.txt
Author: Jennifer Fortuny I Zhan
Date: Thursday, 2023-09-07
"""
"""
This file breifly explains the steps I took to carry out the PCA analysis.
The PCA analysis considers these three required aspects.
(standarisation of data by standard deviation - if attributes have different scales.)
1. The amount of variation explained as a function of the number of PCA components included.
2. The principal directions of the considered PCA components (by plotting or interpretation in temrs of the features).
3. The data projected onto the considered principal components.
----------------------------------------------------------------------------------------------------------------------------------
Based on the assignment's requirements, below is an outline of the steps I took to complete the principal component analysis (PCA).

Create Data Visualisation:
1. Check for outliers.
2. Check if the attributes are normally distributed
3. Check correlations between variables.

After getting a good understanding of the data, if needed,
I will carry out a standarisation using standard deviation

Carry out PCA:
1. Explain the variations compared to the number of PCA components.
2. What is the principal directions of the PCA components?
3. Project the data onto the principal components in consideration.
----------------------------------------------------------------------------------------------------------------------------------
"""
# Beofre carrying out any work, I import my required libraries, and load my data into a pandas DataFrame:
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load data:
data= pd.read_csv('/Users/jenniferfortuny/02450_project/2023-09-15_jennifer_pca_section/2023-09-08_jennifer_filtered_complete_copy.csv')


"""
Create Data Visualisation:
--------------------------------------------------------------------------------------------------------------------------------
1. Check for outliers.
For the continuous attributes: age, edu-num, hour-per-week.
I make a histogram with a boxplot.

For the categorical attributes: workclass, occupation.
I only make a histogram.
"""
# I begin by splitting my attributes into two lists:
continuous_attributes = data.columns[:3]
categorical_attributes = data.columns[3:]
# I use one-hot encoding to encode categorical attributes
categorical_encoded = pd.get_dummies(data, columns=categorical_attributes, drop_first=True)


# I plot the histogram with box plot for each of the three continous attributes:
for column in continuous_attributes:
    fig = plt.figure(figsize=(10, 6))

    # Create grid
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Histogram on the top axis (ax1)
    sns.histplot(data[column], kde = True, ax = ax1)
    ax1.set_title(f'Histogram with Boxplot for {column}')
    ax1.set_xlabel('') # This is to keep the x-axis label empty on the top
    ax1.set_ylabel('Frequency')

    # Box plot on the bottom axis (ax2)
    sns.boxplot(x = data[column], ax = ax2)
    ax2.set_xlabel(column)

    # I find and label the outliers on the box plot
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[column] < (Q1 - 1.5 * IQR))
                                   | (data[column] > (Q3 + 1.5 * IQR))][column]
    for outlier in outliers:
        ax2.text(outlier, -0.18, f'{outlier:.0f}', ha='center', va='top', fontsize=8, color='blue')
        # -0.02 places my text at y = -0.02, i.e. below the dot on the box plot.
    plt.tight_layout()
    ax2.set_ylabel(column)
    plt.show()


# I plot the histogram-only plots for each of the three continous attributes:
for column in categorical_attributes:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde = True)
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    # I angle the x-axis labels a bit to show all the words clearly
    plt.xticks(rotation=20, ha='right', fontsize=10)

    plt.tight_layout()
    plt.show()

"""
2. Check for outliers.
# To check if the attributes are normally distributes,
# I will begin by reflectiong on the results of the histograms:
# continous attributes:
#     age: tail to the right, so right sqewed distribution.
#     edu-num: looks like a bimodial distribution.
#     hours-per-week: looks like an extreme plot with on highly frequent value at 35-40.
# categorical attributes:
#    workclass: extreme with "Private" at the highest frequency.
#    occupation: some outliers, most seem to be at the similar frequency, no clear trend.

# I will use Q-Q plots to determin if they attributes have a formal normal distribution.
# If the data are mostly on the y=x line in the Q-Q plot, then we can assume there is a normal distribution.
# I will plot the continous variables' Q-Q plots:
"""

# I will rename the previous variable:
for column in continuous_attributes:
    # Since I am also using statsmodels now, in addition to matplotlib.
    # Here I create a figure and axis just for the Q-Q plot
    plt.figure(figsize=(10, 6))
    sm.qqplot(data[column].dropna(), line = '45', fit = True)
    plt.title(f'Q-Q Plot for {column}')
    plt.show()
    plt.close()
"""
Notes:
The age attribute shows a U-shape pattern.
The points fall below the y=x line at the lower end, and above the line at the higher end.
This suggests we have fewere extreme values than we would expect compared to a perfect normal distribution
i.e. it is a lighter-tailed than a normal distribution.
Therefore we have a somewhat uniform distribution, what is more concentrated around the median, less so towards the tails.
This means our data could have a wide range of ages, 
and when compared to a normal distribution -  as many individuals who are very young or very old.
This could be becasue the data is focused on working age adults, which would include less people who are very young or very old.
The right skew in the histogram suggests relatively less older individuals than younger ones.

The edu-num distribution shows a shape similar to w, this suggests a binomial distribution, in agreement with the histogram.
This suggests the data has two major distributions of education levels.

The hours-per-week Q-Q plot shows a sharp incline, a long flat section, then another sharp incline.
The flat section is the major accumulation of the data, with coincides with the histogram.
The large portion of the dataset are people who work standard full-time hours.
This is a non-linear Q-Q plot, so it does NOT show a normal distribution.

For the categorical variables, I don't expect a normal distribution.
It might be more valuable to understand the frequency and mode of these attributes.
From the observations of the histogram:
The workclass attributes prodominatly represents the private sector.
The occpupation attribute shows a more uniform distribution, with some roles acting as outliers.
"""

"""
3. Check correlations between variables.
I want to visualise this continous with a heatmap, using a correlation matrix.
This will help show the relationships between the continous variables.
"""


# First a calculate the correlation matrix:
corr = data[continuous_attributes].corr()

# Then I make a heatmap:
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot = True, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.title('Correlation Heatmap')
plt.yticks(rotation=0)
plt.xticks(rotation=15)
plt.show()

"""
Notes on the heatmap:
There are no strong correlation between any of these three attributes to eachother.
This means that the variables are independent of each other in the dataset.
"""
"""
Standarising the data - for each attribute to give equal contributions to pca.
I will scenter the data around 0 by substracting the mean of the attribute from the data point,
and scale them using the standard diviation, 
so they have the variance of one.

"""
# I will only use the numerical columns, i.e. the continous attributes.
# I will use Scikit-learn's StandardSaler:
# First I will initialise the standard scaler
scaler = StandardScaler()
# Then, fit and transform the continous data directly
standardised_continuous_attributes = scaler.fit_transform(data[continuous_attributes])

# Now I will make this numpy array I made into a dataframe for the continous data.
standardised_continuous_attributes = pd.DataFrame(standardised_continuous_attributes, columns = continuous_attributes)

"""
Carry out pca:
----------------------------------------------------------------------------------------------------------------------------------
1. Explain the variations compared to the number of pca components.
I will begin with a pca analysis WITHOUT reducing the dimensionality, this will allow me to understadn the total explained variance for each component.
Then, I will plot a cumulative explained variance plot, to have a look at the explained variance for each component.
"""
# I first initialise pca
pca = PCA()

# Then I fit the pca onto our standarised data.
pca.fit(standardised_continuous_attributes)

# I calculate the cumlative explain variance
explained_variances = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variances)

# I make a plot this cumulative explained variance
plt.figure(figsize=(10,6))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')

# I use a for-loop to add labels to my data:
for i, value in enumerate(cumulative_variance):
    plt.text(i, value + 0.02, f"PC{i+1}: {value*100:.2f}%")

plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance as Number of Components Increases')
plt.grid(True)    # Here I added gridlines onto the plot
plt.show()

"""
Notes
The cumulative explained variance plot showed a dotted line with a linear slop going
through three points. This means each PC roughly explains the same amount of variance
in the data. This means there are three PCs extracted, and that the data's variance is mostly evenly
distributed across these three dimensions.
"""
"""
2. Determine the principal directions of the PCA components:
At this point I need to find out how much each individual continous attribute,
i.e. age, edu-num, and hours-per-week.
I will examine how they contribute to each of the PCs, via loadings, weights, coefficients.
If an attribute has a high value for a certain PC,
then I will consider the attribute as having a strong relationship with that PC.
"""
# I isolate the loadings from the PCA model:
loadings = pca.components_
# I transform the loadings into a df and display using print():
loadings_df = pd.DataFrame(loadings, columns=standardised_continuous_attributes.columns,
                           index = [f'PC{i+1}' for i in range(loadings.shape[0])])
print(loadings_df)
"""
From the printed table:
I can see the most positive value is between age and PC2, at 0.82.
The most negative value is between hours-per-week and PC3, at -0.75.
"""

# 3. Project the data onto the principal components in consideration
# I will use the PCA model to transform my original data, based on the calculated principal componentes.
pca_data = pca.transform(standardised_continuous_attributes)
# I visualisat the data that are most represented, i.e. the first two principal components.
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data Projected onto First Two Principal Components')
plt.grid(True)
plt.show()

"""
Notes:
There are three lines, well, linear clusters of points,
that show a negative linear slop on the graph.

"""