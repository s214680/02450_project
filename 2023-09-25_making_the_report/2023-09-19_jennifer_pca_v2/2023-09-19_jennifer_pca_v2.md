Beofre carrying out any work, I import my required libraries, and load my data into a pandas DataFrame.

Import Libraries


```python
import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

Constants


```python
DATA_PATH = '/Users/jenniferfortuny/02450_project/2023-09-15_jennifer_pca_section/2023-09-08_jennifer_filtered_complete_copy.csv'
```

Load the dataset


```python
def load_data(path):
    return pd.read_csv(path)
```


```python
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
        plt.tight_layout()
        plt.show()
```

Create data visualisation


```python
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
```

Q-Q plots for continuous attributes


```python
def plot_qq(data, continous_attributes):
    """Plot a Q-Q plot for a given"""
    for column in continous_attributes:
        # Since I am also using statsmodels now, in addition to matplotlib.
        # Here I create a figure and axis just for the Q-Q plot.
        plt.figure(figsize=(10, 6))
        sm.qqplot(data[column].dropna(), line='45', fit=True)
        plt.title(f'Q-Q Plot for {column}')
        plt.show()
```


```python
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
```

PCA analysis


```python
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
```


```python
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
```

MCA analysis


```python
def perform_mca(categorical_encoded):
    """Perform PCA analysis on the encoded categorical data"""
    # I begin by initialising MCA with the prince module and fit the encoded categorical data:
    # To begin, I do not not specify the number of components.
    mca = prince.MCA()
    mca = mca.fit(categorical_encoded)
    
    # Now I transform the categorical data
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
```

Load the dataset


```python
data = load_data(DATA_PATH)
```

Splitting attributes into continuous and categorical


```python
continuous_attributes = data.columns[:3]
categorical_attributes = data.columns[3:]
```

One-hot encoding for categorical attributes


```python
categorical_encoded = pd.get_dummies(data, columns=categorical_attributes, drop_first=True)
```

Create data visualisations to detect outliers

1. Check for outliers <br> 
To check if the attributes are normally distributes,
I begin by reflectiong on the results of the histograms:
continous attributes:
     age: tail to the right, so right sqewed distribution.
     edu-num: looks like a bimodial distribution.
     hours-per-week: looks like an extreme plot with on highly frequent value at 35-40.
 categorical attributes:
    workclass: extreme with "Private" at the highest frequency.
    occupation: some outliers, most seem to be at the similar frequency, no clear trend.


```python
plot_histogram_boxplot(data, continuous_attributes)
plot_histogram(data, categorical_attributes)
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    



    
![png](output_26_3.png)
    



    
![png](output_26_4.png)
    



    
![png](output_26_5.png)
    


Create Q-Q plots for continuous attributes to test for normal-distribution.

I use Q-Q plots to determine if they attributes have a formal normal distribution.
If the data are mostly on the y=x line in the Q-Q plot, then we can assume there is a normal distribution.
I plot the continous variables' Q-Q plots:

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


```python
plot_qq(data, continuous_attributes)
```


    <Figure size 1000x600 with 0 Axes>



    
![png](output_28_1.png)
    



    <Figure size 1000x600 with 0 Axes>



    
![png](output_28_3.png)
    



    <Figure size 1000x600 with 0 Axes>



    
![png](output_28_5.png)
    


Create correlation heatmap


```python
plot_correlation_heatmap(data, continuous_attributes)
```


    
![png](output_30_0.png)
    


Standarize continuous data


```python
continuous_attributes_standarized = StandardScaler().fit_transform(data[continuous_attributes].dropna())
```

PCA analysis, with 3D scatter plot


2.Carry-out the PCA analysis:

If your attributes have different scales you,
should include the step where the data is standardizes
by the standard deviation prior to the PCA analysis.

1. The amount of variation explained as a function of the number of PCA components included.
2. The principal direction of the considered PCA components (either find a way to plot them or interpret them in terms of the features).
3. The data projected onto the considered principal components.

Now I wil carry out the PCA analysis.
I start by running the PCA without initially setting a number of components.
This helps me understand the total explained variance for each component.
It is also helpful in determining the number of PCs that would capture most of the dataset's variance.

Given that we are working with a large number of potential PCs
I carry out some dimensional reduction, 
and only use three PCs for the PCA analysis and visualisation.
I also try to retain as much information about the data as possible.
21 PCs is a lot of PCs. This is most likely the result of having used one-hot encoding for our categorical data.

Given this situation. I would like to apply the PCA only to the continous part,
and the MCA on the categorical part.


```python
pca, pca_coordinates = perform_pca(continuous_attributes_standarized)
plot_3d_scatter_for_pca(pca_coordinates)
```


    
![png](output_34_0.png)
    



    
![png](output_34_1.png)
    


MCA analysis

Noting the result from this MCA analysis on the categorical data,
where 2 PCs explain all the variation.


```python
perform_mca(categorical_encoded)
```


    
![png](output_36_0.png)
    



    
![png](output_36_1.png)
    





<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MCA()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">MCA</label><div class="sk-toggleable__content"><pre>MCA()</pre></div></div></div></div></div>


