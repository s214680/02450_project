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
    for column in continuous_attributes:
        fig = plt.figure(figsize=(10, 6))

        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        sns.histplot(data[column], kde=True, ax=ax1)
        ax1.set_title(f'Histogram with Boxplot for {column}')
        ax1.set_xlabel('')
        ax1.set_ylabel('Frequency')

        sns.boxplot(x=data[column], ax=ax2)
        ax2.set_xlabel(column)

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))][column]
        for outlier in outliers:
            ax2.text(outlier, -0.18, f'{outlier: .0f}', ha='center', va='top', fontsize=8, color='blue')
        plt.tight_layout()
        plt.show()

# Create data visualisation
def plot_histogram(data, categorical_attributes):
    for column in categorical_attributes:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=20, ha='right', fontsize=10)
        plt.tight_layout()
        plt.show()

# Q-Q plots for continuous attributes
def plot_qq(data, continous_attributes):
    for column in continous_attributes:
        plt.figure(figsize=(10, 6))
        sm.qqplot(data[column].dropna(), line='45', fit=True)
        plt.title(f'Q-Q Plot for {column}')
        plt.show()

# Correlation heatmap
def plot_correlation_heatmap(data, continous_attributes):
    corr = data[continous_attributes].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.yticks(rotation=0)
    plt.xticks(rotation=15)
    plt.show()

# PCA analysis
def perform_pca(data_standarized):
    pca = PCA()
    principal_components_full = pca.fit_transform(data_standarized)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    for i in range(len(cumulative_variance)):
        plt.annotate(f"PC{i+1}: {cumulative_variance[i] * 100:.2f}%", (i, cumulative_variance[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')
    plt.xlabel('Number of Compnents')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance as Number of Components Increases')
    plt.grid(True)
    plt.show()
    return pca

# MCA analysis
def perform_mca(categorical_encoded):
    mca = prince.MCA()
    mca = mca.fit(categorical_encoded)
    mca_coordinates = mca.transform(categorical_encoded)
    eigenvalues = mca.eigenvalues_
    total_inertia = sum(eigenvalues)
    explained_inertia = [eig/total_inertia for eig in eigenvalues]
    plt.figure(figsize=(10, 6))
    plt.plot(explained_inertia, marker='o', linestyle='--', color='b')
    for i, inertia in enumerate(explained_inertia):
        plt.annotate(f"PC{i+1}: {inertia*100:.2f}%", (i, inertia), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Inertia')
    plt.title('Explained Inertia as Number of Components Increases')
    plt.grid(True)
    plt.show()
    return mca

# Load the dataset
data = load_data(DATA_PATH)

# Splitting attributes
continuous_attributes = data.columns[:3]
categorical_attributes = data.columns[3:]

# One-hot encoding for categorical attributes
categorical_encoded = pd.get_dummies(data, columns=categorical_attributes, drop_first=True)

# Create data visualisations for outliers
plot_histogram_boxplot(data, continuous_attributes)
plot_histogram(data, categorical_attributes)

# Q-Q plots for continuous attributes
plot_qq(data, continuous_attributes)

# Correlation heatmap
plot_correlation_heatmap(data, continuous_attributes)

# Standarize continuous data
continuous_attributes_standarized = StandardScaler().fit_transform(data[continuous_attributes].dropna())

# PCA analysis
perform_pca(continuous_attributes_standarized)

# MCA analysis
perform_mca(categorical_encoded)
