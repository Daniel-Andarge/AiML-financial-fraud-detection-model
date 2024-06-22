import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, cdist
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway
import ipywidgets as widgets
from ipywidgets import interact
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
import folium
import geopandas as gpd
from geopy.geocoders import Nominatim
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from libpysal.weights import lat2W
from esda.moran import Moran
import plotly.express as px


def univariate_analysis(df):
    """
    Performs univariate analysis on the given dataset.
    
    Parameters:
    df (pandas.DataFrame): The input dataset.
    """
    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")

    # Numeric feature analysis
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for feature in numeric_features:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.histplot(data=df, x=feature, kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {feature}")
        
        sns.boxplot(data=df, x=feature, ax=axes[1])
        axes[1].set_title(f"Boxplot of {feature}")
        
        plt.tight_layout()
        plt.show()
        
        # Handle missing values before outlier analysis
        feature_data = df[feature].dropna()
        
        # Identify outliers
        q1 = feature_data.quantile(0.25)
        q3 = feature_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
        print(f"Outliers in {feature}: {len(outliers)}")
        
    # Categorical feature analysis
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        
        df[feature].value_counts().plot(kind='bar', ax=axes)
        axes.set_title(f"Distribution of {feature}")
        
        plt.tight_layout()
        plt.show()
        
    # Analyze the target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='class')
    plt.title("Distribution of the Target Variable (Fraud/Non-Fraud)")
    plt.show()

def bivariate_analysis(df):
    """
    Performs bivariate analysis on the given dataset.
    
    Parameters:
    df (pandas.DataFrame): The input dataset.
    """
    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")

    def plot_numeric(feature):
        if feature != 'class':
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            sns.scatterplot(ax=axes[0], data=df, x=feature, y='class')
            axes[0].set_title(f"Relationship between {feature} and the Target Variable")
            
            sns.boxplot(ax=axes[1], data=df, x='class', y=feature)
            axes[1].set_title(f"Boxplot of {feature} by Target Variable")
            
            plt.tight_layout()
            plt.show()
            
            corr = df[[feature, 'class']].corr().iloc[0, 1]
            print(f"Correlation between {feature} and the Target Variable: {corr:.2f}")
    
    def plot_categorical(feature):
        if feature != 'class':
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            df[feature].value_counts().plot(kind='bar', ax=axes[0])
            axes[0].set_title(f"Distribution of {feature}")
            
            contingency_table = pd.crosstab(df[feature], df['class'])
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1])
            axes[1].set_title(f"Contingency Table of {feature} and Target Variable")
            
            plt.tight_layout()
            plt.show()
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"Chi-square test for {feature} and the Target Variable:")
            print(f"Chi-square statistic: {chi2:.2f}")
            print(f"p-value: {p_value:.4f}")
    
    def plot_anova():
        anova_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('class')
        anova_results = {}
        for feature in anova_features:
            groups = [df[df['class'] == c][feature].dropna().values for c in df['class'].unique()]
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                anova_results[feature] = (f_stat, p_value)
        
        print("ANOVA Results:")
        for feature, result in anova_results.items():
            f_stat, p_value = result
            print(f"{feature}: F-statistic={f_stat:.2f}, p-value={p_value:.4f}")
    
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    
    interact(plot_numeric, feature=numeric_features)
    interact(plot_categorical, feature=categorical_features)
    plot_anova()

def geospatial_analysis(df):
    """
    Performs geospatial analysis on the given dataset.
    
    Parameters:
    df (pandas.DataFrame): The input dataset.
    """
    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    
    # Convert IP addresses to geographic coordinates
    df['latitude'], df['longitude'] = convert_ip_to_coords(df['ip_address'])
    
    # Filter out rows where coordinates couldn't be determined
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Create a choropleth map to visualize the distribution of transactions
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fraud_map = folium.Map(location=[0, 0], zoom_start=2)
    
    folium.Choropleth(
        geo_data=world,
        name='Choropleth',
        data=df.groupby('country')['class'].mean().reset_index(),
        columns=['country', 'class'],
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Likelihood of Fraud'
    ).add_to(fraud_map)
    
    folium.LayerControl().add_to(fraud_map)
    
    fraud_map.save('fraud_map.html')
    
    # Analyze spatial autocorrelation
    coords = df[['latitude', 'longitude']].values
    fraud_status = df['class'].values
    
    # Calculate Moran's I for local spatial autocorrelation
    w = lat2W(df.shape[0], df.shape[0])
    moran = Moran(fraud_status, w)
    
    df['moran_index'] = moran.I
    df['moran_sig'] = moran.p_sim
    
    # Investigate the relationship between location and fraud
    fraud_coords = coords[fraud_status == 1]
    non_fraud_coords = coords[fraud_status == 0]
    
    fraud_distance = pdist(fraud_coords).mean() if len(fraud_coords) > 1 else 0
    non_fraud_distance = pdist(non_fraud_coords).mean() if len(non_fraud_coords) > 1 else 0
    
    print(f"Average distance between fraudulent transactions: {fraud_distance:.2f}")
    print(f"Average distance between non-fraudulent transactions: {non_fraud_distance:.2f}")

def convert_ip_to_coords(ip_addresses):
    """
    Converts IP addresses to geographic coordinates using GeoPy (Nominatim).
    
    Parameters:
    ip_addresses (pandas.Series): The IP addresses to be converted.
    
    Returns:
    latitude (pandas.Series), longitude (pandas.Series): The geographic coordinates.
    """
    geolocator = Nominatim(user_agent="geoapiExercises")
    latitude = []
    longitude = []
    
    for ip in ip_addresses:
        try:
            location = geolocator.geocode(ip)
            if location:
                latitude.append(location.latitude)
                longitude.append(location.longitude)
            else:
                latitude.append(None)
                longitude.append(None)
        except Exception as e:
            print(f"Error converting IP address {ip} to coordinates: {str(e)}")
            latitude.append(None)
            longitude.append(None)
    
    return pd.Series(latitude), pd.Series(longitude)

def fraud_class_distribution(df):
    """
    Visualizes the distribution of the 'class' (target variable) across different features.
    
    Args:
        df (pandas.DataFrame): The input dataset.
    """
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    
    # Fraud class distribution by numerical features (2 plots per row)
    num_features = ['age']
    num_plots = len(num_features)
    num_rows = (num_plots + 1) // 2  # Ensure enough rows for all plots
    fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 6))
    axes = axes.flatten()
    
    for i, feature in enumerate(num_features):
        ax = axes[i]
        df[df['class'] == 0][feature].plot(kind='hist', bins=30, alpha=0.5, label='Non-Fraud', ax=ax)
        df[df['class'] == 1][feature].plot(kind='hist', bins=30, alpha=0.5, label='Fraud', ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Fraud Class Distribution by {feature}')
        ax.legend()
    
    # Remove any unused subplot axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    # Fraud class distribution by categorical features (2 plots per row)
    cat_features = ['source', 'browser', 'sex']
    cat_plots = len(cat_features)
    cat_rows = (cat_plots + 1) // 2 
    fig, axes = plt.subplots(cat_rows, 2, figsize=(16, cat_rows * 6))
    axes = axes.flatten()
    
    for i, feature in enumerate(cat_features):
        ax = axes[i]
        sns.countplot(x=feature, hue='class', data=df, palette='viridis', ax=ax)
        ax.set_title(f'Fraud Class Distribution by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.legend(title='Class', loc='upper right', labels=['Non-Fraud', 'Fraud'])
        ax.tick_params(axis='x', rotation=45)
    
    # Plot the top 10 categories for the 'country' feature
    ax = axes[-1]
    top_categories = df['country'].value_counts().head(10).index
    sns.countplot(x='country', hue='class', data=df[df['country'].isin(top_categories)], palette='viridis', ax=ax)
    ax.set_title('Fraud Class Distribution by Top 10 Countries')
    ax.set_xlabel('Country')
    ax.set_ylabel('Count')
    ax.legend(title='Class', loc='upper right', labels=['Non-Fraud', 'Fraud'])
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def time_series_analysis(df):
    """
    Performs time-series analysis on the given dataset.
    
    Parameters:
    df (pandas.DataFrame): The input dataset.
    """
    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    
    # Parse the purchase_time to datetime
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Extract time components
    df['transaction_hour'] = df['purchase_time'].dt.hour
    df['transaction_day'] = df['purchase_time'].dt.day
    df['transaction_month'] = df['purchase_time'].dt.month
    
    # Plot distribution of transactions by hour
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    df.groupby('transaction_hour')['class'].mean().plot(kind='bar')
    plt.title('Distribution of Transactions by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Likelihood of Fraud')
    
    # Plot distribution of transactions by day
    plt.subplot(1, 2, 2)
    df.groupby('transaction_day')['class'].mean().plot(kind='line')
    plt.title('Distribution of Transactions by Day')
    plt.xlabel('Day')
    plt.ylabel('Likelihood of Fraud')
    plt.tight_layout()
    plt.show()
    
    # Plot distribution of transactions by month
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    df.groupby('transaction_month')['class'].mean().plot(kind='line')
    plt.title('Distribution of Transactions by Month')
    plt.xlabel('Month')
    plt.ylabel('Likelihood of Fraud')
    
    # Plot relationship between transaction time and fraud
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df['purchase_time'], y=df['class'])
    plt.title('Relationship between Transaction Time and Fraud')
    plt.xlabel('Transaction Time')
    plt.ylabel('Fraud Status')
    plt.tight_layout()
    plt.show()
    
    # Identify seasonality, trends, and change points
    transaction_series = df.set_index('purchase_time').resample('D')['class'].mean().dropna()
    result = seasonal_decompose(transaction_series, model='additive', period=30)  # Assuming monthly seasonality (30 days)
    
    plt.figure(figsize=(12, 8))
    result.plot()
    plt.suptitle('Time Series Decomposition', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Test for stationarity
    adf_result = adfuller(transaction_series)
    print(f"ADF Statistic: {adf_result[0]:.2f}")
    print(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")
