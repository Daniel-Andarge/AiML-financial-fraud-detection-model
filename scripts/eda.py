import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

    
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

def load_processed_data(file_path):
    """
    Load processed fraud dataset from file.

    Parameters:
    file_path (str): Path to the processed dataset CSV file.

    Returns:
    pd.DataFrame: Processed fraud dataset.
    """
    return pd.read_csv(file_path)

def summary_statistics(dataframe, numeric_variables):
    """
    Display summary statistics for numeric variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    numeric_variables (list): List of numeric variable names.

    Returns:
    None
    """
    print("Summary Statistics:")
    print(dataframe[numeric_variables].describe())

def plot_histograms(dataframe, numeric_variables):
    """
    Plot histograms for numeric variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    numeric_variables (list): List of numeric variable names.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_variables, start=1):
        plt.subplot(2, 2, i)
        sns.histplot(dataframe[col], kde=True)
        plt.title(col)
        plt.xlabel('')
    plt.tight_layout()
    plt.show()

def plot_boxplots(dataframe, numeric_variables):
    """
    Plot box plots for numeric variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    numeric_variables (list): List of numeric variable names.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_variables, start=1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=dataframe[col])
        plt.title(col)
        plt.xlabel('')
    plt.tight_layout()
    plt.show()

def frequency_counts(dataframe, categorical_variables):
    """
    Display frequency counts for categorical variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    categorical_variables (list): List of categorical variable names.

    Returns:
    None
    """
    print("\nFrequency Counts for Categorical Variables:")
    for col in categorical_variables:
        print(dataframe[col].value_counts(normalize=True))
        print()

def plot_countplots(dataframe, categorical_variables):
    """
    Plot bar plots for categorical variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    categorical_variables (list): List of categorical variable names.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(categorical_variables, start=1):
        plt.subplot(2, 2, i)
        sns.countplot(x=col, data=dataframe, palette='viridis')
        plt.title(col)
        plt.xlabel('')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_scatterplot(dataframe, x_variable, y_variable):
    """
    Plot scatter plot for two numeric variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    x_variable (str): Name of the x-axis variable (numeric).
    y_variable (str): Name of the y-axis variable (numeric).

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_variable, y=y_variable, data=dataframe, alpha=0.5)
    plt.title(f'{y_variable} vs {x_variable}')
    plt.show()

def plot_heatmap(dataframe, numeric_variables):
    """
    Plot correlation heatmap for numeric variables.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    numeric_variables (list): List of numeric variable names.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataframe[numeric_variables].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.show()

def plot_boxplot_by_category(dataframe, x_variable, y_variable):
    """
    Plot box plot of a numeric variable by a categorical variable.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    x_variable (str): Name of the categorical variable for x-axis.
    y_variable (str): Name of the numeric variable for y-axis.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x_variable, y=y_variable, data=dataframe, palette='Set2')
    plt.title(f'{y_variable} by {x_variable}')
    plt.show()

def plot_stacked_barplot(dataframe, x_variable, hue_variable, hue_categories):
    """
    Plot stacked bar plot of a categorical variable by another categorical variable.

    Parameters:
    dataframe (pd.DataFrame): Processed fraud dataset.
    x_variable (str): Name of the x-axis categorical variable.
    hue_variable (str): Name of the hue variable for stacking bars.
    hue_categories (list): List of categories in the hue variable.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    stacked_data = dataframe.groupby(x_variable)[hue_variable].value_counts(normalize=True).unstack()
    stacked_data[hue_categories].plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Fraud Class Distribution by {x_variable}')
    plt.xlabel(x_variable)
    plt.ylabel('Proportion')
    plt.legend(title=hue_variable, labels=hue_categories, loc='upper right')
    plt.show()



# PLOTLY
def exploratory_data_analysis(df):
    """
    Performs Exploratory Data Analysis (EDA) on the given dataset.
    
    Args:
        df (pandas.DataFrame): The input dataset.
    """
    # Univariate Analysis
    
    # Numerical Features
    num_features = ['user_id', 'purchase_value', 'age', 'ip_address', 'class']
    for feature in num_features:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[feature]))
        fig.update_layout(title=f'Distribution of {feature}')
        fig.show()
    
    # Categorical Features
    cat_features = ['device_id', 'source', 'browser', 'sex', 'country']
    for feature in cat_features:
        fig = px.bar(df[feature].value_counts(), x=df[feature].value_counts().index, y=df[feature].value_counts())
        fig.update_layout(title=f'Distribution of {feature}')
        fig.show()
    
    # Bivariate Analysis
    
    # Relationship between Numerical Features
    fig = ff.create_scatterplotmatrix(df[num_features], diag='histogram')
    fig.update_layout(title='Relationship between Numerical Features')
    fig.show()
    
    # Relationship between Numerical and Categorical Features
    for num_feature in num_features:
        for cat_feature in cat_features:
            fig = px.box(df, x=cat_feature, y=num_feature)
            fig.update_layout(title=f'Relationship between {cat_feature} and {num_feature}')
            fig.show()
    
    # Relationship between Target Variable and Other Features
    fig = go.Figure(data=go.Heatmap(
        x=df.columns,
        y=df.columns,
        z=df.corr(),
        colorscale='YlOrRd'
    ))
    fig.update_layout(title='Correlation Heatmap')
    fig.show()



def fraud_class_distribution(df):
    """
    Visualizes the distribution of the 'class' (target variable) across different features.
    
    Args:
        df (pandas.DataFrame): The input dataset.
    """
    # Fraud class distribution by numerical features
    num_features = ['age']
    for feature in num_features:
        fig = px.histogram(df, x=feature, color='class', barmode='group')
        fig.update_layout(title=f'Fraud Class Distribution by {feature}')
        fig.show()
    
    # Fraud class distribution by categorical features
    cat_features = ['source', 'browser', 'sex', 'country']
    for feature in cat_features:
        fig = px.bar(df.groupby(['class', feature]).size().reset_index(name='count'), x=feature, y='count', color='class')
        fig.update_layout(title=f'Fraud Class Distribution by {feature}')
        fig.show()