import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def find_missing_values(fraud_data, ip_data, credit_card_data):
    fraud_missing = fraud_data.isnull().sum()
    ip_missing = ip_data.isnull().sum()
    credit_card_missing = credit_card_data.isnull().sum()
    return fraud_missing, ip_missing, credit_card_missing

def find_duplicates(fraud_data, ip_data, credit_card_data):
    fraud_duplicates = fraud_data.duplicated().sum()
    ip_duplicates = ip_data.duplicated().sum()
    credit_card_duplicates = credit_card_data.duplicated().sum()
    return fraud_duplicates, ip_duplicates, credit_card_duplicates

def handle_missing_values(fraud_data, ip_data, credit_card_data):
    fraud_data = fraud_data.ffill()
    ip_data = ip_data.ffill()
    credit_card_data = credit_card_data.ffill()
    return fraud_data, ip_data, credit_card_data


def remove_duplicates(fraud_data, ip_data, credit_card_data):
    fraud_data.drop_duplicates(inplace=True)
    ip_data.drop_duplicates(inplace=True)
    credit_card_data.drop_duplicates(inplace=True)
    return fraud_data, ip_data, credit_card_data

def correct_data_types(fraud_data):
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    return fraud_data



def convert_ip_to_int(df, ip_column_name):
    """
    Convert IP addresses from float to integer format.

    Parameters:
    df (pd.DataFrame): DataFrame containing the IP addresses.
    ip_column_name (str): Name of the column containing the IP addresses.

    Returns:
    pd.DataFrame: DataFrame with IP addresses converted to integer format.
    """
    df[ip_column_name] = df[ip_column_name].astype(int)
    return df

def map_ip_to_country(fraud_df, ip_to_country_df, ip_column_name):
    """
    Map IP addresses to countries based on IP ranges.

    Parameters:
    fraud_df (pd.DataFrame): DataFrame containing fraud data with IP addresses.
    ip_to_country_df (pd.DataFrame): DataFrame containing IP address ranges and corresponding countries.
    ip_column_name (str): Name of the column containing the IP addresses in the fraud data.

    Returns:
    pd.DataFrame: DataFrame with an additional column for the country.
    """
    def ip_to_country(ip):
        row = ip_to_country_df[(ip_to_country_df['lower_bound_ip_address'] <= ip) & (ip_to_country_df['upper_bound_ip_address'] >= ip)]
        if not row.empty:
            return row.iloc[0]['country']
        else:
            return 'Unknown'

    fraud_df['country'] = fraud_df[ip_column_name].apply(ip_to_country)
    return fraud_df

# Convert IP to int and Merge
def convert_ip_and_merge(fraud_data, ip_to_country):
    """
    Load fraud data and IP to country mapping data,
    convert IP addresses to integer format, and map IP addresses to countries.

    Parameters:
    fraud_data_path (str): File path to Fraud_Data.csv.
    ip_to_country_path (str): File path to IpAddress_to_Country.csv.

    Returns:
    pd.DataFrame: Processed fraud data with country information.
    """


    # Convert IP addresses to integer format in both datasets
    fraud_data = convert_ip_to_int(fraud_data, 'ip_address')
    ip_to_country = convert_ip_to_int(ip_to_country, 'lower_bound_ip_address')
    ip_to_country = convert_ip_to_int(ip_to_country, 'upper_bound_ip_address')

    # Map IP addresses to countries
    fraud_data = map_ip_to_country(fraud_data, ip_to_country, 'ip_address')

    return fraud_data



def remove_outliers(df, class_col='class'):
    """
    Removes outliers from a DataFrame based on the IQR method, except for the "class" feature.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    class_col (str): The name of the "class" feature column. Default is 'class'.
    
    Returns:
    pandas.DataFrame: The DataFrame with outliers removed.
    """
    # Get the numeric columns, excluding the "class" feature
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'uint64']).columns
    numeric_cols = numeric_cols[~numeric_cols.isin([class_col])]
    
    # Create a new DataFrame to store the cleaned data
    cleaned_df = df.copy()
    
    # Iterate over the numeric columns and remove outliers
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Remove outliers and update the cleaned DataFrame
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df








