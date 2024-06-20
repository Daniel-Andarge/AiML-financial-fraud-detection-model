import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def feature_engineering(fraud_data):
    # Transaction frequency and velocity
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['device_transaction_count'] = fraud_data.groupby('device_id')['device_id'].transform('count')
    
    # Time-Based Features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    return fraud_data

def normalize_and_scale(fraud_data):
    scaler = StandardScaler()
    scaled_features = ['purchase_value', 'age', 'transaction_count', 'device_transaction_count']
    fraud_data[scaled_features] = scaler.fit_transform(fraud_data[scaled_features])
    return fraud_data



def encode_categorical_features(df):
    """
    Encode categorical features using specified encoding methods.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing categorical features.

    Returns:
    pd.DataFrame: DataFrame with categorical features encoded.
    """
    # Hashing Encoding for device_id (high cardinality)
    encoder_device = ce.HashingEncoder(n_components=8, cols=['device_id'])
    df_device_encoded = encoder_device.fit_transform(df)

    # Target Encoding for country (high cardinality)
    target_encoder_country = ce.TargetEncoder(cols='country')
    df['country_encoded'] = target_encoder_country.fit_transform(df['country'], df['class'])

    # Frequency Encoding for source
    source_freq = df['source'].value_counts(normalize=True)
    df['source_encoded'] = df['source'].map(source_freq)

    # Frequency Encoding for browser
    browser_freq = df['browser'].value_counts(normalize=True)
    df['browser_encoded'] = df['browser'].map(browser_freq)

    # Label Encoding for sex
    encoder_sex = LabelEncoder()
    df['sex_encoded'] = encoder_sex.fit_transform(df['sex'])

    # Drop original categorical columns
    df = df.drop(columns=['device_id', 'country', 'source', 'browser', 'sex'])

    # Concatenate encoded columns with original DataFrame
    df_encoded = pd.concat([df_device_encoded, df], axis=1)

    return df_encoded