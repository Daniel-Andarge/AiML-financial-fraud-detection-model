import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_datasets():
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_data = pd.read_csv('IpAddress_to_Country.csv')
    credit_card_data = pd.read_csv('creditcard.csv')
    return fraud_data, ip_data, credit_card_data

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
    fraud_data.fillna(method='ffill', inplace=True)
    ip_data.fillna(method='ffill', inplace=True)
    credit_card_data.fillna(method='ffill', inplace=True)
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

def ip_to_int(ip):
    parts = ip.split('.')
    return int(parts[0])*256**3 + int(parts[1])*256**2 + int(parts[2])*256 + int(parts[3])

def merge_ip_data(fraud_data, ip_data):
    fraud_data['ip_address'] = fraud_data['ip_address'].apply(ip_to_int)
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].apply(ip_to_int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].apply(ip_to_int)

    fraud_data = fraud_data.sort_values('ip_address').reset_index(drop=True)
    ip_data = ip_data.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    fraud_data['country'] = np.nan
    i = 0
    for idx, row in fraud_data.iterrows():
        while i < len(ip_data) and row['ip_address'] > ip_data.iloc[i]['upper_bound_ip_address']:
            i += 1
        if i < len(ip_data) and ip_data.iloc[i]['lower_bound_ip_address'] <= row['ip_address'] <= ip_data.iloc[i]['upper_bound_ip_address']:
            fraud_data.at[idx, 'country'] = ip_data.iloc[i]['country']
    return fraud_data

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

def encode_categorical_features(fraud_data):
    encoder = OneHotEncoder(sparse=False)
    categorical_features = ['source', 'browser', 'sex', 'country']
    encoded_features = encoder.fit_transform(fraud_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    fraud_data = fraud_data.join(encoded_df)
    fraud_data.drop(columns=categorical_features, inplace=True)
    return fraud_data

def save_processed_data(fraud_data, filename='Processed_Fraud_Data.csv'):
    fraud_data.to_csv(filename, index=False)

# Main execution flow
def main():
    fraud_data, ip_data, credit_card_data = load_datasets()
    fraud_data, ip_data, credit_card_data = handle_missing_values(fraud_data, ip_data, credit_card_data)
    fraud_data, ip_data, credit_card_data = remove_duplicates(fraud_data, ip_data, credit_card_data)
    fraud_data = correct_data_types(fraud_data)
    fraud_data = merge_ip_data(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    fraud_data = normalize_and_scale(fraud_data)
    fraud_data = encode_categorical_features(fraud_data)
    save_processed_data(fraud_data)

# RunMain function
if __name__ == '__main__':
    main()
