import pandas as pd
import xxhash
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()


    def create_features(self):
        """
        Create Timestamp-based Features, Transaction Value-based Features, Device-based Features, User-based Features, Geolocation-based Features, Behavioral-based Features, Transaction Frequency, and Transaction Velocity from the given DataFrame.

        Parameters:
        self (FeatureEngineering): The FeatureEngineering object.

        Returns:
        pandas.DataFrame: The original DataFrame with the new features added.
        """

        # Timestamp-based Features
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.days

        # Transaction Value-based Features
        self.df['purchase_value_log'] = np.log(self.df['purchase_value'])
        self.df['purchase_value_percentile'] = self.df['purchase_value'].rank(method='dense', pct=True)

        # Device-based Feature
        self.df['device_reuse'] = self.df.groupby('device_id')['device_id'].transform('count') > 1

        # User-based Features
        self.df['num_transactions'] = self.df.groupby('user_id')['user_id'].transform('count')
        self.df['avg_purchase_value'] = self.df.groupby('user_id')['purchase_value'].transform('mean')


        # Geolocation-based Features
        self.df['ip_country'] = self.df['country']
        self.df['ip_location_change'] = self.df.groupby('user_id')['ip_country'].shift(1) != self.df['ip_country']

        # Behavioral-based Features
        self.df['source_change'] = self.df.groupby('user_id')['source'].shift(1) != self.df['source']
        self.df['browser_change'] = self.df.groupby('user_id')['browser'].shift(1) != self.df['browser']

  
        # Calculate user transaction count
        user_transaction_count = self.df.groupby('user_id').size().reset_index(name='user_transaction_count')
        self.df = pd.merge(self.df, user_transaction_count, on='user_id', how='left')

        # Calculate transaction velocity transactions per day for users
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        self.df['days_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.days
        self.df['days_since_signup'] = self.df['days_since_signup'].replace(0, 1)  

        self.df['user_transaction_velocity'] = self.df['user_transaction_count'] / self.df['days_since_signup']

        # Dropping intermediate columns
        self.df.drop(columns=['days_since_signup'], inplace=True)

        return self.df

    def encode_categorical_features(self):
        """
        Encode categorical features using specified encoding methods.
        """
        if 'country' in self.df.columns:
            # Hashing Encoding for 
            self.df['ip_country_hash'] = self.df['ip_country'].apply(
                lambda x: np.uint64(xxhash.xxh64(str(x).encode('utf-8')).intdigest())
            )

        if 'source' in self.df.columns:
            # Frequency Encoding for source
            source_freq = self.df['source'].value_counts(normalize=True)
            self.df['source_encoded'] = self.df['source'].map(source_freq)

        if 'browser' in self.df.columns:
            # Frequency Encoding for browser
            browser_freq = self.df['browser'].value_counts(normalize=True)
            self.df['browser_encoded'] = self.df['browser'].map(browser_freq)

        # Label Encoding for boolean features
        boolean_features = ['device_reuse', 'ip_location_change', 'source_change', 'browser_change', 'sex']
        if all(feature in self.df.columns for feature in boolean_features):
            encoder_bool = LabelEncoder()
            for feature in boolean_features:
                self.df[f"{feature}_encoded"] = encoder_bool.fit_transform(self.df[feature])

        # Drop original categorical columns
        self.df.drop(columns=['device_reuse', 'ip_location_change', 'source_change', 'browser_change','device_id','country', 'ip_country', 'source', 'browser', 'sex', 'signup_time', 'purchase_time'], inplace=True, errors='ignore')

    def scale_features(self):
        """
        Scale numeric features using StandardScaler, while keeping the encoded boolean features and the target feature ('class') unscaled.
        """
        # Select the numeric columns, excluding the encoded boolean features and the target feature ('class')
        numeric_columns = [col for col in self.df.select_dtypes(include=['float64', 'int32', 'int64', 'uint64']).columns
                        if col not in ['device_reuse_encoded', 'ip_location_change_encoded', 'source_change_encoded',
                                        'browser_change_encoded', 'sex_encoded', 'class']]
        
        # Scale the numeric features
        self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])
        
        return self.df

    def undersample(self):
        """
        Perform undersampling to balance the dataset.
        """
        if 'class' in self.df.columns:
            normal = self.df[self.df['class'] == 0.0]
            fraud = self.df[self.df['class'] == 1.0]
            normal_sample = normal.sample(n=len(fraud), random_state=42)
            self.df = pd.concat([normal_sample, fraud], ignore_index=True)

    def preprocess(self):
        """
        Perform full preprocessing pipeline.
        """
        self.create_features()
        self.encode_categorical_features()
        self.undersample()
        self.df = self.scale_features()
        return self.df
