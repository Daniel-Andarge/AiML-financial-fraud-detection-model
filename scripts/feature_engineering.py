import pandas as pd
import hashlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import category_encoders as ce
import xxhash
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()

    def create_user_profile_features(self):
        """
        Create hashed features for device_id and combined ip_address with device_id.
        """
        # Hash device_id to create numerical representation
        self.df['device_id_hash'] = self.df['device_id'].apply(
            lambda x: xxhash.xxh64(str(x).encode('utf-8')).intdigest()
        )

        # Create ip_device_hash feature
        self.df['ip_device_hash'] = self.df.apply(
            lambda row: xxhash.xxh64((str(row['ip_address']) + str(row['device_id_hash'])).encode('utf-8')).intdigest(),
            axis=1
        )

    def transaction_frequency_velocity(self):
        """
        Calculate transaction frequency and velocity (transactions per day) per user.
        """
        # Calculate transaction frequency per user
        user_transaction_count = self.df.groupby('user_id').size().reset_index(name='user_transaction_count')
        self.df = pd.merge(self.df, user_transaction_count, on='user_id', how='left')

        # Calculate transaction velocity (transactions per day) for users
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        self.df['days_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.days
        self.df['days_since_signup'] = self.df['days_since_signup'].replace(0, 1)  # To avoid division by zero

        self.df['user_transaction_velocity'] = self.df['user_transaction_count'] / self.df['days_since_signup']

        # Dropping intermediate columns
        self.df.drop(columns=['days_since_signup'], inplace=True)

    def time_based_features(self):
        """
        Extract time-based features from signup and purchase times.
        """
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])

        # Extract hour of day and day of week for purchase_time
        self.df['purchase_hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['purchase_day_of_week'] = self.df['purchase_time'].dt.dayofweek

        # Extract hour of day and day of week for signup_time
        self.df['signup_hour_of_day'] = self.df['signup_time'].dt.hour
        self.df['signup_day_of_week'] = self.df['signup_time'].dt.dayofweek

    def encode_categorical_features(self):
        """
        Encode categorical features using specified encoding methods.
        """
        # Hashing Encoding for country_id_hash
        self.df['country_id_hash'] = self.df['country'].apply(
            lambda x: np.uint64(xxhash.xxh64(str(x).encode('utf-8')).intdigest())
        )

        # Frequency Encoding for source
        source_freq = self.df['source'].value_counts(normalize=True)
        self.df['source_encoded'] = self.df['source'].map(source_freq)

        # Frequency Encoding for browser
        browser_freq = self.df['browser'].value_counts(normalize=True)
        self.df['browser_encoded'] = self.df['browser'].map(browser_freq)

        # Label Encoding for sex
        encoder_sex = LabelEncoder()
        self.df['sex_encoded'] = encoder_sex.fit_transform(self.df['sex'])

        # Drop original categorical columns
        self.df.drop(columns=['device_id', 'country', 'source', 'browser', 'sex', 'signup_time', 'purchase_time'], inplace=True)

    def scale_features(self):
        """
        Scale numeric features using StandardScaler.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int32', 'int64', 'uint64']).columns
        self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])

    def normalize_features(self):
        """
        Normalize numeric features using MinMaxScaler.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int32', 'int64', 'uint64']).columns
        self.df[numeric_columns] = self.normalizer.fit_transform(self.df[numeric_columns])
    
    def undersample(self):
        """
        Perform undersampling to balance the dataset.
        """
        normal = self.df[self.df['class'] == 0.0]
        fraud = self.df[self.df['class'] == 1.0]
        normal_sample = normal.sample(n=len(fraud), random_state=42)
        new_data = pd.concat([normal_sample, fraud], ignore_index=True)
        self.df = new_data

    def preprocess(self):
        """
        Perform full preprocessing pipeline.
        """
        self.create_user_profile_features()
        self.transaction_frequency_velocity()
        self.time_based_features()
        self.encode_categorical_features()
        self.scale_features()
        self.normalize_features()
        self.undersample()
        return self.df
