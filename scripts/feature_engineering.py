import pandas as pd
import xxhash
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class FeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()

    def create_user_profile_features(self):
        """
        Create hashed features for device_id and combined ip_address with device_id.
        """
        if 'device_id' in self.df.columns and 'ip_address' in self.df.columns:
            self.df['device_id_hash'] = self.df['device_id'].apply(
                lambda x: xxhash.xxh64(str(x).encode('utf-8')).intdigest()
            )

            self.df['ip_device_hash'] = self.df.apply(
                lambda row: xxhash.xxh64((str(row['ip_address']) + str(row['device_id_hash'])).encode('utf-8')).intdigest(),
                axis=1
            )

    def transaction_frequency_velocity(self):
        """
        Calculate transaction frequency and velocity (transactions per day) per user.
        """
        if 'user_id' in self.df.columns and 'signup_time' in self.df.columns and 'purchase_time' in self.df.columns:
            # Calculate transaction frequency per user
            user_transaction_count = self.df.groupby('user_id').size().reset_index(name='user_transaction_count')
            self.df = pd.merge(self.df, user_transaction_count, on='user_id', how='left')

            # Calculate transaction velocity (transactions per day) for users
            self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
            self.df['days_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.days
            self.df['days_since_signup'] = self.df['days_since_signup'].replace(0, 1)  # Avoid division by zero

            self.df['user_transaction_velocity'] = self.df['user_transaction_count'] / self.df['days_since_signup']

            # Dropping intermediate columns
            self.df.drop(columns=['days_since_signup'], inplace=True)

    def time_based_features(self):
        """
        Extract time-based features from signup and purchase times.
        """
        if 'signup_time' in self.df.columns and 'purchase_time' in self.df.columns:
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
        if 'country' in self.df.columns:
            # Hashing Encoding for country_id_hash
            self.df['country_id_hash'] = self.df['country'].apply(
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

        if 'sex' in self.df.columns:
            # Label Encoding for sex
            encoder_sex = LabelEncoder()
            self.df['sex_encoded'] = encoder_sex.fit_transform(self.df['sex'])

        # Drop original categorical columns
        self.df.drop(columns=['device_id', 'country', 'source', 'browser', 'sex', 'signup_time', 'purchase_time'], inplace=True, errors='ignore')

    def scale_features(self):
        """
        Scale numeric features using StandardScaler, while keeping the target feature ('class') unscaled.
        """
        # Select the numeric columns, excluding the target feature ('class')
        numeric_columns = [col for col in self.df.select_dtypes(include=['float64', 'int32', 'int64', 'uint64']).columns if col != 'class']
        
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
        self.create_user_profile_features()
        self.transaction_frequency_velocity()
        self.time_based_features()
        self.encode_categorical_features()
        self.undersample()
        self.df = self.scale_features()
        return self.df
