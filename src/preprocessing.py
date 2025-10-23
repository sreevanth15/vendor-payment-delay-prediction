"""
Data Preprocessing Module for Vendor Payment Delay Prediction

This module handles data cleaning, feature engineering, and preprocessing
for the payment delay prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert date columns
        date_columns = ['invoice_date', 'due_date', 'actual_payment_date']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        print(f"Loaded {len(df)} records")
        return df
    
    def create_features(self, df):
        """Create additional features for the model"""
        df = df.copy()
        
        # Convert date columns to datetime
        date_columns = ['invoice_date', 'due_date', 'actual_payment_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Date-based features
        df['invoice_month'] = df['invoice_date'].dt.month
        df['invoice_day_of_week'] = df['invoice_date'].dt.dayofweek
        df['invoice_quarter'] = df['invoice_date'].dt.quarter
        df['is_month_end'] = (df['invoice_date'].dt.day > 25).astype(int)
        df['is_weekend'] = (df['invoice_date'].dt.dayofweek >= 5).astype(int)
        
        # Calculate days_until_due if missing
        if 'days_until_due' not in df.columns:
            if 'due_date' in df.columns and 'invoice_date' in df.columns:
                df['days_until_due'] = (df['due_date'] - df['invoice_date']).dt.days
            else:
                df['days_until_due'] = df['payment_terms'] if 'payment_terms' in df.columns else 30
        
        # Payment terms categories
        df['payment_terms_category'] = pd.cut(df['payment_terms'], 
                                            bins=[0, 15, 30, 45, 100], 
                                            labels=['Short', 'Standard', 'Extended', 'Long'])
        
        # Amount-based features
        df['log_invoice_amount'] = np.log1p(df['invoice_amount'])
        df['amount_category'] = pd.cut(df['invoice_amount'], 
                                     bins=[0, 10000, 50000, 100000, np.inf], 
                                     labels=['Small', 'Medium', 'Large', 'XLarge'])
        
        # Vendor performance features
        df['vendor_reliability_score'] = np.where(df['vendor_past_delays'] == 0, 1.0,
                                                1.0 / (1 + df['vendor_past_delays']))
        
        # Cash flow ratios - handle missing column
        if 'month_cash_available' not in df.columns:
            # Estimate based on cash flow level
            cash_mapping = {'Low': 1500000, 'Medium': 4000000, 'High': 8000000}
            df['month_cash_available'] = df['cash_flow_level'].map(cash_mapping)
        
        df['cash_to_invoice_ratio'] = df['month_cash_available'] / df['invoice_amount']
        df['cash_availability_category'] = pd.cut(df['month_cash_available'], 
                                                bins=[0, 2000000, 5000000, 10000000, np.inf], 
                                                labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Seasonal features
        df['is_year_end'] = ((df['invoice_month'] == 12) | (df['invoice_month'] == 1)).astype(int)
        df['is_mid_year'] = ((df['invoice_month'] >= 6) & (df['invoice_month'] <= 8)).astype(int)
        
        # Vendor relationship features - handle missing column
        if 'vendor_relationship_years' not in df.columns:
            if 'vendor_relationship_length' in df.columns:
                df['vendor_relationship_years'] = df['vendor_relationship_length'] / 12
            else:
                df['vendor_relationship_years'] = 2  # Default
        
        df['relationship_category'] = pd.cut(df['vendor_relationship_years'], 
                                           bins=[0, 1, 3, 5, np.inf], 
                                           labels=['New', 'Developing', 'Established', 'Long_term'])
        
        # Payment frequency features - handle missing column
        if 'payment_frequency' not in df.columns:
            df['payment_frequency'] = 3  # Default
        
        df['high_frequency_vendor'] = (df['payment_frequency'] > 5).astype(int)
        
        # Economic features - handle missing column
        if 'economic_indicator' not in df.columns:
            df['economic_indicator'] = 90  # Default neutral value
        
        df['economic_stress'] = (df['economic_indicator'] < 85).astype(int)
        
        # Has dispute - handle missing column
        if 'has_dispute' not in df.columns:
            df['has_dispute'] = 0  # Default no dispute
        
        # Risk score combination
        df['vendor_risk_score'] = (
            (df['vendor_past_delays'] > 2).astype(int) * 0.3 +
            (df['cash_flow_level'] == 'Low').astype(int) * 0.3 +
            df['has_dispute'] * 0.4
        )
        
        print(f"Created features. Dataset now has {len(df.columns)} columns")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        # Define categorical columns
        categorical_columns = [
            'vendor_category', 'cash_flow_level', 'payment_terms_category',
            'amount_category', 'cash_availability_category', 'relationship_category'
        ]
        
        # One-hot encode categorical variables
        encoded_dfs = []
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    encoded = pd.get_dummies(df[col], prefix=col, drop_first=True)
                else:
                    # For prediction, we need to ensure same columns
                    encoded = pd.get_dummies(df[col], prefix=col, drop_first=True)
                encoded_dfs.append(encoded)
                df = df.drop(col, axis=1)
        
        # Concatenate all encoded features
        if encoded_dfs:
            encoded_df = pd.concat(encoded_dfs, axis=1)
            df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        # Define features to use
        feature_columns = [
            # Numerical features
            'payment_terms', 'log_invoice_amount', 'vendor_past_delays',
            'vendor_relationship_years', 'payment_frequency', 'month_cash_available',
            'economic_indicator', 'cash_to_invoice_ratio', 'vendor_reliability_score',
            'vendor_risk_score',
            
            # Date features
            'invoice_month', 'invoice_day_of_week', 'invoice_quarter',
            'is_month_end', 'is_weekend', 'is_year_end', 'is_mid_year',
            
            # Binary features
            'has_dispute', 'high_frequency_vendor', 'economic_stress'
        ]
        
        # Add one-hot encoded features
        categorical_prefixes = [
            'vendor_category_', 'cash_flow_level_', 'payment_terms_category_',
            'amount_category_', 'cash_availability_category_', 'relationship_category_'
        ]
        
        for prefix in categorical_prefixes:
            categorical_features = [col for col in df.columns if col.startswith(prefix)]
            feature_columns.extend(categorical_features)
        
        # Select only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features]
    
    def prepare_data(self, df, target_column='is_delayed', test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        print("Starting data preparation...")
        
        # Create features
        df_processed = self.create_features(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_processed)
        
        # Select features
        X = self.select_features(df_encoded)
        y = df_processed[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"Selected {len(self.feature_names)} features")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        print(f"Training set: {len(X_train_scaled)} samples")
        print(f"Test set: {len(X_test_scaled)} samples")
        print(f"Class distribution - Delayed: {y_train.mean():.2%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_importance_data(self, df):
        """Get processed data for feature importance analysis"""
        df_processed = self.create_features(df)
        df_encoded = self.encode_categorical_features(df_processed)
        X = self.select_features(df_encoded)
        y = df_processed['is_delayed']
        
        return X, y

def main():
    """Main preprocessing function"""
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('/Users/sreevanthsv/Desktop/DWDM project/data/raw/vendor_payments.csv')
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': preprocessor.feature_names
    }
    
    # Save as pickle for later use
    import joblib
    joblib.dump(processed_data, '/Users/sreevanthsv/Desktop/DWDM project/data/processed/processed_data.pkl')
    joblib.dump(preprocessor, '/Users/sreevanthsv/Desktop/DWDM project/data/processed/preprocessor.pkl')
    
    print("Data preprocessing completed and saved!")
    
    return processed_data

if __name__ == "__main__":
    main()
