"""
Demo Script: Predict Payment Delays for New Invoices

This script demonstrates how to use the trained model to predict
delays for new vendor invoices.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_invoices():
    """Create sample invoices for prediction"""
    
    sample_data = {
        'vendor_id': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'invoice_id': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005'],
        'vendor_category': ['Technology', 'Manufacturing', 'Services', 'Utilities', 'Office Supplies'],
        'payment_terms': [30, 45, 15, 30, 15],
        'invoice_amount': [75000, 150000, 25000, 8000, 5000],
        'vendor_past_delays': [0, 5, 2, 1, 0],
        'vendor_relationship_years': [3.5, 1.2, 5.0, 8.0, 2.0],
        'payment_frequency': [2, 8, 4, 1, 3],
        'has_dispute': [0, 1, 0, 0, 0],
        'economic_indicator': [95, 78, 102, 98, 88],
        'cash_flow_level': ['Medium', 'Low', 'High', 'High', 'Medium'],
        'quarter': ['Q3', 'Q3', 'Q3', 'Q3', 'Q3'],
        'invoice_month': [8, 8, 8, 8, 8]
    }
    
    return pd.DataFrame(sample_data)

def predict_payment_delays():
    """Predict payment delays for sample invoices"""
    
    print("=" * 60)
    print("VENDOR PAYMENT DELAY PREDICTION DEMO")
    print("=" * 60)
    
    # Load training data for encoding reference
    print("\n1. Loading training data for preprocessing...")
    train_data_path = '/Users/sreevanthsv/Desktop/DWDM project/data/raw/vendor_payments.csv'
    train_df = pd.read_csv(train_data_path)
    
    # Create sample invoices to predict
    print("\n2. Creating sample invoices for prediction...")
    new_invoices = create_sample_invoices()
    print("Sample invoices:")
    display_df = new_invoices[['vendor_id', 'vendor_category', 'invoice_amount', 'vendor_past_delays', 'has_dispute']].copy()
    for i, row in display_df.iterrows():
        print(f"  {row['vendor_id']}: {row['vendor_category']}, ${row['invoice_amount']:,}, "
              f"{row['vendor_past_delays']} past delays, dispute: {'Yes' if row['has_dispute'] else 'No'}")
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    
    # Combine for consistent encoding
    combined_df = pd.concat([train_df[['vendor_category', 'cash_flow_level']], 
                            new_invoices[['vendor_category', 'cash_flow_level']]], 
                           ignore_index=True)
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_cash_flow = LabelEncoder()
    
    combined_df['vendor_category_encoded'] = le_category.fit_transform(combined_df['vendor_category'])
    combined_df['cash_flow_encoded'] = le_cash_flow.fit_transform(combined_df['cash_flow_level'])
    
    # Get encodings for new data
    new_invoices['vendor_category_encoded'] = combined_df['vendor_category_encoded'].iloc[-len(new_invoices):].values
    new_invoices['cash_flow_encoded'] = combined_df['cash_flow_encoded'].iloc[-len(new_invoices):].values
    new_invoices['log_amount'] = np.log1p(new_invoices['invoice_amount'])
    
    # Prepare features
    feature_cols = [
        'payment_terms', 'log_amount', 'vendor_past_delays',
        'vendor_relationship_years', 'payment_frequency', 
        'has_dispute', 'economic_indicator', 'invoice_month',
        'vendor_category_encoded', 'cash_flow_encoded'
    ]
    
    X_new = new_invoices[feature_cols]
    
    # Train quick models on full dataset
    print("\n4. Training models on historical data...")
    
    # Prepare training data
    train_df['log_amount'] = np.log1p(train_df['invoice_amount'])
    train_df['invoice_month'] = pd.to_datetime(train_df['invoice_date']).dt.month
    train_df['vendor_category_encoded'] = le_category.transform(train_df['vendor_category'])
    train_df['cash_flow_encoded'] = le_cash_flow.transform(train_df['cash_flow_level'])
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_delayed']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_new_scaled = scaler.transform(X_new)
    
    # Train models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    print("\n5. Making predictions...")
    
    lr_proba = lr_model.predict_proba(X_new_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_new)[:, 1]
    
    # Average ensemble
    ensemble_proba = (lr_proba + rf_proba) / 2
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # Display results
    print("\n6. PREDICTION RESULTS")
    print("-" * 40)
    
    results_df = new_invoices[['vendor_id', 'vendor_category', 'invoice_amount', 'vendor_past_delays']].copy()
    results_df['delay_probability'] = ensemble_proba
    results_df['prediction'] = ensemble_pred
    results_df['risk_level'] = pd.cut(ensemble_proba, 
                                     bins=[0, 0.3, 0.7, 1.0], 
                                     labels=['LOW', 'MEDIUM', 'HIGH'])
    
    for i, row in results_df.iterrows():
        status = "🔴 DELAYED" if row['prediction'] else "🟢 ON-TIME"
        risk = row['risk_level']
        prob = row['delay_probability']
        
        print(f"\n📋 {row['vendor_id']} ({row['vendor_category']}):")
        print(f"   Amount: ${row['invoice_amount']:,}")
        print(f"   Past delays: {row['vendor_past_delays']}")
        print(f"   Delay probability: {prob:.3f}")
        print(f"   Prediction: {status}")
        print(f"   Risk level: {risk}")
        
        if row['prediction']:
            print(f"   ⚠️  ALERT: High risk of payment delay!")
            print(f"   💡 Recommendation: Contact vendor proactively")
        else:
            print(f"   ✅ Expected on-time payment")
    
    # Summary
    print("\n7. SUMMARY & RECOMMENDATIONS")
    print("-" * 40)
    
    high_risk_count = (results_df['risk_level'] == 'HIGH').sum()
    medium_risk_count = (results_df['risk_level'] == 'MEDIUM').sum()
    total_amount_at_risk = results_df[results_df['prediction'] == 1]['invoice_amount'].sum()
    
    print(f"📊 Risk Distribution:")
    print(f"   High risk: {high_risk_count} invoices")
    print(f"   Medium risk: {medium_risk_count} invoices")
    print(f"   Total amount at risk: ${total_amount_at_risk:,}")
    
    print(f"\n🎯 Action Items:")
    if high_risk_count > 0:
        print(f"   • Contact {high_risk_count} high-risk vendors immediately")
    if medium_risk_count > 0:
        print(f"   • Monitor {medium_risk_count} medium-risk payments closely")
    print(f"   • Review cash flow for potential ${total_amount_at_risk:,} in delays")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Vendors with past delays are higher risk")
    print(f"   • Disputes significantly increase delay probability")
    print(f"   • Large amounts require extra attention")
    print(f"   • Proactive communication can prevent delays")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    results = predict_payment_delays()
