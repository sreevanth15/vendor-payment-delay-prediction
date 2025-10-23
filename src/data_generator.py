"""
Data Generator for Vendor Payment Delay Prediction

This module generates synthetic vendor payment data for training and testing
the payment delay prediction model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class VendorPaymentDataGenerator:
    def __init__(self, n_records=10000, random_state=42):
        """
        Initialize the data generator
        
        Args:
            n_records (int): Number of records to generate
            random_state (int): Random seed for reproducibility
        """
        self.n_records = n_records
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define vendor categories and their characteristics
        self.vendor_categories = {
            'Technology': {'avg_amount': 50000, 'delay_prob': 0.15, 'payment_terms': 30},
            'Office Supplies': {'avg_amount': 5000, 'delay_prob': 0.25, 'payment_terms': 15},
            'Manufacturing': {'avg_amount': 100000, 'delay_prob': 0.20, 'payment_terms': 45},
            'Services': {'avg_amount': 25000, 'delay_prob': 0.30, 'payment_terms': 30},
            'Raw Materials': {'avg_amount': 75000, 'delay_prob': 0.18, 'payment_terms': 60},
            'Transportation': {'avg_amount': 15000, 'delay_prob': 0.22, 'payment_terms': 30},
            'Utilities': {'avg_amount': 8000, 'delay_prob': 0.10, 'payment_terms': 15}
        }
        
        # Company financial health indicators
        self.financial_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        self.cash_flow_levels = ['Low', 'Medium', 'High']
        
    def generate_vendor_data(self):
        """Generate synthetic vendor payment data"""
        
        data = []
        start_date = datetime(2020, 1, 1)
        
        for i in range(self.n_records):
            # Basic identifiers
            vendor_id = f"V{i//50 + 1:04d}"  # Multiple invoices per vendor
            invoice_id = f"INV{i+1:06d}"
            
            # Vendor characteristics
            vendor_category = np.random.choice(list(self.vendor_categories.keys()))
            vendor_info = self.vendor_categories[vendor_category]
            
            # Invoice details
            invoice_date = start_date + timedelta(days=np.random.randint(0, 1800))
            payment_terms = vendor_info['payment_terms']
            due_date = invoice_date + timedelta(days=payment_terms)
            
            # Amount with some variation
            base_amount = vendor_info['avg_amount']
            invoice_amount = np.random.normal(base_amount, base_amount * 0.3)
            invoice_amount = max(1000, invoice_amount)  # Minimum amount
            
            # Company financial state
            quarter = f"Q{((invoice_date.month - 1) // 3) + 1}"
            year = invoice_date.year
            
            # Cash flow based on seasonality and randomness
            if quarter in ['Q1', 'Q4']:  # Typically better cash flow
                cash_flow = np.random.choice(['Medium', 'High'], p=[0.3, 0.7])
            else:
                cash_flow = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
            
            # Monthly cash availability (simulated)
            month_cash_available = np.random.normal(5000000, 1500000)  # $5M average
            month_cash_available = max(1000000, month_cash_available)
            
            # Vendor performance history
            vendor_past_delays = np.random.poisson(2)  # Average 2 past delays
            vendor_relationship_years = np.random.exponential(3) + 0.5  # Average 3.5 years
            
            # Payment frequency (invoices per month from this vendor)
            payment_frequency = np.random.poisson(3) + 1  # 1-10 invoices per month
            
            # Dispute indicators
            has_dispute = np.random.choice([0, 1], p=[0.85, 0.15])
            
            # External factors
            economic_indicator = np.random.normal(100, 15)  # Economic index
            
            # Calculate delay probability based on multiple factors
            base_delay_prob = vendor_info['delay_prob']
            
            # Adjust probability based on various factors
            if cash_flow == 'Low':
                base_delay_prob *= 1.5
            elif cash_flow == 'High':
                base_delay_prob *= 0.7
                
            if invoice_amount > base_amount * 1.5:  # Large invoice
                base_delay_prob *= 1.3
                
            if vendor_past_delays > 3:
                base_delay_prob *= 1.2
                
            if has_dispute:
                base_delay_prob *= 2.0
                
            if month_cash_available < 2000000:  # Low cash available
                base_delay_prob *= 1.4
                
            # Seasonal effects
            if quarter in ['Q2', 'Q3']:  # Mid-year cash flow issues
                base_delay_prob *= 1.1
                
            # Cap probability
            delay_prob = min(0.8, base_delay_prob)
            
            # Determine if payment is delayed
            is_delayed = np.random.random() < delay_prob
            
            if is_delayed:
                # Generate delay in days
                delay_days = np.random.exponential(7) + 1  # Average 8 days delay
                actual_payment_date = due_date + timedelta(days=int(delay_days))
            else:
                # Early or on-time payment
                early_days = np.random.exponential(2)  # Sometimes pay early
                actual_payment_date = due_date - timedelta(days=int(early_days))
                if actual_payment_date < invoice_date:
                    actual_payment_date = due_date
            
            # Calculate days difference (positive = delayed, negative = early)
            days_difference = (actual_payment_date - due_date).days
            
            data.append({
                'vendor_id': vendor_id,
                'invoice_id': invoice_id,
                'vendor_category': vendor_category,
                'invoice_date': invoice_date,
                'due_date': due_date,
                'payment_terms': payment_terms,
                'invoice_amount': round(invoice_amount, 2),
                'quarter': quarter,
                'year': year,
                'cash_flow_level': cash_flow,
                'month_cash_available': round(month_cash_available, 2),
                'vendor_past_delays': vendor_past_delays,
                'vendor_relationship_years': round(vendor_relationship_years, 2),
                'payment_frequency': payment_frequency,
                'has_dispute': has_dispute,
                'economic_indicator': round(economic_indicator, 2),
                'actual_payment_date': actual_payment_date,
                'days_difference': days_difference,
                'is_delayed': 1 if is_delayed else 0
            })
        
        return pd.DataFrame(data)
    
    def save_data(self, output_path):
        """Generate and save the data to CSV"""
        df = self.generate_vendor_data()
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} records and saved to {output_path}")
        
        # Print basic statistics
        print(f"\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Delayed payments: {df['is_delayed'].sum()} ({df['is_delayed'].mean():.2%})")
        print(f"On-time payments: {(1-df['is_delayed']).sum()} ({(1-df['is_delayed']).mean():.2%})")
        print(f"Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
        print(f"Amount range: ${df['invoice_amount'].min():,.2f} to ${df['invoice_amount'].max():,.2f}")
        
        return df

if __name__ == "__main__":
    # Generate the dataset
    generator = VendorPaymentDataGenerator(n_records=10000, random_state=42)
    df = generator.save_data('/Users/sreevanthsv/Desktop/DWDM project/data/raw/vendor_payments.csv')
    
    print("\nVendor Categories Distribution:")
    print(df['vendor_category'].value_counts())
    
    print("\nDelay Rate by Category:")
    delay_by_category = df.groupby('vendor_category')['is_delayed'].agg(['count', 'sum', 'mean'])
    delay_by_category.columns = ['Total', 'Delayed', 'Delay_Rate']
    print(delay_by_category.sort_values('Delay_Rate', ascending=False))
