"""
Main execution script for Vendor Payment Delay Prediction

This script runs the complete pipeline from data generation to model evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import VendorPaymentDataGenerator
from preprocessing import DataPreprocessor
from models import PaymentDelayPredictor
from evaluation import ModelEvaluator
import joblib
import warnings
warnings.filterwarnings('ignore')

def run_complete_pipeline():
    """Run the complete machine learning pipeline"""
    
    print("=" * 60)
    print("VENDOR PAYMENT DELAY PREDICTION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Load Existing Data
    print("\n1. LOADING EXISTING DATA")
    print("-" * 30)
    import pandas as pd
    data_path = '/Users/sreevanthsv/Desktop/DWDM/data/raw/vendor_payments.csv'
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records from {data_path}")
    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Target distribution: {df['is_delayed'].value_counts().to_dict()}")
    
    # Step 2: Data Preprocessing
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
        # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Create processed directory if it doesn't exist
    import os
    processed_dir = '/Users/sreevanthsv/Desktop/DWDM/data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    joblib.dump(processed_data, '/Users/sreevanthsv/Desktop/DWDM/data/processed/processed_data.pkl')
    joblib.dump(preprocessor, '/Users/sreevanthsv/Desktop/DWDM/data/processed/preprocessor.pkl')
    
    # Step 3: Model Training
    print("\n3. MODEL TRAINING")
    print("-" * 30)
    predictor = PaymentDelayPredictor()
    models = predictor.train_all_models(X_train, y_train, use_smote=True, tune_hyperparameters=False)
    
    # Step 4: Model Evaluation
    print("\n4. MODEL EVALUATION")
    print("-" * 30)
    results = predictor.evaluate_models(X_test, y_test)
    
    # Step 5: Feature Importance Analysis
    print("\n5. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    feature_importance = predictor.get_feature_importance('random_forest', top_n=20)
    print("Top 20 Most Important Features:")
    print(feature_importance)
    
    # Plot feature importance
    reports_dir = '/Users/sreevanthsv/Desktop/DWDM/reports/figures'
    os.makedirs(reports_dir, exist_ok=True)
    predictor.plot_feature_importance(
        model_name='random_forest',
        save_path='/Users/sreevanthsv/Desktop/DWDM/reports/figures/feature_importance.png'
    )
    
    # Step 6: Save Models
    print("\n6. SAVING MODELS")
    print("-" * 30)
    models_dir = '/Users/sreevanthsv/Desktop/DWDM/models/trained_models'
    os.makedirs(models_dir, exist_ok=True)
    predictor.save_models('/Users/sreevanthsv/Desktop/DWDM/models/trained_models')
    
    # Step 7: Business Impact Analysis
    print("\n7. BUSINESS IMPACT ANALYSIS")
    print("-" * 30)
    evaluator = ModelEvaluator()
    
    # Get best model results
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_y_pred_proba = results[best_model_name]['y_pred_proba']
    
    business_impact = evaluator.business_impact_analysis(y_test, best_y_pred_proba)
    
    # Step 8: Generate Summary Report
    print("\n8. SUMMARY REPORT")
    print("-" * 30)
    print_summary_report(results, feature_importance, business_impact)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return predictor, results, evaluator

def print_summary_report(results, feature_importance, business_impact):
    """Print a comprehensive summary report"""
    
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("-" * 40)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  • Accuracy: {result['accuracy']:.4f}")
        print(f"  • AUC-ROC: {result['auc']:.4f}")
    
    # Best model
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    print(f"\n🏆 BEST MODEL: {best_model.upper()}")
    print(f"   AUC-ROC Score: {results[best_model]['auc']:.4f}")
    
    print("\n🔍 TOP 10 MOST IMPORTANT FEATURES")
    print("-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    print("\n💰 BUSINESS IMPACT")
    print("-" * 40)
    print(f"  • Total Benefit: ${business_impact['total_benefit']:,}")
    print(f"  • Total Cost: ${business_impact['total_cost']:,}")
    print(f"  • Net Benefit: ${business_impact['net_benefit']:,}")
    print(f"  • Correctly Identified Delays: {business_impact['tp']}")
    print(f"  • Missed Delays: {business_impact['fn']}")
    
    print("\n🎯 KEY INSIGHTS")
    print("-" * 40)
    print("  • Payment terms and vendor history are key predictors")
    print("  • Cash flow levels significantly impact delay probability")
    print("  • Seasonal patterns affect payment delays")
    print("  • Early warning system can prevent costly delays")
    
    print("\n📈 RECOMMENDATIONS")
    print("-" * 40)
    print("  • Implement real-time monitoring dashboard")
    print("  • Set up automated alerts for high-risk payments")
    print("  • Prioritize vendor relationship management")
    print("  • Optimize cash flow management processes")

def demonstrate_prediction():
    """Demonstrate how to use the trained model for new predictions"""
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION: PREDICTING NEW PAYMENT DELAYS")
    print("=" * 60)
    
    # Load trained model and preprocessor
    predictor = joblib.load('/Users/sreevanthsv/Desktop/DWDM/models/trained_models/payment_delay_predictor.pkl')
    preprocessor = joblib.load('/Users/sreevanthsv/Desktop/DWDM/data/processed/preprocessor.pkl')
    
    # Create sample new data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    new_data = pd.DataFrame({
        'vendor_id': ['V0001', 'V0002', 'V0003'],
        'invoice_id': ['INV001', 'INV002', 'INV003'],
        'vendor_category': ['Technology', 'Manufacturing', 'Services'],
        'invoice_date': [datetime.now() - timedelta(days=5),
                        datetime.now() - timedelta(days=10),
                        datetime.now() - timedelta(days=3)],
        'due_date': [datetime.now() + timedelta(days=25),
                    datetime.now() + timedelta(days=35),
                    datetime.now() + timedelta(days=12)],
        'payment_terms': [30, 45, 15],
        'invoice_amount': [50000, 150000, 25000],
        'quarter': ['Q3', 'Q3', 'Q3'],
        'year': [2024, 2024, 2024],
        'cash_flow_level': ['Medium', 'Low', 'High'],
        'month_cash_available': [4000000, 2000000, 8000000],
        'vendor_past_delays': [1, 5, 0],
        'vendor_relationship_years': [2.5, 5.0, 1.0],
        'payment_frequency': [3, 8, 2],
        'has_dispute': [0, 1, 0],
        'economic_indicator': [95, 85, 105]
    })
    
    # Preprocess the new data
    new_data_processed = preprocessor.create_features(new_data)
    new_data_encoded = preprocessor.encode_categorical_features(new_data_processed, fit=False)
    new_X = preprocessor.select_features(new_data_encoded)
    
    # Ensure all required features are present
    missing_features = set(predictor.feature_names) - set(new_X.columns)
    for feature in missing_features:
        new_X[feature] = 0
    
    # Reorder columns to match training data
    new_X = new_X[predictor.feature_names]
    
    # Scale features
    new_X_scaled = preprocessor.scaler.transform(new_X)
    new_X_scaled = pd.DataFrame(new_X_scaled, columns=predictor.feature_names)
    
    # Make predictions
    predictions = predictor.predict_payment_delay(new_X_scaled, return_proba=True)
    binary_predictions = (predictions >= 0.5).astype(int)
    
    # Display results
    print("\n📋 SAMPLE PREDICTIONS")
    print("-" * 40)
    for i, (vendor_id, prob, binary_pred) in enumerate(zip(new_data['vendor_id'], predictions, binary_predictions)):
        status = "DELAYED" if binary_pred else "ON-TIME"
        risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        
        print(f"\nVendor {vendor_id}:")
        print(f"  • Delay Probability: {prob:.3f}")
        print(f"  • Prediction: {status}")
        print(f"  • Risk Level: {risk_level}")
        
        if binary_pred:
            print(f"  • ⚠️  ALERT: High risk of payment delay!")
        else:
            print(f"  • ✅ Expected on-time payment")

if __name__ == "__main__":
    # Run complete pipeline
    predictor, results, evaluator = run_complete_pipeline()
    
    # Demonstrate predictions
    demonstrate_prediction()
