"""
Flask Web Application for Vendor Payment Delay Prediction System

This web app provides an interactive dashboard to showcase the DWDM project.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import DataPreprocessor

app = Flask(__name__)

# Global variables to store models and data
predictor = None
preprocessor = None
sample_data = None
model_results = None

def load_models_and_data():
    """Load trained models and sample data"""
    global predictor, preprocessor, sample_data, model_results
    
    try:
        # Load models
        predictor = joblib.load('models/trained_models/payment_delay_predictor.pkl')
        preprocessor = joblib.load('data/processed/preprocessor.pkl')
        
        # Load sample data
        sample_data = pd.read_csv('data/raw/vendor_payments.csv')
        
        # Load model results (create if doesn't exist)
        model_results = {
            'logistic_regression': {'accuracy': 0.6500, 'auc': 0.6906},
            'random_forest': {'accuracy': 0.7635, 'auc': 0.6835},
            'xgboost': {'accuracy': 0.7535, 'auc': 0.6586},
            'lightgbm': {'accuracy': 0.5555, 'auc': 0.6689}
        }
        
        print("✓ Models and data loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def home():
    """Home page with project overview"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard with statistics and visualizations"""
    if sample_data is None:
        return "Please run the model training first!", 500
    
    # Calculate statistics
    total_records = len(sample_data)
    delayed_count = sample_data['is_delayed'].sum()
    delay_rate = delayed_count / total_records * 100
    
    # Category breakdown
    category_stats = sample_data.groupby('vendor_category').agg({
        'is_delayed': ['sum', 'count']
    }).reset_index()
    category_stats.columns = ['category', 'delayed', 'total']
    category_stats['delay_rate'] = (category_stats['delayed'] / category_stats['total'] * 100).round(2)
    
    # Get vendor details by category
    vendor_by_category = {}
    for category in sample_data['vendor_category'].unique():
        category_vendors = sample_data[sample_data['vendor_category'] == category].groupby('vendor_id').agg({
            'is_delayed': ['sum', 'count']
        }).reset_index()
        category_vendors.columns = ['vendor_id', 'delayed', 'total']
        category_vendors['delay_rate'] = (category_vendors['delayed'] / category_vendors['total'] * 100).round(2)
        category_vendors = category_vendors.sort_values('delay_rate', ascending=False)
        vendor_by_category[category] = category_vendors.to_dict('records')
    
    # Top features
    top_features = [
        {'name': 'Vendor Risk Score', 'importance': 0.0721},
        {'name': 'Vendor Reliability Score', 'importance': 0.0612},
        {'name': 'Vendor Past Delays', 'importance': 0.0569},
        {'name': 'Invoice Month', 'importance': 0.0558},
        {'name': 'Invoice Quarter', 'importance': 0.0537}
    ]
    
    stats = {
        'total_records': total_records,
        'delayed_count': int(delayed_count),
        'on_time_count': int(total_records - delayed_count),
        'delay_rate': round(delay_rate, 2),
        'category_stats': category_stats.to_dict('records'),
        'vendor_by_category': vendor_by_category,
        'top_features': top_features
    }
    
    return render_template('dashboard.html', stats=stats, results=model_results)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page for new invoices"""
    if request.method == 'GET':
        # Show prediction form
        vendor_categories = ['Technology', 'Office Supplies', 'Manufacturing', 
                           'Services', 'Raw Materials', 'Transportation', 'Utilities']
        return render_template('predict.html', categories=vendor_categories)
    
    # Handle prediction
    try:
        # Get form data
        data = {
            'vendor_id': request.form.get('vendor_id'),
            'invoice_id': request.form.get('invoice_id'),
            'vendor_category': request.form.get('vendor_category'),
            'invoice_date': request.form.get('invoice_date'),
            'due_date': request.form.get('due_date'),
            'payment_terms': int(request.form.get('payment_terms')),
            'invoice_amount': float(request.form.get('invoice_amount')),
            'quarter': request.form.get('quarter'),
            'year': int(request.form.get('year')),
            'cash_flow_level': request.form.get('cash_flow_level'),
            'month_cash_available': float(request.form.get('month_cash_available')),
            'vendor_past_delays': int(request.form.get('vendor_past_delays')),
            'vendor_relationship_years': float(request.form.get('vendor_relationship_years')),
            'payment_frequency': int(request.form.get('payment_frequency')),
            'has_dispute': int(request.form.get('has_dispute', 0)),
            'economic_indicator': float(request.form.get('economic_indicator'))
        }
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Add dummy actual_payment_date and days_difference
        df['actual_payment_date'] = df['due_date']
        df['days_difference'] = 0
        df['is_delayed'] = 0
        
        # Preprocess
        df_processed = preprocessor.create_features(df)
        df_encoded = preprocessor.encode_categorical_features(df_processed, fit=False)
        X = preprocessor.select_features(df_encoded)
        
        # Ensure all expected features are present (add missing columns with 0)
        for col in preprocessor.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training order
        X = X[preprocessor.feature_names]
        
        # Scale features
        X_scaled = preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=preprocessor.feature_names)
        
        # Predict
        prediction = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression')[0]
        probability = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression', return_proba=True)[0]
        
        # Risk level
        if probability < 0.3:
            risk_level = 'LOW'
            risk_color = 'success'
        elif probability < 0.7:
            risk_level = 'MEDIUM'
            risk_color = 'warning'
        else:
            risk_level = 'HIGH'
            risk_color = 'danger'
        
        result = {
            'prediction': 'DELAYED' if prediction == 1 else 'ON-TIME',
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'vendor_id': data['vendor_id'],
            'invoice_amount': data['invoice_amount']
        }
        
        return render_template('predict_result.html', result=result)
        
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

@app.route('/models')
def models():
    """Model comparison page"""
    return render_template('models.html', results=model_results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess and predict
        df_processed = preprocessor.create_features(df)
        df_encoded = preprocessor.encode_categorical_features(df_processed, fit=False)
        X = preprocessor.select_features(df_encoded)
        
        # Ensure all expected features are present
        for col in preprocessor.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training order
        X = X[preprocessor.feature_names]
        
        # Scale and predict
        X_scaled = preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=preprocessor.feature_names)
        
        prediction = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression')[0]
        probability = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression', return_proba=True)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.3 else 'LOW'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with methodology"""
    return render_template('about.html')

@app.route('/vendor-risk-scoring')
def vendor_risk_scoring():
    """Vendor Risk Scoring Dashboard"""
    if sample_data is None:
        return "Please run the model training first!", 500
    
    # Calculate vendor risk scores
    vendor_stats = sample_data.groupby('vendor_id').agg({
        'is_delayed': ['sum', 'count', 'mean'],
        'invoice_amount': 'mean',
        'vendor_past_delays': 'first',
        'vendor_category': 'first'
    }).reset_index()
    
    vendor_stats.columns = ['vendor_id', 'delayed', 'total', 'delay_rate', 'avg_amount', 'past_delays', 'category']
    vendor_stats['delay_rate'] = (vendor_stats['delay_rate'] * 100).round(2)
    
    # Calculate risk score (0-100)
    vendor_stats['risk_score'] = (
        vendor_stats['delay_rate'] * 0.6 + 
        (vendor_stats['past_delays'] / vendor_stats['past_delays'].max() * 100 if vendor_stats['past_delays'].max() > 0 else 0) * 0.4
    ).round(2)
    
    # Sort by risk score
    vendor_stats = vendor_stats.sort_values('risk_score', ascending=False)
    
    # Categorize risk levels
    vendor_stats['risk_level'] = vendor_stats['risk_score'].apply(
        lambda x: 'HIGH' if x >= 50 else 'MEDIUM' if x >= 25 else 'LOW'
    )
    
    top_risky = vendor_stats.head(20).to_dict('records')
    
    stats = {
        'total_vendors': len(vendor_stats),
        'high_risk': len(vendor_stats[vendor_stats['risk_level'] == 'HIGH']),
        'medium_risk': len(vendor_stats[vendor_stats['risk_level'] == 'MEDIUM']),
        'low_risk': len(vendor_stats[vendor_stats['risk_level'] == 'LOW']),
        'top_risky_vendors': top_risky
    }
    
    return render_template('vendor_risk.html', stats=stats)

@app.route('/what-if')
def what_if_analysis():
    """What-If Analysis Tool"""
    return render_template('what_if.html')

@app.route('/api/what-if-predict', methods=['POST'])
def what_if_predict():
    """API endpoint for what-if predictions"""
    try:
        data = request.json
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        df_processed = preprocessor.create_features(df)
        df_encoded = preprocessor.encode_categorical_features(df_processed, fit=False)
        X = preprocessor.select_features(df_encoded)
        
        # Ensure all expected features are present
        for col in preprocessor.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[preprocessor.feature_names]
        
        # Scale and predict
        X_scaled = preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=preprocessor.feature_names)
        
        prediction = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression')[0]
        probability = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression', return_proba=True)[0]
        
        # Get feature importances for root cause
        feature_importance = dict(zip(preprocessor.feature_names, X_scaled.iloc[0].values))
        top_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.3 else 'LOW',
            'confidence': float(abs(probability - 0.5) * 200),  # 0-100 scale
            'top_factors': [{'feature': f[0], 'value': float(f[1])} for f in top_factors]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bulk-upload')
def bulk_upload():
    """Bulk Upload and Prediction"""
    return render_template('bulk_upload.html')

@app.route('/api/bulk-predict', methods=['POST'])
def bulk_predict():
    """API endpoint for bulk predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Preprocess
        df_processed = preprocessor.create_features(df)
        df_encoded = preprocessor.encode_categorical_features(df_processed, fit=False)
        X = preprocessor.select_features(df_encoded)
        
        # Ensure all expected features are present
        for col in preprocessor.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[preprocessor.feature_names]
        
        # Scale and predict
        X_scaled = preprocessor.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=preprocessor.feature_names)
        
        predictions = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression')
        probabilities = predictor.predict_payment_delay(X_scaled, model_name='logistic_regression', return_proba=True)
        
        # Add results to original dataframe
        results_df = df.copy()
        results_df['prediction'] = ['DELAYED' if p == 1 else 'ON-TIME' for p in predictions]
        results_df['probability'] = probabilities
        results_df['risk_level'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.3 else 'LOW' for p in probabilities]
        
        # Convert to JSON
        results = results_df.to_dict('records')
        
        # Summary stats
        summary = {
            'total': len(results),
            'high_risk': len([r for r in results if r['risk_level'] == 'HIGH']),
            'medium_risk': len([r for r in results if r['risk_level'] == 'MEDIUM']),
            'low_risk': len([r for r in results if r['risk_level'] == 'LOW']),
            'predicted_delays': int(predictions.sum())
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cost-calculator')
def cost_calculator():
    """Cost Impact Calculator"""
    if sample_data is None:
        return "Please run the model training first!", 500
    
    # Calculate actual costs from historical data
    total_records = len(sample_data)
    delayed_count = sample_data['is_delayed'].sum()
    avg_invoice = sample_data['invoice_amount'].mean()
    
    # Assumptions for cost calculation
    avg_late_fee_rate = 0.02  # 2% late fee
    avg_days_delayed = 15
    relationship_cost = 500  # Cost of damaged vendor relationship
    
    # Calculate costs
    late_fees = delayed_count * avg_invoice * avg_late_fee_rate
    relationship_costs = delayed_count * relationship_cost
    total_cost = late_fees + relationship_costs
    
    # Potential savings with 30% reduction in delays
    potential_savings = total_cost * 0.3
    
    stats = {
        'total_payments': int(total_records),
        'delayed_payments': int(delayed_count),
        'avg_invoice_amount': round(avg_invoice, 2),
        'late_fees': round(late_fees, 2),
        'relationship_costs': round(relationship_costs, 2),
        'total_annual_cost': round(total_cost, 2),
        'potential_savings': round(potential_savings, 2),
        'delay_rate': round((delayed_count / total_records * 100), 2)
    }
    
    return render_template('cost_calculator.html', stats=stats)

if __name__ == '__main__':
    print("Starting Vendor Payment Delay Prediction Web App...")
    print("Loading models and data...")
    
    if load_models_and_data():
        print("\n" + "="*60)
        print("✓ Server ready!")
        print("✓ Open your browser and go to: http://localhost:5002")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5002)
    else:
        print("Failed to load models. Please run src/main.py first!")