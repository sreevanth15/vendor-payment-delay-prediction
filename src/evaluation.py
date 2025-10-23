"""
Model Evaluation Module for Vendor Payment Delay Prediction

This module provides comprehensive evaluation metrics and visualizations
for the payment delay prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator"""
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['On-time', 'Delayed'],
                    yticklabels=['On-time', 'Delayed'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        print(f"Confusion Matrix Analysis for {model_name}:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
        print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return precision, recall, avg_precision
    
    def compare_models(self, results, save_path=None):
        """Compare multiple models performance"""
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            model_metrics = {}
            model_metrics['Model'] = model_name.replace('_', ' ').title()
            
            y_true = results[model_name].get('y_true')
            y_pred = results[model_name].get('y_pred')
            y_pred_proba = results[model_name].get('y_pred_proba')
            
            if y_true is not None and y_pred is not None and y_pred_proba is not None:
                calculated_metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
                model_metrics.update(calculated_metrics)
            else:
                # Use existing metrics from results
                for metric in metrics:
                    if metric in results[model_name]:
                        model_metrics[metric] = results[model_name][metric]
            
            comparison_data.append(model_metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[i]
                bars = ax.bar(comparison_df['Model'], comparison_df[metric])
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def plot_learning_curves(self, model, X, y, model_name, save_path=None):
        """Plot learning curves to analyze model performance vs training size"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', 
                label='Training AUC')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_scores_mean, 'o-', color='red', 
                label='Validation AUC')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('AUC Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_prediction_distribution(self, y_pred_proba, model_name, save_path=None):
        """Analyze the distribution of prediction probabilities"""
        plt.figure(figsize=(12, 5))
        
        # Histogram of prediction probabilities
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Prediction Probabilities - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(y_pred_proba)
        plt.ylabel('Prediction Probability')
        plt.title(f'Box Plot of Prediction Probabilities - {model_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"Prediction Probability Statistics for {model_name}:")
        print(f"Mean: {np.mean(y_pred_proba):.4f}")
        print(f"Median: {np.median(y_pred_proba):.4f}")
        print(f"Std: {np.std(y_pred_proba):.4f}")
        print(f"Min: {np.min(y_pred_proba):.4f}")
        print(f"Max: {np.max(y_pred_proba):.4f}")
    
    def business_impact_analysis(self, y_true, y_pred_proba, threshold=0.5):
        """Analyze business impact of the predictions"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate business metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Assume business costs
        cost_false_positive = 100  # Cost of unnecessarily flagging on-time payment
        cost_false_negative = 1000  # Cost of missing a delayed payment
        benefit_true_positive = 500  # Benefit of correctly identifying delayed payment
        
        total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
        total_benefit = tp * benefit_true_positive
        net_benefit = total_benefit - total_cost
        
        print("Business Impact Analysis:")
        print(f"True Positives (Correctly identified delays): {tp}")
        print(f"False Positives (Incorrectly flagged on-time): {fp}")
        print(f"False Negatives (Missed delays): {fn}")
        print(f"True Negatives (Correctly identified on-time): {tn}")
        print(f"\nCost Analysis:")
        print(f"Cost from False Positives: ${fp * cost_false_positive:,}")
        print(f"Cost from False Negatives: ${fn * cost_false_negative:,}")
        print(f"Total Cost: ${total_cost:,}")
        print(f"Total Benefit: ${total_benefit:,}")
        print(f"Net Benefit: ${net_benefit:,}")
        
        return {
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'net_benefit': net_benefit,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def threshold_analysis(self, y_true, y_pred_proba, save_path=None):
        """Analyze performance across different threshold values"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics_by_threshold = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'net_benefit': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Business impact
            business_impact = self.business_impact_analysis(y_true, y_pred_proba, threshold)
            net_benefit = business_impact['net_benefit']
            
            metrics_by_threshold['threshold'].append(threshold)
            metrics_by_threshold['precision'].append(precision)
            metrics_by_threshold['recall'].append(recall)
            metrics_by_threshold['f1_score'].append(f1)
            metrics_by_threshold['net_benefit'].append(net_benefit)
        
        # Plot threshold analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision vs Threshold
        axes[0, 0].plot(metrics_by_threshold['threshold'], metrics_by_threshold['precision'])
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Threshold
        axes[0, 1].plot(metrics_by_threshold['threshold'], metrics_by_threshold['recall'])
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score vs Threshold
        axes[1, 0].plot(metrics_by_threshold['threshold'], metrics_by_threshold['f1_score'])
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Net Benefit vs Threshold
        axes[1, 1].plot(metrics_by_threshold['threshold'], metrics_by_threshold['net_benefit'])
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Net Benefit ($)')
        axes[1, 1].set_title('Net Benefit vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal threshold
        optimal_idx = np.argmax(metrics_by_threshold['net_benefit'])
        optimal_threshold = metrics_by_threshold['threshold'][optimal_idx]
        optimal_net_benefit = metrics_by_threshold['net_benefit'][optimal_idx]
        
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        print(f"Net Benefit at Optimal Threshold: ${optimal_net_benefit:,}")
        
        return pd.DataFrame(metrics_by_threshold), optimal_threshold

def main():
    """Main evaluation function"""
    # Load models and results
    print("Loading models and test data...")
    
    # Load processed data
    processed_data = joblib.load('/Users/sreevanthsv/Desktop/DWDM project/data/processed/processed_data.pkl')
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Load trained predictor
    predictor = joblib.load('/Users/sreevanthsv/Desktop/DWDM project/models/trained_models/payment_delay_predictor.pkl')
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    results = {}
    for model_name, model in predictor.models.items():
        print(f"\nEvaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name,
            save_path=f'/Users/sreevanthsv/Desktop/DWDM project/reports/figures/confusion_matrix_{model_name}.png'
        )
        
        # Plot ROC curve
        evaluator.plot_roc_curve(
            y_test, y_pred_proba, model_name,
            save_path=f'/Users/sreevanthsv/Desktop/DWDM project/reports/figures/roc_curve_{model_name}.png'
        )
        
        # Business impact analysis
        evaluator.business_impact_analysis(y_test, y_pred_proba)
    
    # Compare all models
    comparison_df = evaluator.compare_models(
        results, 
        save_path='/Users/sreevanthsv/Desktop/DWDM project/reports/figures/model_comparison.png'
    )
    
    print("\nModel Comparison Summary:")
    print(comparison_df)
    
    # Threshold analysis for best model
    best_model_name = comparison_df.loc[comparison_df['auc_roc'].idxmax(), 'Model'].lower().replace(' ', '_')
    if best_model_name in results:
        print(f"\nPerforming threshold analysis for best model: {best_model_name}")
        threshold_df, optimal_threshold = evaluator.threshold_analysis(
            results[best_model_name]['y_true'],
            results[best_model_name]['y_pred_proba'],
            save_path='/Users/sreevanthsv/Desktop/DWDM project/reports/figures/threshold_analysis.png'
        )
    
    print("Model evaluation completed!")
    return evaluator, results, comparison_df

if __name__ == "__main__":
    main()
