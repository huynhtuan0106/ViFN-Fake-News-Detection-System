#!/usr/bin/env python3
"""
💾 RESULTS SAVER MODULE
Lưu kết quả chi tiết với biểu đồ, metrics và confusion matrix
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveResultsSaver:
    """
    Lưu kết quả comprehensive với visualization và metrics
    """
    
    def __init__(self, save_dir='results'):
        """
        Args:
            save_dir: Thư mục lưu kết quả
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def save_all_results(self, test_results, history=None, balance_info=None, 
                        model_name='model', experiment_config=None):
        """
        Lưu tất cả kết quả bao gồm plots và metrics
        
        Args:
            test_results: Dict với predictions, labels, probabilities, etc.
            history: Training history (optional)
            balance_info: Data balancing info (optional)
            model_name: Tên model để đặt tên file
            experiment_config: Cấu hình thí nghiệm
        """
        
        print(f"SAVING COMPREHENSIVE RESULTS")
        print("="*50)
        
        # 1. Tính toán detailed metrics
        detailed_metrics = self._calculate_detailed_metrics(test_results)
        
        # 2. Lưu metrics JSON
        self._save_metrics_json(detailed_metrics, balance_info, experiment_config, model_name)
        
        # 3. Tạo và lưu plots
        self._create_and_save_plots(test_results, detailed_metrics, history, model_name)
        
        # 4. Lưu confusion matrix
        self._save_confusion_matrix(test_results, model_name)
        
        # 5. Lưu classification report
        self._save_classification_report(test_results, model_name)
        
        # 6. Tạo summary report
        self._create_summary_report(detailed_metrics, balance_info, model_name)
        
        print(f"All results saved to: {self.save_dir}")
        return detailed_metrics
    
    def _calculate_detailed_metrics(self, test_results):
        """Tính toán detailed metrics"""
        
        y_true = test_results['labels']
        y_pred = test_results['predictions']
        y_prob = test_results.get('probabilities', None)
        
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        
        # Ensure arrays have proper shape
        if y_true.ndim == 0:
            y_true = np.array([y_true])
        if y_pred.ndim == 0:
            y_pred = np.array([y_pred])
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC nếu có probabilities
        roc_auc = None
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])  # Probability of positive class
                else:
                    roc_auc = roc_auc_score(y_true, y_prob)
            except:
                roc_auc = None
        
        detailed_metrics = {
            'accuracy': float(accuracy),
            'precision': {
                'macro': float(precision_macro),
                'weighted': float(precision_weighted),
                'real_news': float(precision_per_class[0]),
                'fake_news': float(precision_per_class[1])
            },
            'recall': {
                'macro': float(recall_macro),
                'weighted': float(recall_weighted),
                'real_news': float(recall_per_class[0]),
                'fake_news': float(recall_per_class[1])
            },
            'f1_score': {
                'macro': float(f1_macro),
                'weighted': float(f1_weighted),
                'real_news': float(f1_per_class[0]),
                'fake_news': float(f1_per_class[1])
            },
            'confusion_matrix': cm.tolist(),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'total_samples': len(y_true),
            'class_distribution': {
                'real_news': int(np.sum(y_true == 0)),
                'fake_news': int(np.sum(y_true == 1))
            }
        }
        
        return detailed_metrics
    
    def _save_metrics_json(self, detailed_metrics, balance_info, experiment_config, model_name):
        """Lưu metrics dưới dạng JSON"""
        
        results_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': detailed_metrics,
            'balance_info': balance_info,
            'experiment_config': experiment_config
        }
        
        metrics_path = os.path.join(self.save_dir, f'{model_name}_detailed_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics saved to: {metrics_path}")
    
    def _create_and_save_plots(self, test_results, detailed_metrics, history, model_name):
        """Tạo và lưu tất cả biểu đồ"""
        
        # Figure với multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix (2x2)
        ax1 = plt.subplot(3, 3, 1)
        self._plot_confusion_matrix(test_results, ax1)
        
        # 2. Metrics Bar Chart
        ax2 = plt.subplot(3, 3, 2)
        self._plot_metrics_bar_chart(detailed_metrics, ax2)
        
        # 3. ROC Curve (nếu có)
        ax3 = plt.subplot(3, 3, 3)
        self._plot_roc_curve(test_results, ax3)
        
        # 4. Per-class Performance
        ax4 = plt.subplot(3, 3, 4)
        self._plot_per_class_performance(detailed_metrics, ax4)
        
        # 5. Training History (nếu có)
        if history:
            ax5 = plt.subplot(3, 3, 5)
            self._plot_training_loss(history, ax5)
            
            ax6 = plt.subplot(3, 3, 6)
            self._plot_validation_metrics(history, ax6)
        
        # 6. Class Distribution
        ax7 = plt.subplot(3, 3, 7)
        self._plot_class_distribution(detailed_metrics, ax7)
        
        # 7. Prediction Confidence Distribution
        ax8 = plt.subplot(3, 3, 8)
        self._plot_prediction_confidence(test_results, ax8)
        
        # 8. Error Analysis
        ax9 = plt.subplot(3, 3, 9)
        self._plot_error_analysis(test_results, ax9)
        
        plt.tight_layout()
        
        # Lưu comprehensive plot
        plot_path = os.path.join(self.save_dir, f'{model_name}_comprehensive_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plots saved to: {plot_path}")
        
        # Lưu individual plots
        self._save_individual_plots(test_results, detailed_metrics, history, model_name)
        
        plt.close()
    
    def _plot_confusion_matrix(self, test_results, ax):
        """Plot confusion matrix"""
        cm = confusion_matrix(test_results['labels'], test_results['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Real News', 'Fake News'],
                   yticklabels=['Real News', 'Fake News'])
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    def _plot_metrics_bar_chart(self, detailed_metrics, ax):
        """Plot metrics bar chart"""
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        values = [
            detailed_metrics['accuracy'],
            detailed_metrics['precision']['macro'],
            detailed_metrics['recall']['macro'],
            detailed_metrics['f1_score']['macro']
        ]
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
        ax.set_title('Overall Performance Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_roc_curve(self, test_results, ax):
        """Plot ROC curve"""
        if 'probabilities' in test_results and test_results['probabilities'] is not None:
            try:
                y_true = test_results['labels']
                y_prob = test_results['probabilities']
                
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    y_prob = y_prob[:, 1]  # Probability of positive class
                
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = roc_auc_score(y_true, y_prob)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {auc_score:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve', fontweight='bold')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'ROC Curve\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ROC Curve', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'ROC Curve\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROC Curve', fontweight='bold')
    
    def _plot_per_class_performance(self, detailed_metrics, ax):
        """Plot per-class performance"""
        classes = ['Real News', 'Fake News']
        metrics = ['Precision', 'Recall', 'F1-Score']
        
        real_values = [
            detailed_metrics['precision']['real_news'],
            detailed_metrics['recall']['real_news'],
            detailed_metrics['f1_score']['real_news']
        ]
        
        fake_values = [
            detailed_metrics['precision']['fake_news'],
            detailed_metrics['recall']['fake_news'],
            detailed_metrics['f1_score']['fake_news']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_values, width, label='Real News', alpha=0.8)
        bars2 = ax.bar(x + width/2, fake_values, width, label='Fake News', alpha=0.8)
        
        ax.set_title('Per-Class Performance', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_training_loss(self, history, ax):
        """Plot training loss"""
        if 'train_losses' in history:
            epochs = range(1, len(history['train_losses']) + 1)
            ax.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Training Loss')
            ax.set_title('Training Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Training Loss\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Loss', fontweight='bold')
    
    def _plot_validation_metrics(self, history, ax):
        """Plot validation metrics"""
        if 'val_accuracies' in history and 'val_f1_scores' in history:
            epochs = range(1, len(history['val_accuracies']) + 1)
            ax.plot(epochs, history['val_accuracies'], 'g-', linewidth=2, label='Accuracy')
            ax.plot(epochs, history['val_f1_scores'], 'r-', linewidth=2, label='F1-Score')
            ax.set_title('Validation Metrics', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Validation Metrics\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Metrics', fontweight='bold')
    
    def _plot_class_distribution(self, detailed_metrics, ax):
        """Plot class distribution"""
        classes = ['Real News', 'Fake News']
        counts = [
            detailed_metrics['class_distribution']['real_news'],
            detailed_metrics['class_distribution']['fake_news']
        ]
        
        colors = ['lightblue', 'lightcoral']
        bars = ax.bar(classes, counts, color=colors, alpha=0.8)
        ax.set_title('Test Set Class Distribution', fontweight='bold')
        ax.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_prediction_confidence(self, test_results, ax):
        """Plot prediction confidence distribution"""
        if 'probabilities' in test_results and test_results['probabilities'] is not None:
            try:
                y_prob = test_results['probabilities']
                if len(y_prob.shape) > 1:
                    # Get max probability for each prediction
                    confidence = np.max(y_prob, axis=1)
                else:
                    confidence = y_prob
                
                ax.hist(confidence, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title('Prediction Confidence Distribution', fontweight='bold')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'Confidence\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Prediction Confidence', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Confidence\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Confidence', fontweight='bold')
    
    def _plot_error_analysis(self, test_results, ax):
        """Plot error analysis"""
        y_true = test_results['labels']
        y_pred = test_results['predictions']
        
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        
        # Ensure arrays have proper shape
        if y_true.ndim == 0:
            y_true = np.array([y_true])
        if y_pred.ndim == 0:
            y_pred = np.array([y_pred])
        
        # Calculate error types
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        categories = ['True\nPositives', 'True\nNegatives', 'False\nPositives', 'False\nNegatives']
        counts = [true_positives, true_negatives, false_positives, false_negatives]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title('Prediction Error Analysis', fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
    
    def _save_individual_plots(self, test_results, detailed_metrics, history, model_name):
        """Lưu các plots riêng lẻ"""
        
        # Confusion Matrix riêng
        plt.figure(figsize=(8, 6))
        self._plot_confusion_matrix(test_results, plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name}_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Metrics comparison riêng
        plt.figure(figsize=(10, 6))
        self._plot_metrics_bar_chart(detailed_metrics, plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name}_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_confusion_matrix(self, test_results, model_name):
        """Lưu confusion matrix dưới dạng CSV"""
        cm = confusion_matrix(test_results['labels'], test_results['predictions'])
        
        df_cm = pd.DataFrame(cm, 
                           columns=['Predicted_Real', 'Predicted_Fake'],
                           index=['Actual_Real', 'Actual_Fake'])
        
        cm_path = os.path.join(self.save_dir, f'{model_name}_confusion_matrix.csv')
        df_cm.to_csv(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
    
    def _save_classification_report(self, test_results, model_name):
        """Lưu classification report"""
        report = classification_report(
            test_results['labels'], 
            test_results['predictions'],
            target_names=['Real News', 'Fake News'],
            output_dict=True
        )
        
        # Convert to DataFrame for easier reading
        df_report = pd.DataFrame(report).transpose()
        
        report_path = os.path.join(self.save_dir, f'{model_name}_classification_report.csv')
        df_report.to_csv(report_path)
        print(f"Classification report saved to: {report_path}")
    
    def _create_summary_report(self, detailed_metrics, balance_info, model_name):
        """Tạo summary report"""
        
        report_path = os.path.join(self.save_dir, f'{model_name}_summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VIETNAMESE FAKE NEWS DETECTION - RESULTS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Model info
            f.write(f"Model: {model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*30 + "\n")
            f.write(f"Accuracy: {detailed_metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {detailed_metrics['precision']['macro']:.4f}\n")
            f.write(f"Recall (Macro): {detailed_metrics['recall']['macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {detailed_metrics['f1_score']['macro']:.4f}\n\n")
            
            # Per-class performance
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-"*30 + "\n")
            f.write("Real News:\n")
            f.write(f"  Precision: {detailed_metrics['precision']['real_news']:.4f}\n")
            f.write(f"  Recall: {detailed_metrics['recall']['real_news']:.4f}\n")
            f.write(f"  F1-Score: {detailed_metrics['f1_score']['real_news']:.4f}\n\n")
            f.write("Fake News:\n")
            f.write(f"  Precision: {detailed_metrics['precision']['fake_news']:.4f}\n")
            f.write(f"  Recall: {detailed_metrics['recall']['fake_news']:.4f}\n")
            f.write(f"  F1-Score: {detailed_metrics['f1_score']['fake_news']:.4f}\n\n")
            
            # Test set info
            f.write("TEST SET INFORMATION:\n")
            f.write("-"*30 + "\n")
            f.write(f"Total samples: {detailed_metrics['total_samples']}\n")
            f.write(f"Real news samples: {detailed_metrics['class_distribution']['real_news']}\n")
            f.write(f"Fake news samples: {detailed_metrics['class_distribution']['fake_news']}\n")
            
            if detailed_metrics['roc_auc']:
                f.write(f"ROC AUC: {detailed_metrics['roc_auc']:.4f}\n")
            
            # Balance info
            if balance_info:
                f.write(f"\nDATA BALANCING INFO:\n")
                f.write("-"*30 + "\n")
                f.write(f"Balancing method: {balance_info.get('balance_method', 'N/A')}\n")
                f.write(f"Original size: {balance_info.get('original_size', 'N/A')}\n")
                f.write(f"Balanced size: {balance_info.get('balanced_size', 'N/A')}\n")
                f.write(f"Synthetic samples: {balance_info.get('synthetic_samples', 'N/A')}\n")
        
        print(f"Summary report saved to: {report_path}")

def save_comprehensive_results(test_results, save_dir='results', model_name='model',
                             history=None, balance_info=None, experiment_config=None):
    """
    Hàm tiện ích để lưu comprehensive results
    
    Args:
        test_results: Dict với predictions, labels, probabilities
        save_dir: Thư mục lưu kết quả
        model_name: Tên model
        history: Training history (optional)
        balance_info: Data balancing info (optional)
        experiment_config: Experiment configuration (optional)
    
    Returns:
        detailed_metrics: Dict với detailed metrics
    """
    
    saver = ComprehensiveResultsSaver(save_dir)
    return saver.save_all_results(
        test_results, history, balance_info, model_name, experiment_config
    )

if __name__ == "__main__":
    print("Testing Results Saver...")
    
    # Tạo sample test results
    np.random.seed(42)
    n_samples = 1000
    
    # Fake test results
    test_results = {
        'labels': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'predictions': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'probabilities': np.random.rand(n_samples, 2)
    }
    
    # Fake history
    history = {
        'train_losses': [0.8, 0.6, 0.4, 0.3, 0.25],
        'val_accuracies': [0.75, 0.82, 0.85, 0.87, 0.88],
        'val_f1_scores': [0.72, 0.80, 0.83, 0.85, 0.86]
    }
    
    # Test saving
    detailed_metrics = save_comprehensive_results(
        test_results, 
        save_dir='test_results',
        model_name='test_model',
        history=history
    )
    
    print(f"Test completed!")
    print(f"Detailed metrics: {detailed_metrics}") 