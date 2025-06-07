#!/usr/bin/env python3
"""
🔬 BERT MODEL COMPARISON EXPERIMENT
So sánh hiệu suất của các BERT models và fusion strategies khác nhau

MỤC ĐÍCH:
- So sánh PhoBERT vs Multilingual BERT
- So sánh các fusion strategies (concat, attention, gated)
- Đánh giá impact của domain features
- Tìm ra configuration tốt nhất

CÁCH SỬ DỤNG:
- Chạy từ main.py (option 2)
- Hoặc import: from main_bert_experiment import run_comprehensive_bert_experiment
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bert_training import main_bert_training
from config import DEVICE

def run_comprehensive_bert_experiment(real_file_path, fake_file_path, 
                                     output_dir='bert_comprehensive_results'):
    """
    Chạy thí nghiệm so sánh toàn diện các BERT models
    
    Args:
        real_file_path: Đường dẫn file tin thật
        fake_file_path: Đường dẫn file tin giả
        output_dir: Thư mục lưu kết quả
    
    Returns:
        results: Dictionary chứa kết quả các experiments
        experiment_dir: Đường dẫn thư mục kết quả
    """
    
    print("STARTING COMPREHENSIVE BERT COMPARISON")
    print("="*80)
    
    # Tạo thư mục output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"comparison_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # CẤU HÌNH CÁC THÍ NGHIỆM
    configurations = [
        # PhoBERT với các fusion strategies
        {
            'name': 'PhoBERT_Concat',
            'bert_model': 'vinai/phobert-base',
            'fusion_type': 'concat',
            'use_domain': True,
            'description': 'PhoBERT với concat fusion'
        },
        {
            'name': 'PhoBERT_Attention',
            'bert_model': 'vinai/phobert-base',
            'fusion_type': 'attention',
            'use_domain': True,
            'description': 'PhoBERT với attention fusion + domain regularization'
        },
        {
            'name': 'PhoBERT_Gated',
            'bert_model': 'vinai/phobert-base',
            'fusion_type': 'gated',
            'use_domain': True,
            'description': 'PhoBERT với gated fusion'
        },
        
        # Multilingual BERT để so sánh
        {
            'name': 'MultiBERT_Attention',
            'bert_model': 'google-bert/bert-base-multilingual-cased',
            'fusion_type': 'attention',
            'use_domain': True,
            'description': 'Multilingual BERT với attention fusion'
        },
        
        # Test impact của domain features
        {
            'name': 'PhoBERT_Attention_NoDomain',
            'bert_model': 'vinai/phobert-base',
            'fusion_type': 'attention',
            'use_domain': False,
            'description': 'PhoBERT attention KHÔNG dùng domain features'
        }
    ]
    
    # Cấu hình training chung
    common_config = {
        'num_epochs': 3,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'freeze_bert_layers': 0,
        'balance_strategy': 'smotetomek',
        'balance_target_ratio': 0.65
    }
    
    results = {}
    
    print(f"📊 Will run {len(configurations)} experiments:")
    for i, config in enumerate(configurations, 1):
        print(f"   {i}. {config['name']}: {config['description']}")
    
    # Chạy từng thí nghiệm
    for i, config in enumerate(configurations):
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {i+1}/{len(configurations)}: {config['name']}")
        print(f"📝 {config['description']}")
        print(f"{'='*80}")
        
        try:
            # Merge config
            full_config = {**config, **common_config}
            
            # Chạy thí nghiệm
            test_results, history = main_bert_training(
                real_file_path=real_file_path,
                fake_file_path=fake_file_path,
                bert_model=full_config['bert_model'],
                fusion_type=full_config['fusion_type'],
                use_domain=full_config['use_domain'],
                num_epochs=full_config['num_epochs'],
                batch_size=full_config['batch_size'],
                learning_rate=full_config['learning_rate'],
                freeze_bert_layers=full_config['freeze_bert_layers'],
                balance_strategy=full_config['balance_strategy'],
                balance_target_ratio=full_config.get('balance_target_ratio', 0.65),
                save_dir=experiment_dir
            )
            
            # Lưu kết quả
            results[config['name']] = {
                'config': full_config,
                'test_results': test_results,
                'history': history,
                'status': 'success'
            }
            
            print(f"✅ {config['name']} COMPLETED!")
            print(f"   📊 Accuracy: {test_results['accuracy']:.4f}")
            print(f"   🎯 F1-Fake: {test_results['f1_fake']:.4f}")
            
        except Exception as e:
            print(f"❌ {config['name']} FAILED: {str(e)}")
            results[config['name']] = {
                'config': config,
                'error': str(e),
                'status': 'failed'
            }
    
    # Tạo báo cáo so sánh
    print(f"\nCreating comparison report...")
    create_comparison_report(results, experiment_dir)
    
    return results, experiment_dir

def create_comparison_report(results, save_dir):
    """Tạo báo cáo so sánh chi tiết"""
    
    print(f"\nCREATING DETAILED COMPARISON REPORT")
    print("="*60)
    
    # Thu thập dữ liệu thành công
    comparison_data = []
    
    for name, result in results.items():
        if result['status'] == 'success':
            test_results = result['test_results']
            config = result['config']
            
            comparison_data.append({
                'Model': name,
                'BERT_Type': 'PhoBERT' if 'phobert' in config['bert_model'].lower() else 'MultiBERT',
                'Fusion_Type': config['fusion_type'].capitalize(),
                'Use_Domain': config['use_domain'],
                'Accuracy': test_results['accuracy'],
                'F1_Macro': test_results['f1_macro'],
                'F1_Weighted': test_results['f1_weighted'],
                'F1_Fake': test_results['f1_fake'],
                'Description': config.get('description', '')
            })
    
    if not comparison_data:
        print("No successful experiments to compare")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Lưu bảng so sánh
    comparison_path = os.path.join(save_dir, 'model_comparison.csv')
    df_comparison.to_csv(comparison_path, index=False)
    
    # In bảng so sánh
    print(f"\nMODEL PERFORMANCE COMPARISON:")
    print("-"*100)
    print(df_comparison[['Model', 'BERT_Type', 'Fusion_Type', 'Use_Domain', 'Accuracy', 'F1_Fake']].round(4).to_string(index=False))
    
    # Tìm model tốt nhất
    best_f1_idx = df_comparison['F1_Fake'].idxmax()
    best_model = df_comparison.loc[best_f1_idx]
    
    print(f"\nBEST MODEL FOUND:")
    print(f"   Model: {best_model['Model']}")
    print(f"   F1-Fake News: {best_model['F1_Fake']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   BERT Type: {best_model['BERT_Type']}")
    print(f"   Fusion: {best_model['Fusion_Type']}")
    print(f"   Uses Domain: {best_model['Use_Domain']}")
    
    # Tạo visualization
    create_comparison_plots(df_comparison, save_dir)
    
    # Lưu báo cáo chi tiết
    save_detailed_report(results, df_comparison, save_dir)
    
    return df_comparison

def create_comparison_plots(df_comparison, save_dir):
    """Tạo biểu đồ so sánh đẹp mắt"""
    
    print(f"Creating comparison visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BERT Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # F1-Fake Score comparison (Main metric)
    bars1 = axes[0, 0].bar(range(len(df_comparison)), df_comparison['F1_Fake'], 
                          color='lightcoral', alpha=0.8, edgecolor='darkred')
    axes[0, 0].set_title('🎯 F1-Score for Fake News Detection', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_xticks(range(len(df_comparison)))
    axes[0, 0].set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Overall Accuracy comparison
    bars2 = axes[0, 1].bar(range(len(df_comparison)), df_comparison['Accuracy'], 
                          color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    axes[0, 1].set_title('📊 Overall Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(len(df_comparison)))
    axes[0, 1].set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Multiple metrics comparison
    metrics = ['Accuracy', 'F1_Macro', 'F1_Weighted', 'F1_Fake']
    x = range(len(df_comparison))
    width = 0.2
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[1, 0].bar([p + i*width for p in x], df_comparison[metric], 
                      width, label=metric, alpha=0.8, color=color)
    
    axes[1, 0].set_title('📈 Multiple Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks([p + width*1.5 for p in x])
    axes[1, 0].set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # BERT Type vs Fusion Strategy heatmap
    if len(df_comparison) > 1:
        try:
            pivot_table = df_comparison.pivot_table(
                values='F1_Fake', 
                index='BERT_Type', 
                columns='Fusion_Type', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                        ax=axes[1, 1], cbar_kws={'label': 'F1-Fake'})
            axes[1, 1].set_title('🔥 BERT Type vs Fusion Strategy', fontsize=14, fontweight='bold')
        except:
            # Fallback if pivot fails
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor heatmap', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('🔥 BERT Type vs Fusion Strategy', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'model_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plots saved to: {plot_path}")
    
    plt.show()

def save_detailed_report(results, df_comparison, save_dir):
    """Lưu báo cáo chi tiết"""
    
    report_path = os.path.join(save_dir, 'detailed_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BERT FINE-TUNING COMPREHENSIVE EXPERIMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful experiments: {len(df_comparison)}\n")
        f.write(f"Failed experiments: {len(results) - len(df_comparison)}\n")
        f.write(f"Device used: {DEVICE}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL COMPARISON TABLE:\n")
        f.write("-"*40 + "\n")
        f.write(df_comparison.round(4).to_string(index=False))
        f.write("\n\n")
        
        # Best model analysis
        if not df_comparison.empty:
            best_f1_idx = df_comparison['F1_Fake'].idxmax()
            best_model_name = df_comparison.loc[best_f1_idx, 'Model']
            
            f.write("BEST MODEL ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Best model: {best_model_name}\n")
            
            if best_model_name in results:
                best_result = results[best_model_name]
                test_results = best_result['test_results']
                
                f.write(f"Performance metrics:\n")
                f.write(f"  - Accuracy: {test_results['accuracy']:.4f}\n")
                f.write(f"  - F1-Macro: {test_results['f1_macro']:.4f}\n")
                f.write(f"  - F1-Weighted: {test_results['f1_weighted']:.4f}\n")
                f.write(f"  - F1-Fake News: {test_results['f1_fake']:.4f}\n\n")
                
                f.write("Classification Report:\n")
                report = test_results['classification_report']
                for class_name in ['0', '1']:  # Real, Fake
                    class_data = report[class_name]
                    label = 'Real News' if class_name == '0' else 'Fake News'
                    f.write(f"  {label}:\n")
                    f.write(f"    - Precision: {class_data['precision']:.4f}\n")
                    f.write(f"    - Recall: {class_data['recall']:.4f}\n")
                    f.write(f"    - F1-Score: {class_data['f1-score']:.4f}\n")
            
            # Key findings
            f.write("\nKEY FINDINGS:\n")
            f.write("-"*40 + "\n")
            
            # Compare BERT types
            phobert_results = df_comparison[df_comparison['BERT_Type'] == 'PhoBERT']
            multibert_results = df_comparison[df_comparison['BERT_Type'] == 'MultiBERT']
            
            if not phobert_results.empty:
                avg_phobert_f1 = phobert_results['F1_Fake'].mean()
                f.write(f"1. PhoBERT average F1-Fake: {avg_phobert_f1:.4f}\n")
            
            if not multibert_results.empty:
                avg_multibert_f1 = multibert_results['F1_Fake'].mean()
                f.write(f"2. MultiBERT average F1-Fake: {avg_multibert_f1:.4f}\n")
            
            # Compare fusion strategies
            if len(df_comparison) > 1:
                fusion_performance = df_comparison.groupby('Fusion_Type')['F1_Fake'].mean().sort_values(ascending=False)
                f.write(f"3. Best fusion strategy: {fusion_performance.index[0]} (F1: {fusion_performance.iloc[0]:.4f})\n")
            
            # Domain impact
            with_domain = df_comparison[df_comparison['Use_Domain'] == True]
            without_domain = df_comparison[df_comparison['Use_Domain'] == False]
            
            if not with_domain.empty and not without_domain.empty:
                avg_with_domain = with_domain['F1_Fake'].mean()
                avg_without_domain = without_domain['F1_Fake'].mean()
                f.write(f"4. Impact of domain features:\n")
                f.write(f"   - With domain: {avg_with_domain:.4f}\n")
                f.write(f"   - Without domain: {avg_without_domain:.4f}\n")
                
                if avg_with_domain > avg_without_domain:
                    f.write(f"   - Domain features improve performance by {avg_with_domain - avg_without_domain:.4f}\n")
                else:
                    f.write(f"   - Domain features decrease performance by {avg_without_domain - avg_with_domain:.4f}\n")
        
        # Failed experiments
        failed_experiments = [name for name, result in results.items() if result['status'] == 'failed']
        if failed_experiments:
            f.write(f"\nFAILED EXPERIMENTS:\n")
            f.write("-"*40 + "\n")
            for name in failed_experiments:
                f.write(f"- {name}: {results[name]['error']}\n")
    
    print(f"📄 Detailed report saved to: {report_path}")

# 🧪 TEST FUNCTION
if __name__ == "__main__":
    print("This is the BERT comparison module")
    print("Run from main.py (option 2) for full comparison experiment")
    print("Or use: from main_bert_experiment import run_comprehensive_bert_experiment") 