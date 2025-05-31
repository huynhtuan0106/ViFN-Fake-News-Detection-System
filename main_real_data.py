#!/usr/bin/env python3
"""
Main script để chạy Late Fusion với dữ liệu thực tế
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

# Import custom modules
from config import MODEL_CONFIG, DEVICE
from data_loader import load_and_preprocess_real_data  # File mới cho dữ liệu thực
from embeddings import create_all_embeddings
from models import create_model
from dataset import create_data_splits, create_data_loaders
from training import train_model, evaluate_model, calculate_metrics, print_evaluation_results

def main_with_real_data(real_file_path, fake_file_path, use_bert=False, output_dir='results'):
    """
    Main function để chạy với dữ liệu thực tế
    
    Args:
        real_file_path: Đường dẫn file CSV chứa tin thật
        fake_file_path: Đường dẫn file CSV chứa tin giả  
        use_bert: Có sử dụng BERT embeddings không (tốn tài nguyên)
        output_dir: Thư mục lưu kết quả
    """
    
    print("="*80)
    print("VIETNAMESE FAKE NEWS DETECTION - LATE FUSION ARCHITECTURE")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Real data file: {real_file_path}")
    print(f"Fake data file: {fake_file_path}")
    print(f"Use BERT embeddings: {use_bert}")
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load và preprocess dữ liệu thực tế
        print("\n" + "="*60)
        print("1. LOADING AND PREPROCESSING REAL DATA")
        print("="*60)
        
        df, domain_features, domain_dim = load_and_preprocess_real_data(
            real_file_path, fake_file_path
        )
        
        # Kiểm tra dữ liệu
        print(f"\nFinal dataset info:")
        print(f"- Total articles: {len(df)}")
        print(f"- Real articles: {len(df[df['label'] == 0])}")
        print(f"- Fake articles: {len(df[df['label'] == 1])}")
        print(f"- Unique domains: {df['domain'].nunique()}")
        
        # 2. Tạo embeddings
        print("\n" + "="*60)
        print("2. CREATING EMBEDDINGS")
        print("="*60)
        
        embeddings, tfidf_embedder = create_all_embeddings(df, use_real_bert=use_bert)
        
        # Sử dụng TF-IDF embeddings (có thể thay bằng BERT nếu muốn)
        title_features = embeddings['tfidf']['title']
        summary_features = embeddings['tfidf']['summary']
        content_features = embeddings['tfidf']['content']
        labels = df['label'].values
        
        print(f"\nEmbedding shapes:")
        print(f"- Title: {title_features.shape}")
        print(f"- Summary: {summary_features.shape}")
        print(f"- Content: {content_features.shape}")
        print(f"- Domain: {domain_features.shape}")
        
        # 3. Chia dữ liệu
        print("\n" + "="*60)
        print("3. SPLITTING DATA")
        print("="*60)
        
        splits = create_data_splits(
            title_features, summary_features, content_features, domain_features, labels
        )
        
        train_loader, val_loader, test_loader = create_data_loaders(*splits)
        
        # 4. Định nghĩa input dimensions
        input_dims = {
            'title': title_features.shape[1],
            'summary': summary_features.shape[1],
            'content': content_features.shape[1],
            'domain': domain_features.shape[1]
        }
        
        print(f"\nModel input dimensions: {input_dims}")
        
        # 5. Training và evaluation các mô hình
        fusion_types = ['concat', 'attention', 'gated']
        results = {}
        
        for i, fusion_type in enumerate(fusion_types):
            print(f"\n" + "="*80)
            print(f"4.{i+1}. TRAINING {fusion_type.upper()} FUSION MODEL")
            print(f"({i+1}/{len(fusion_types)})")
            print("="*80)
            
            # Tạo model
            model = create_model(input_dims, fusion_type=fusion_type)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
            # Training
            print(f"\nStarting training...")
            train_losses, val_accuracies = train_model(
                model, train_loader, val_loader, 
                num_epochs=MODEL_CONFIG['num_epochs']
            )
            
            # Evaluation
            print(f"\nEvaluating model...")
            predictions, test_labels, probabilities = evaluate_model(model, test_loader)
            metrics = calculate_metrics(predictions, test_labels, probabilities)
            
            # Lưu kết quả
            results[fusion_type] = {
                'model': model,
                'metrics': metrics,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            # In kết quả
            print_evaluation_results(f"{fusion_type.capitalize()} Fusion", metrics)
            
            # Lưu model
            model_path = os.path.join(output_dir, f'{fusion_type}_fusion_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dims': input_dims,
                'fusion_type': fusion_type,
                'metrics': metrics,
                'model_config': MODEL_CONFIG
            }, model_path)
            print(f"Model saved to: {model_path}")
        
        # 6. So sánh kết quả
        print("\n" + "="*80)
        print("5. FINAL COMPARISON AND ANALYSIS")
        print("="*80)
        
        comparison_df = create_comparison_table(results, fusion_types)
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        # Tìm mô hình tốt nhất
        best_model_idx = comparison_df['Accuracy'].idxmax()
        best_model_name = comparison_df.loc[best_model_idx, 'Model']
        best_fusion_type = fusion_types[best_model_idx]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"📊 Best Accuracy: {comparison_df.loc[best_model_idx, 'Accuracy']:.4f}")
        print(f"📈 Best AUC-ROC: {comparison_df.loc[best_model_idx, 'AUC-ROC']:.4f}")
        print(f"🎯 Best F1-Score: {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}")
        
        # 7. Visualization
        print(f"\n6. CREATING VISUALIZATIONS")
        plot_path = visualize_results(results, comparison_df, best_fusion_type, 
                                    test_labels, output_dir)
        print(f"Plots saved to: {plot_path}")
        
        # 8. Lưu best model riêng
        save_best_model(results[best_fusion_type]['model'], best_fusion_type,
                       comparison_df.loc[best_model_idx], output_dir)
        
        # 9. Lưu báo cáo
        save_report(df, comparison_df, best_fusion_type, output_dir)
        
        return results, comparison_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def create_comparison_table(results, fusion_types):
    """Tạo bảng so sánh kết quả"""
    comparison_data = []
    
    for fusion_type in fusion_types:
        metrics = results[fusion_type]['metrics']
        comparison_data.append({
            'Model': f"{fusion_type.capitalize()} Fusion",
            'Accuracy': metrics['accuracy'],
            'AUC-ROC': metrics['auc_roc'],
            'F1-Score': metrics['classification_report']['macro avg']['f1-score'],
            'Precision': metrics['classification_report']['macro avg']['precision'],
            'Recall': metrics['classification_report']['macro avg']['recall']
        })
    
    return pd.DataFrame(comparison_data)

def visualize_results(results, comparison_df, best_fusion_type, test_labels, output_dir):
    """Tạo visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training loss comparison
    for fusion_type in results.keys():
        axes[0, 0].plot(results[fusion_type]['train_losses'], 
                       label=f'{fusion_type.capitalize()} Fusion', 
                       marker='o', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation accuracy comparison
    for fusion_type in results.keys():
        if results[fusion_type]['val_accuracies']:
            axes[0, 1].plot(results[fusion_type]['val_accuracies'], 
                           label=f'{fusion_type.capitalize()} Fusion', 
                           marker='s', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Performance metrics comparison
    metrics = ['Accuracy', 'AUC-ROC', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        axes[1, 0].bar(x + i*width, comparison_df[metric], width, 
                      label=metric, alpha=0.8)
    
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=15)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confusion matrix cho best model
    best_predictions = results[best_fusion_type]['predictions']
    cm = confusion_matrix(test_labels, best_predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Real News', 'Fake News'], 
                yticklabels=['Real News', 'Fake News'])
    axes[1, 1].set_title(f'Confusion Matrix - {best_fusion_type.capitalize()} Fusion', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fusion_results_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_path

def save_best_model(model, fusion_type, best_metrics, output_dir):
    """Lưu mô hình tốt nhất"""
    
    model_path = os.path.join(output_dir, f'BEST_MODEL_{fusion_type}_fusion.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'fusion_type': fusion_type,
        'best_metrics': best_metrics.to_dict(),
        'model_config': MODEL_CONFIG,
        'note': 'This is the best performing model'
    }, model_path)
    
    print(f"\n💾 Best model saved to: {model_path}")

def save_report(df, comparison_df, best_fusion_type, output_dir):
    """Lưu báo cáo chi tiết"""
    
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("VIETNAMESE FAKE NEWS DETECTION - EXPERIMENT REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total articles: {len(df)}\n")
        f.write(f"Real articles: {len(df[df['label'] == 0])}\n")
        f.write(f"Fake articles: {len(df[df['label'] == 1])}\n")
        f.write(f"Unique domains: {df['domain'].nunique()}\n")
        f.write(f"Average title length: {df['title'].str.len().mean():.2f}\n")
        f.write(f"Average content length: {df['content'].str.len().mean():.2f}\n\n")
        
        # Top domains
        f.write("TOP DOMAINS:\n")
        f.write("-"*30 + "\n")
        for domain, count in df['domain'].value_counts().head(10).items():
            f.write(f"{domain}: {count}\n")
        f.write("\n")
        
        # Model comparison
        f.write("MODEL COMPARISON:\n")
        f.write("-"*30 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best model
        best_idx = comparison_df['Accuracy'].idxmax()
        f.write("BEST MODEL:\n")
        f.write("-"*30 + "\n")
        f.write(f"Model: {comparison_df.loc[best_idx, 'Model']}\n")
        f.write(f"Accuracy: {comparison_df.loc[best_idx, 'Accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {comparison_df.loc[best_idx, 'AUC-ROC']:.4f}\n")
        f.write(f"F1-Score: {comparison_df.loc[best_idx, 'F1-Score']:.4f}\n")
    
    print(f"📝 Experiment report saved to: {report_path}")

if __name__ == "__main__":
    # Cách sử dụng:
    # python main_real_data.py
    
    # Thay đổi đường dẫn file theo dữ liệu của bạn
    REAL_FILE_PATH = "FakeNew/Crawler/processing/deduplicated_articles_real.csv"  # Tin thật
    FAKE_FILE_PATH = "FakeNew/Crawler/processing/deduplicated_articles_fake.csv"  # Tin giả
    
    # Kiểm tra file tồn tại
    if not os.path.exists(REAL_FILE_PATH):
        print(f"❌ Real data file not found: {REAL_FILE_PATH}")
        print("Please update REAL_FILE_PATH with correct path")
        sys.exit(1)
        
    if not os.path.exists(FAKE_FILE_PATH):
        print(f"❌ Fake data file not found: {FAKE_FILE_PATH}")
        print("Please update FAKE_FILE_PATH with correct path")
        sys.exit(1)
    
    # Chạy thí nghiệm
    try:
        results, comparison_df = main_with_real_data(
            real_file_path=REAL_FILE_PATH,
            fake_file_path=FAKE_FILE_PATH,
            use_bert=False,  # Đặt True nếu muốn dùng BERT (tốn tài nguyên)
            output_dir='experiment_results'
        )
        
        if results is not None:
            print("\n" + "="*80)
            print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("📁 Check 'experiment_results' folder for:")
            print("   - Trained models (.pth files)")
            print("   - Visualization plots (.png)")
            print("   - Experiment report (.txt)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc() 