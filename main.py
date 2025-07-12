"""
VIETNAMESE FAKE NEWS DETECTION - MAIN SCRIPT
Sử dụng BERT/PhoBERT Fine-tuning với Multimodal Fusion

CÁCH SỬ DỤNG:
1. Cập nhật đường dẫn file dữ liệu ở cuối file này
2. Chọn cấu hình mong muốn (model, fusion type, etc.)
3. Chạy: python main.py
"""

import os
import sys
from datetime import datetime

# Tối ưu GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    print("="*80)
    print("VIETNAMESE FAKE NEWS DETECTION SYSTEM")
    print("BERT/PhoBERT Fine-tuning with Multimodal Fusion")
    print("="*80)

def check_requirements():
    """Kiểm tra requirements"""
    try:
        import torch
        import transformers
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np
        
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("\n🎮 GPU Information:")
            print(f"   - CUDA Available: Yes")
            print(f"   - GPU Device: {torch.cuda.get_device_name()}")
            print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   - CUDA Version: {torch.version.cuda}")
        else:
            print("\nNo GPU detected, will use CPU")
        
        print("\nAll required packages are installed")
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install requirements:")
        print("pip install -r requirements_updated.txt")
        return False

def check_data_files(real_path, fake_path):
    """Kiểm tra file dữ liệu"""
    print(f"\nCHECKING DATA FILES:")
    
    if not os.path.exists(real_path):
        print(f"Real data file not found: {real_path}")
        return False
    
    if not os.path.exists(fake_path):
        print(f"Fake data file not found: {fake_path}")
        return False
    
    print(f"Real data: {real_path}")
    print(f"Fake data: {fake_path}")
    return True

def run_single_experiment(real_file_path, fake_file_path, config):    
    print(f"\nSTARTING SINGLE EXPERIMENT")
    print(f"Configuration: {config['name']}")
    print("-"*50)
    
    try:
        from bert_training import main_bert_training
        
        # Chạy thí nghiệm
        test_results, history = main_bert_training(
            real_file_path=real_file_path,
            fake_file_path=fake_file_path,
            bert_model=config['bert_model'],
            fusion_type=config['fusion_type'],
            use_domain=config['use_domain'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            freeze_bert_layers=config['freeze_bert_layers'],
            balance_strategy=config['balance_strategy'],
            balance_target_ratio=config.get('balance_target_ratio', 0.65),
            save_dir=config['save_dir']
        )
        
        # In kết quả
        print(f"\nEXPERIMENT COMPLETED!")
        print(f"RESULTS:")
        print(f"   - Accuracy: {test_results['accuracy']:.4f}")
        print(f"   - F1-Score (Fake News): {test_results['f1_fake']:.4f}")
        print(f"   - F1-Score (Macro): {test_results['f1_macro']:.4f}")
        print(f"   - F1-Score (Weighted): {test_results['f1_weighted']:.4f}")
        
        return test_results, history
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def run_comparison_experiment(real_file_path, fake_file_path, output_dir='results'):
    
    print(f"\nSTARTING COMPARISON EXPERIMENT")
    print("-"*50)
    
    try:
        from main_bert_experiment import run_comprehensive_bert_experiment
        
        results, experiment_dir = run_comprehensive_bert_experiment(
            real_file_path=real_file_path,
            fake_file_path=fake_file_path,
            output_dir=output_dir
        )
        
        print(f"\nCOMPARISON COMPLETED!")
        print(f"Results saved to: {experiment_dir}")
        
        return results, experiment_dir
        
    except Exception as e:
        print(f"Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():   
    # In banner
    print_banner()
    
    # Kiểm tra requirements
    if not check_requirements():
        sys.exit(1)
    
    # ================================
    # CẤU HÌNH DỮ LIỆU
    # ================================
    
    # CẬP NHẬT ĐƯỜNG DẪN FILE DỮ LIỆU TẠI ĐÂY:
    REAL_FILE_PATH = "D:/Huynh Tuan/STUDY/KHOA_LUAN_TOT_NGHIEP/Vietnamese-Fake-News-Dataset-Version3/processed/deduplicated_articles_real.csv"
    FAKE_FILE_PATH = "D:/Huynh Tuan/STUDY/KHOA_LUAN_TOT_NGHIEP/Vietnamese-Fake-News-Dataset-Version3/processed/deduplicated_articles_fake.csv"
    
    # Kiểm tra file dữ liệu
    if not check_data_files(REAL_FILE_PATH, FAKE_FILE_PATH):
        print("\nPlease update the file paths in main.py")
        sys.exit(1)
    
    # ================================
    # CHỌN LOẠI 
    # ================================
    
    print(f"\nSELECT EXPERIMENT TYPE:")
    print("1. Single Model Training (Recommended for testing)")
    print("2. Model Comparison (Train multiple models)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    # ================================
    # CẤU HÌNH THÍ NGHIỆM
    # ================================
    
    if choice == '1':
        # Single experiment configuration
        print(f"\nSINGLE EXPERIMENT CONFIGURATION:")
        
        # Import config từ config.py
        from config import TRAINING_CONFIG, BERT_CONFIG, FUSION_CONFIG, BALANCE_CONFIG, BATCH_SIZE_BY_GPU
        
        # Cấu hình sử dụng settings từ config.py
        config = {
            'name': 'PhoBERT_Attention_SMOTETomek_Optimized',
            'bert_model': BERT_CONFIG['model_name'],     # Từ config.py
            'fusion_type': FUSION_CONFIG['fusion_type'], # Từ config.py
            'use_domain': FUSION_CONFIG['use_domain'],   # Từ config.py
            'num_epochs': TRAINING_CONFIG['num_epochs'], # Từ config.py
            'batch_size': TRAINING_CONFIG['batch_size'], # Từ config.py - CHÍNH
            'learning_rate': TRAINING_CONFIG['learning_rate'], # Từ config.py
            'freeze_bert_layers': BERT_CONFIG['freeze_layers'], # Từ config.py
            'balance_strategy': BALANCE_CONFIG['strategy'], # Từ config.py
            'balance_target_ratio': BALANCE_CONFIG['target_ratio'], # Từ config.py
            'save_dir': f'results_single_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        
        print(f"Configuration loaded from config.py:")
        print(f"   - GPU Memory Recommendations:")
        for gpu_mem, batch_size in BATCH_SIZE_BY_GPU.items():
            print(f"     {gpu_mem}: batch_size={batch_size}")
        print(f"\n   - Current Settings:")
        for key, value in config.items():
            if key != 'save_dir':
                if key == 'batch_size':
                    print(f"   - {key}: {value} (From config.py - modify TRAINING_CONFIG to change)")
                elif key == 'balance_strategy':
                    print(f"   - {key}: {value} (Data balancing with synthetic samples)")
                elif key == 'balance_target_ratio':
                    print(f"   - {key}: {value} (Target: {value*100:.0f}% Real, {(1-value)*100:.0f}% Fake)")
                else:
                    print(f"   - {key}: {value}")
        
        # Chạy thí nghiệm
        test_results, history = run_single_experiment(REAL_FILE_PATH, FAKE_FILE_PATH, config)
        
        if test_results:
            print(f"\nResults saved to: {config['save_dir']}")
        
    else:
        # Comparison experiment
        print(f"\nCOMPARISON EXPERIMENT:")
        print("Will compare multiple BERT models and fusion strategies")
        print("This may take 1-2 hours depending on your hardware...")
        
        # Import config để hiển thị thông tin
        from config import TRAINING_CONFIG, BERT_CONFIG, FUSION_CONFIG, BALANCE_CONFIG, BATCH_SIZE_BY_GPU
        
        print(f"\nConfiguration loaded from config.py:")
        print(f"   - Training epochs: {TRAINING_CONFIG['num_epochs']}")
        print(f"   - Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"   - Learning rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"   - Balance strategy: {BALANCE_CONFIG['strategy']}")
        print(f"   - Balance ratio: {BALANCE_CONFIG['target_ratio']}")
        print(f"   - Use domain features: {FUSION_CONFIG['use_domain']}")
        print(f"   - GPU Memory Recommendations:")
        for gpu_mem, batch_size in BATCH_SIZE_BY_GPU.items():
            print(f"     {gpu_mem}: batch_size={batch_size}")
        
        # Xác nhận
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Experiment cancelled")
            sys.exit(0)
        
        # Chạy comparison
        results, experiment_dir = run_comparison_experiment(
            REAL_FILE_PATH, FAKE_FILE_PATH, 
            output_dir=f'results_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        if results:
            print(f"\nComparison results saved to: {experiment_dir}")
    
    print(f"\nProgram completed successfully!")
    print(f"Check the results folder for detailed outputs")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc() 