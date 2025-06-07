import torch

# Device configuration - Tự động detect GPU và tối ưu memory
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GPU optimization settings
GPU_CONFIG = {
    'mixed_precision': True,  # Enable AMP for memory saving
    'gradient_checkpointing': True,  # Disable for faster training
    'pin_memory': False,
    'prefetch_factor': 1,
    'persistent_workers': False
}

# BERT Fine-tuning configuration
BERT_CONFIG = {
    'model_name': 'vinai/phobert-base',  # Default PhoBERT
    'max_length': 96,  # Max sequence length (↓ memory) - reduced from 128
    'hidden_dim': 768,  # BERT hidden size
    'num_classes': 2,
    'dropout_rate': 0.1,  # Lower dropout for fine-tuning
    'freeze_layers': 6,  # 0 = train all, 6 = freeze first 6 layers
}

# Training parameters optimized for GPU memory
TRAINING_CONFIG = {
    'num_epochs': 3,  # Standard for BERT fine-tuning
    'batch_size': 1,  # Default batch size - adjust based on GPU memory
    'gradient_accumulation_steps': 16,  # Increased for better memory efficiency
    'learning_rate': 2e-5,  # Standard BERT learning rate
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'gradient_clip_norm': 1.0
}

# GPU-specific batch size recommendations (adjust TRAINING_CONFIG['batch_size'] above)
BATCH_SIZE_BY_GPU = {
    '4GB': 1,   # Very small batch for limited memory
    '6GB': 2,   # Conservative batch size  
    '8GB': 4,   # Recommended default
    '12GB': 8, # Good performance
    '24GB': 32, # High performance
}

# Data balancing configuration
BALANCE_CONFIG = {
    'strategy': 'smotetomek',  # SMOTETomek balancing
    'target_ratio': 0.65,  # 65% Real, 35% Fake
    'random_state': 42
}

# Multimodal fusion configuration
FUSION_CONFIG = {
    'fusion_type': 'gated',  # 'concat', 'attention', 'gated'
    'use_domain': True,
    'domain_regularization': True,
    'max_domain_weight': 0.15,  # Limit domain influence to 15%
    'domain_penalty_weight': 0.5
}

# Data split parameters
DATA_SPLIT = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42
}

# File paths (Update these for your data)
DATA_PATHS = {
    'real_file': 'path/to/real_articles.csv',
    'fake_file': 'path/to/fake_articles.csv'
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'log_gpu_memory': True,
    'save_model_checkpoints': True,
    'early_stopping_patience': 3,
    'memory_efficient_evaluation': True,  # Enable memory optimizations
    'cache_clear_interval': 50,  # Clear cache every N batches
    'cpu_fallback': True  # Allow CPU fallback for evaluation
}

# # Trusted domains for credibility scoring
# TRUSTED_DOMAINS = [
#     'vnexpress.net', 'tuoitre.vn', 'dantri.com.vn', 'thanhnien.vn',
#     'vtv.vn', 'vov.vn', 'baomoi.com', 'tienphong.vn',
#     'nld.com.vn', 'laodong.vn', 'vietnamnet.vn', 'zing.vn',
#     'kenh14.vn', 'soha.vn', 'cafef.vn', 'dantri.com',
#     'suckhoedoisong.vn', 'giaoduc.net.vn', 'plo.vn',
#     'nguoiduatin.vn', 'baotintuc.vn', 'cand.com.vn'
# ]