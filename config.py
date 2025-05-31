import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_classes': 2,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'num_epochs': 15
}

# Text processing parameters
TEXT_CONFIG = {
    'max_features_tfidf': 3000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'max_length_bert': 256
}

# Embedding dimensions
EMBEDDING_CONFIG = {
    'bert_dim': 768,
    'tfidf_title_dim': 3000,
    'tfidf_summary_dim': 3000,
    'tfidf_content_dim': 3000,
    'domain_dim': None  # Will be set dynamically
}

# Trusted domains for credibility scoring
TRUSTED_DOMAINS = [
    'vnexpress.net', 'tuoitre.vn', 'vietnamnet.vn', 
    'dantri.com.vn', 'thanhnien.vn', 'zing.vn'
]

# Data split parameters
DATA_SPLIT = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42
}