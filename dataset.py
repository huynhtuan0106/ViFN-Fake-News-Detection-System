import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from config import DATA_SPLIT, MODEL_CONFIG

class FakeNewsDataset(Dataset):
    """Custom dataset for fake news detection"""
    
    def __init__(self, title_features, summary_features, content_features, domain_features, labels):
        self.title_features = torch.FloatTensor(title_features)
        self.summary_features = torch.FloatTensor(summary_features)
        self.content_features = torch.FloatTensor(content_features)
        self.domain_features = torch.FloatTensor(domain_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'title': self.title_features[idx],
            'summary': self.summary_features[idx],
            'content': self.content_features[idx],
            'domain': self.domain_features[idx],
            'label': self.labels[idx]
        }

def create_data_splits(title_features, summary_features, content_features, domain_features, labels):
    """Create train/test splits with stratification"""
    
    print("\n" + "="*60)
    print("CREATING DATA SPLITS")
    print("="*60)
    
    # Split into train and test
    X_title_train, X_title_test, X_summary_train, X_summary_test, \
    X_content_train, X_content_test, X_domain_train, X_domain_test, \
    y_train, y_test = train_test_split(
        title_features, summary_features, content_features, domain_features, labels,
        test_size=DATA_SPLIT['test_size'], 
        random_state=DATA_SPLIT['random_state'], 
        stratify=labels
    )
    
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    return (X_title_train, X_title_test, X_summary_train, X_summary_test,
            X_content_train, X_content_test, X_domain_train, X_domain_test,
            y_train, y_test)

def create_data_loaders(X_title_train, X_title_test, X_summary_train, X_summary_test,
                       X_content_train, X_content_test, X_domain_train, X_domain_test,
                       y_train, y_test, validation_split=True):
    """Create PyTorch data loaders"""
    
    # Create datasets
    train_dataset = FakeNewsDataset(
        X_title_train, X_summary_train, X_content_train, X_domain_train, y_train
    )
    test_dataset = FakeNewsDataset(
        X_title_test, X_summary_test, X_content_test, X_domain_test, y_test
    )
    
    # Create validation split if needed
    if validation_split:
        train_size = int((1 - DATA_SPLIT['val_size']) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=True
        )
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=MODEL_CONFIG['batch_size'], 
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Dataset module for PyTorch data loading.")
    print("Use this through main_real_data.py with processed data.")