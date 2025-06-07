#!/usr/bin/env python3
"""
Dataset loader for BERT fine-tuning with multimodal fake news detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Model config not needed - using direct parameters

class MultimodalBERTDataset(Dataset):
    """
    Dataset for multimodal BERT fine-tuning
    Handles raw text inputs (no pre-computed embeddings)
    """
    
    def __init__(self, dataframe, domain_features=None, include_domain=True):
        """
        Args:
            dataframe: pandas DataFrame with columns: title_processed, summary_processed, content_processed, label
            domain_features: numpy array of domain features (optional)
            include_domain: whether to include domain features
        """
        self.df = dataframe.reset_index(drop=True)
        self.include_domain = include_domain and domain_features is not None
        
        # Text fields
        self.titles = self.df['title_processed'].fillna('').astype(str).tolist()
        self.summaries = self.df['summary_processed'].fillna('').astype(str).tolist()
        self.contents = self.df['content_processed'].fillna('').astype(str).tolist()
        self.labels = self.df['label'].values
        
        # Domain features (optional)
        if self.include_domain:
            self.domain_features = torch.FloatTensor(domain_features)
        else:
            self.domain_features = None
        
        print(f"MultimodalBERTDataset created:")
        print(f"   - Samples: {len(self.df)}")
        print(f"   - Include domain: {self.include_domain}")
        print(f"   - Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            'title': self.titles[idx],
            'summary': self.summaries[idx],
            'content': self.contents[idx],
            'label': torch.LongTensor([self.labels[idx]])[0]
        }
        
        if self.include_domain:
            item['domain'] = self.domain_features[idx]
        
        return item

def create_bert_data_splits(df, domain_features=None, test_size=0.2, val_size=0.2, 
                           random_state=42, include_domain=True):
    """
    Create train/val/test splits for BERT fine-tuning
    
    Args:
        df: DataFrame with processed text
        domain_features: Domain features array
        test_size: Test set proportion
        val_size: Validation set proportion (from train set)
        random_state: Random seed
        include_domain: Whether to include domain features
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    
    print("\n" + "="*60)
    print("CREATING BERT DATA SPLITS")
    print("="*60)
    
    # Split into train and test
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, 
        random_state=random_state,
        stratify=df['label'].values
    )
    
    # Split train into train and validation
    train_labels = df.iloc[train_indices]['label'].values
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size,
        random_state=random_state,
        stratify=train_labels
    )
    
    # Create DataFrames
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    # Split domain features
    train_domain = domain_features[train_indices] if domain_features is not None else None
    val_domain = domain_features[val_indices] if domain_features is not None else None
    test_domain = domain_features[test_indices] if domain_features is not None else None
    
    # Create datasets
    train_dataset = MultimodalBERTDataset(train_df, train_domain, include_domain)
    val_dataset = MultimodalBERTDataset(val_df, val_domain, include_domain)
    test_dataset = MultimodalBERTDataset(test_df, test_domain, include_domain)
    
    print(f"Train set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def bert_collate_fn(batch):
    """Custom collate function for BERT datasets"""
    # Separate different types of data
    titles = [item['title'] for item in batch]
    summaries = [item['summary'] for item in batch]
    contents = [item['content'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    collated = {
        'titles': titles,
        'summaries': summaries, 
        'contents': contents,
        'labels': labels
    }
    
    # Add domain features if available
    if 'domain' in batch[0]:
        domains = torch.stack([item['domain'] for item in batch])
        collated['domains'] = domains
    
    return collated

def create_bert_data_loaders(train_dataset, val_dataset, test_dataset, 
                            batch_size=1, num_workers=0):
    """
    Create DataLoaders for BERT training optimized for low memory
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        batch_size: Batch size (optimized for GPU memory)
        num_workers: Number of worker processes for data loading
    """
    
    # Import config và auto-adjust for num_workers=0
    from config import GPU_CONFIG
    
    # ⬆️ AUTO-ADJUST: Khi num_workers=0, disable multiprocessing-only features
    use_multiprocessing = num_workers > 0
    pin_memory = GPU_CONFIG['pin_memory'] if use_multiprocessing else False
    prefetch_factor = GPU_CONFIG['prefetch_factor'] if use_multiprocessing else None
    persistent_workers = GPU_CONFIG['persistent_workers'] if use_multiprocessing else False
    
    print(f"DataLoader Configuration:")
    print(f"  Num workers: {num_workers}")
    print(f"  Use multiprocessing: {use_multiprocessing}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Prefetch factor: {prefetch_factor}")
    print(f"  Persistent workers: {persistent_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=bert_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=bert_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=bert_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    print(f"\n✅ Data loaders created successfully:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory optimized for GPU: {'Yes' if not use_multiprocessing else 'No'}")
    
    return train_loader, val_loader, test_loader

def balance_bert_dataset(train_dataset, strategy='undersample'):
    """
    Balance dataset for BERT training
    
    Args:
        train_dataset: Training dataset
        strategy: 'undersample', 'oversample', or 'weighted'
    
    Returns:
        balanced_dataset or class_weights
    """
    
    labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\nOriginal distribution: {dict(zip(unique, counts))}")
    
    if strategy == 'undersample':
        # Undersample majority class
        min_count = min(counts)
        balanced_indices = []
        
        for label in unique:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            sampled_indices = np.random.choice(
                label_indices, size=min_count, replace=False
            )
            balanced_indices.extend(sampled_indices)
        
        # Create new dataset with balanced indices
        balanced_df = train_dataset.df.iloc[balanced_indices].reset_index(drop=True)
        balanced_domain = None
        if train_dataset.include_domain:
            balanced_domain = train_dataset.domain_features[balanced_indices]
        
        balanced_dataset = MultimodalBERTDataset(
            balanced_df, balanced_domain, train_dataset.include_domain
        )
        
        return balanced_dataset
    
    elif strategy == 'weighted':
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced', classes=unique, y=labels
        )
        return torch.FloatTensor(class_weights)
    
    else:
        return train_dataset

if __name__ == "__main__":
    print("🧪 Testing BERT Dataset...")
    
    # Create sample data
    sample_data = {
        'title_processed': ['Tin tức giả mạo', 'Tin tức thật', 'Tin khác'],
        'summary_processed': ['Tóm tắt giả', 'Tóm tắt thật', 'Tóm tắt khác'],
        'content_processed': ['Nội dung giả', 'Nội dung thật', 'Nội dung khác'],
        'label': [1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    domain_features = np.random.randn(3, 10)
    
    # Test dataset creation
    dataset = MultimodalBERTDataset(df, domain_features)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)
    batch = next(iter(loader))
    print(f"Batch: {batch}")
    
    print("BERT Dataset test completed!") 