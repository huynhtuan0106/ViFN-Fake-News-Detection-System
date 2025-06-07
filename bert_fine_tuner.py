#!/usr/bin/env python3
"""
BERT/PhoBERT Fine-tuning module for Vietnamese Fake News Detection
Supports multimodal fusion with domain regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from config import DEVICE, GPU_CONFIG
import warnings
warnings.filterwarnings('ignore')

class FineTunedBERTEmbedder(nn.Module):
    """
    Fine-tuned BERT/PhoBERT for Vietnamese fake news detection
    Supports end-to-end training
    """
    
    def __init__(self, model_name='vinai/phobert-base', max_length=128, 
                 freeze_layers=0, dropout_rate=0.1):
        super(FineTunedBERTEmbedder, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.freeze_layers = freeze_layers
        
        # Load pre-trained model
        print(f"Loading {model_name} for fine-tuning...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Additional layers for fine-tuning
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = self.config.hidden_size
        
        print(f"Model loaded: {model_name}")
        print(f"   - Hidden size: {self.hidden_dim}")
        print(f"   - Max length: {max_length}")
        print(f"   - Frozen layers: {freeze_layers}")
    
    def _freeze_layers(self, freeze_layers):
        """Freeze early BERT layers"""
        modules_to_freeze = [
            self.bert.embeddings,
            *self.bert.encoder.layer[:freeze_layers]
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        print(f"Frozen first {freeze_layers} layers")
    
    def encode_texts(self, texts, batch_size=16):
        """
        Encode texts with fine-tuned BERT
        Returns embeddings for downstream fusion
        """
        self.eval()
        embeddings = []
        
        print(f"Encoding {len(texts)} texts with fine-tuned BERT...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(DEVICE)
                attention_mask = encoded['attention_mask'].to(DEVICE)
                
                # Get embeddings
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use CLS token + dropout
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = self.dropout(cls_embeddings)
                
                embeddings.extend(cls_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass for training"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return cls_output

class MultimodalBERTFusionModel(nn.Module):
    """
    Multimodal fusion model with fine-tuned BERT/PhoBERT
    Combines title, summary, content, and domain features
    """
    
    def __init__(self, bert_model_name='vinai/phobert-base', 
                 domain_dim=None, fusion_type='attention',
                 max_length=128, freeze_bert_layers=0,
                 use_domain_regularization=True,
                 max_domain_weight=0.15):
        super(MultimodalBERTFusionModel, self).__init__()
        
        self.fusion_type = fusion_type
        self.use_domain_regularization = use_domain_regularization
        self.max_domain_weight = max_domain_weight
        
        # Fine-tuned BERT for each modality
        self.title_bert = FineTunedBERTEmbedder(
            bert_model_name, max_length, freeze_bert_layers
        )
        self.summary_bert = FineTunedBERTEmbedder(
            bert_model_name, max_length, freeze_bert_layers
        )
        self.content_bert = FineTunedBERTEmbedder(
            bert_model_name, max_length, freeze_bert_layers
        )
        
        # Shared tokenizer
        self.tokenizer = self.title_bert.tokenizer
        self.bert_hidden_dim = self.title_bert.hidden_dim
        
        # Domain processor
        if domain_dim:
            self.domain_processor = nn.Sequential(
                nn.Linear(domain_dim, self.bert_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.bert_hidden_dim // 4, self.bert_hidden_dim // 4)
            )
            self.domain_dim = self.bert_hidden_dim // 4
        else:
            self.domain_processor = None
            self.domain_dim = 0
        
        # Fusion layers
        self._setup_fusion_layers()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.bert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.bert_hidden_dim // 2, 2)  # Binary classification
        )
        
        # Domain regularization tracking
        self.current_domain_penalty = 0.0
        self.attention_weights_history = []
        
        print(f"MultimodalBERTFusionModel initialized:")
        print(f"   - BERT model: {bert_model_name}")
        print(f"   - Fusion type: {fusion_type}")
        print(f"   - Domain regularization: {use_domain_regularization}")
        print(f"   - Max domain weight: {max_domain_weight:.1%}")
    
    def _setup_fusion_layers(self):
        """Setup fusion mechanism"""
        if self.fusion_type == 'concat':
            self.fusion_dim = self.bert_hidden_dim * 3 + self.domain_dim
            
        elif self.fusion_type == 'attention':
            # Learnable attention weights
            num_modalities = 4 if self.domain_dim > 0 else 3
            self.attention_weights = nn.Parameter(torch.randn(num_modalities))
            self.fusion_dim = self.bert_hidden_dim
            
        elif self.fusion_type == 'gated':
            concat_dim = self.bert_hidden_dim * 3 + self.domain_dim
            self.gate = nn.Sequential(
                nn.Linear(concat_dim, concat_dim),
                nn.Sigmoid()
            )
            self.fusion_dim = concat_dim
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def tokenize_texts(self, texts):
        """Tokenize texts for BERT input"""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.title_bert.max_length,
            return_tensors='pt'
        )
        
        return encoded['input_ids'].to(DEVICE), encoded['attention_mask'].to(DEVICE)
    
    def forward(self, title_texts, summary_texts, content_texts, domain_features=None):
        """Forward pass"""
        # Tokenize texts
        title_ids, title_mask = self.tokenize_texts(title_texts)
        summary_ids, summary_mask = self.tokenize_texts(summary_texts)
        content_ids, content_mask = self.tokenize_texts(content_texts)
        
        # Get BERT embeddings
        title_emb = self.title_bert(title_ids, title_mask)
        summary_emb = self.summary_bert(summary_ids, summary_mask)
        content_emb = self.content_bert(content_ids, content_mask)
        
        # Process domain features
        if domain_features is not None and self.domain_processor:
            domain_emb = self.domain_processor(domain_features)
            modalities = [title_emb, summary_emb, content_emb, domain_emb]
        else:
            modalities = [title_emb, summary_emb, content_emb]
        
        # Apply fusion
        fused = self._apply_fusion(modalities)
        
        # Classification
        output = self.classifier(fused)
        return output
    
    def _apply_fusion(self, modalities):
        """Apply fusion strategy with domain regularization"""
        if self.fusion_type == 'concat':
            return torch.cat(modalities, dim=1)
            
        elif self.fusion_type == 'attention':
            return self._attention_fusion_with_regularization(modalities)
            
        elif self.fusion_type == 'gated':
            concat_features = torch.cat(modalities, dim=1)
            gate_weights = self.gate(concat_features)
            return concat_features * gate_weights
    
    def _attention_fusion_with_regularization(self, modalities):
        """Attention fusion with domain weight regularization"""
        # Compute attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        # Domain regularization
        if (self.use_domain_regularization and 
            len(modalities) == 4 and self.training):
            
            domain_weight = weights[3]  # Domain is last modality
            
            if domain_weight > self.max_domain_weight:
                excess = domain_weight - self.max_domain_weight
                self.current_domain_penalty = (excess ** 2)
            else:
                self.current_domain_penalty = 0.0
            
            # Log for analysis
            self.attention_weights_history.append(
                weights.detach().cpu().numpy()
            )
        
        # Pad modalities to same dimension if needed
        max_dim = max(mod.size(1) for mod in modalities)
        padded_modalities = []
        
        for mod in modalities:
            if mod.size(1) < max_dim:
                padding = torch.zeros(
                    mod.size(0), max_dim - mod.size(1), 
                    device=mod.device
                )
                mod = torch.cat([mod, padding], dim=1)
            padded_modalities.append(mod)
        
        # Apply attention
        stacked = torch.stack(padded_modalities, dim=1)
        attended = torch.sum(stacked * weights.view(1, -1, 1), dim=1)
        
        return attended
    
    def get_current_penalty(self):
        """Get current domain penalty"""
        return self.current_domain_penalty
    
    def get_attention_weights(self):
        """Get current attention weights"""
        if hasattr(self, 'attention_weights'):
            return F.softmax(self.attention_weights, dim=0).detach().cpu().numpy()
        return None

def create_multimodal_bert_model(domain_dim=None, 
                                bert_model='vinai/phobert-base',
                                fusion_type='attention',
                                freeze_bert_layers=0,
                                use_domain_regularization=True):
    """Factory function to create multimodal BERT model"""
    
    # Import max_length from config
    from config import BERT_CONFIG
    max_length = BERT_CONFIG.get('max_length', 128)
    
    model = MultimodalBERTFusionModel(
        bert_model_name=bert_model,
        domain_dim=domain_dim,
        fusion_type=fusion_type,
        max_length=max_length,
        freeze_bert_layers=freeze_bert_layers,
        use_domain_regularization=use_domain_regularization
    )
    
    # ⬆️ Bật gradient checkpointing cho tất cả BERT models nếu có trong config
    from config import GPU_CONFIG
    if GPU_CONFIG.get('gradient_checkpointing', False):
        print("Enabling gradient checkpointing for memory optimization...")
        
        # Enable gradient checkpointing for all BERT models
        if hasattr(model.title_bert, 'bert'):
            model.title_bert.bert.gradient_checkpointing_enable()
        if hasattr(model.summary_bert, 'bert'):
            model.summary_bert.bert.gradient_checkpointing_enable()
        if hasattr(model.content_bert, 'bert'):
            model.content_bert.bert.gradient_checkpointing_enable()
            
        print("✅ Gradient checkpointing enabled for all BERT models")
    
    return model.to(DEVICE)

def prepare_bert_optimizer(model, learning_rate=2e-5, 
                          warmup_steps=1000, total_steps=10000):
    """Prepare optimizer for BERT fine-tuning"""
    
    # Different learning rates for BERT and classification layers
    bert_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': learning_rate},
        {'params': classifier_params, 'lr': learning_rate * 5}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

if __name__ == "__main__":
    print("Testing Multimodal BERT Fine-tuning...")
    
    # Test model creation
    model = create_multimodal_bert_model(
        domain_dim=50,
        bert_model='vinai/phobert-base',
        fusion_type='attention'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    title_texts = ["Tin tức giả mạo", "Tin tức thật"]
    summary_texts = ["Tóm tắt tin giả", "Tóm tắt tin thật"]
    content_texts = ["Nội dung tin giả rất dài", "Nội dung tin thật rất dài"]
    domain_features = torch.randn(batch_size, 50).to(DEVICE)
    
    output = model(title_texts, summary_texts, content_texts, domain_features)
    print(f"Output shape: {output.shape}")
    print("Model test completed!") 