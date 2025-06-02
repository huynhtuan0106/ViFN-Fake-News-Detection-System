import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class LateFusionModel(nn.Module):
    """Late Fusion model for multimodal fake news detection"""
    
    def __init__(self, input_dims, hidden_dim=256, num_classes=2, fusion_type='concat'):
        super(LateFusionModel, self).__init__()
        self.fusion_type = fusion_type
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Individual field processors
        self.title_processor = self._create_field_processor(
            input_dims['title'], hidden_dim
        )
        self.summary_processor = self._create_field_processor(
            input_dims['summary'], hidden_dim
        )
        self.content_processor = self._create_field_processor(
            input_dims['content'], hidden_dim
        )
        self.domain_processor = self._create_field_processor(
            input_dims['domain'], hidden_dim // 4
        )
        
        # Fusion layers
        self._setup_fusion_layers(hidden_dim)
        
        # Classification head
        self.classifier = self._create_classifier(self.fusion_dim, hidden_dim, num_classes)
    
    def _create_field_processor(self, input_dim, output_dim):
        """Create a field processor network"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(output_dim, output_dim // 2)
        )
    
    def _setup_fusion_layers(self, hidden_dim):
        """Setup fusion layers based on fusion type"""
        if self.fusion_type == 'concat':
            # FIX: Tính chính xác fusion dimension dựa trên output thực tế của processors
            title_out_dim = hidden_dim // 2  # 128
            summary_out_dim = hidden_dim // 2  # 128  
            content_out_dim = hidden_dim // 2  # 128
            domain_out_dim = (hidden_dim // 4) // 2  # 32 (do có 2 linear layers)
            
            self.fusion_dim = title_out_dim + summary_out_dim + content_out_dim + domain_out_dim
            # = 128 + 128 + 128 + 32 = 416 ✅
            self.fusion_layer = nn.Identity()
            
        elif self.fusion_type == 'attention':
            self.attention_weights = nn.Parameter(torch.randn(4))
            self.fusion_dim = hidden_dim // 2
            
        elif self.fusion_type == 'gated':
            # FIX: Cũng cần sửa cho gated fusion
            title_out_dim = hidden_dim // 2
            summary_out_dim = hidden_dim // 2  
            content_out_dim = hidden_dim // 2
            domain_out_dim = (hidden_dim // 4) // 2
            
            concat_dim = title_out_dim + summary_out_dim + content_out_dim + domain_out_dim
            self.gate = nn.Sequential(
                nn.Linear(concat_dim, concat_dim),  # 416 -> 416 ✅
                nn.Sigmoid()
            )
            self.fusion_dim = concat_dim  # Sử dụng concat_dim thay vì hidden_dim
    
    def _create_classifier(self, fusion_dim, hidden_dim, num_classes):
        """Create classification head"""
        return nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, title, summary, content, domain):
        """Forward pass"""
        # Process individual fields
        title_emb = self.title_processor(title)
        summary_emb = self.summary_processor(summary)
        content_emb = self.content_processor(content)
        domain_emb = self.domain_processor(domain)
        
        # Fusion
        fused = self._apply_fusion(title_emb, summary_emb, content_emb, domain_emb)
        
        # Classification
        output = self.classifier(fused)
        return output
    
    def _apply_fusion(self, title_emb, summary_emb, content_emb, domain_emb):
        """Apply fusion strategy"""
        if self.fusion_type == 'concat':
            return torch.cat([title_emb, summary_emb, content_emb, domain_emb], dim=1)
        
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            weights = F.softmax(self.attention_weights, dim=0)
            # Pad domain embedding to match others
            domain_padded = F.pad(domain_emb, (0, title_emb.size(1) - domain_emb.size(1)))
            embeddings = torch.stack([title_emb, summary_emb, content_emb, domain_padded], dim=1)
            return torch.sum(embeddings * weights.view(1, 4, 1), dim=1)
        
        elif self.fusion_type == 'gated':
            # Gated fusion
            concat_features = torch.cat([title_emb, summary_emb, content_emb, domain_emb], dim=1)
            gate_weights = self.gate(concat_features)
            return concat_features * gate_weights

class SelfAttentionFusion(nn.Module):
    """Self-attention based fusion module"""
    
    def __init__(self, input_dim, num_heads=8):
        super(SelfAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, embeddings):
        """
        embeddings: [batch_size, num_fields, embedding_dim]
        """
        attn_output, _ = self.multihead_attn(embeddings, embeddings, embeddings)
        return self.norm(attn_output + embeddings)

class AdvancedLateFusionModel(LateFusionModel):
    """Advanced Late Fusion model with self-attention"""
    
    def __init__(self, input_dims, hidden_dim=256, num_classes=2, fusion_type='self_attention'):
        super().__init__(input_dims, hidden_dim, num_classes, 'concat')  # Initialize base
        
        if fusion_type == 'self_attention':
            self.self_attention = SelfAttentionFusion(hidden_dim // 2)
            self.fusion_type = 'self_attention'
            self.fusion_dim = hidden_dim // 2
    
    def _apply_fusion(self, title_emb, summary_emb, content_emb, domain_emb):
        """Apply advanced fusion strategies"""
        if self.fusion_type == 'self_attention':
            # Pad domain embedding
            domain_padded = F.pad(domain_emb, (0, title_emb.size(1) - domain_emb.size(1)))
            # Stack embeddings
            embeddings = torch.stack([title_emb, summary_emb, content_emb, domain_padded], dim=1)
            # Apply self-attention
            attended = self.self_attention(embeddings)
            # Global average pooling
            return torch.mean(attended, dim=1)
        else:
            return super()._apply_fusion(title_emb, summary_emb, content_emb, domain_emb)

def create_model(input_dims, fusion_type='concat', advanced=False):
    """Factory function to create models"""
    if advanced and fusion_type == 'self_attention':
        return AdvancedLateFusionModel(
            input_dims, 
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_classes=MODEL_CONFIG['num_classes'],
            fusion_type=fusion_type
        )
    else:
        return LateFusionModel(
            input_dims,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_classes=MODEL_CONFIG['num_classes'],
            fusion_type=fusion_type
        )

if __name__ == "__main__":
    # Test model creation
    input_dims = {'title': 100, 'summary': 100, 'content': 100, 'domain': 10}
    
    for fusion_type in ['concat', 'attention', 'gated']:
        model = create_model(input_dims, fusion_type)
        print(f"{fusion_type.capitalize()} Fusion Model created successfully!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")