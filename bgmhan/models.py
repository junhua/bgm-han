import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidual(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.transform = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        transformed = self.layer_norm(x)
        transformed = self.transform(transformed)
        return residual + self.gate * transformed

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.qkv_proj = nn.Linear(input_dim, 3 * input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.output_proj(out)
        
        return out, attn

class BGMHAN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_fields=4, num_heads=16, dropout=0.1):
        super().__init__()
        self.num_fields = num_fields
        self.input_dim = input_dim
        self.hidden_dim = (hidden_dim // num_heads) * num_heads

        # Field processing layers
        self.field_processors = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(input_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(input_dim),
                'gate1': GatedResidual(input_dim, dropout),
                'norm2': nn.LayerNorm(input_dim),
                'projection': nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_fields)
        ])

        # Cross-attention mechanism
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_gate = GatedResidual(hidden_dim, dropout)

        # Final layers
        self.final_combine = nn.Sequential(
            nn.Linear(hidden_dim * num_fields, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )

        self.classifier = nn.Linear(hidden_dim // 2, 2)

    def _process_field(self, field_input, processor):
        """Process a single field through its attention and projection layers."""
        batch_size = field_input.size(0)

        # First attention layer
        normed = processor['norm1'](field_input)
        attended, _ = processor['attention'](normed)
        gated = processor['gate1'](attended, normed)

        # Project to hidden dimension
        normed = processor['norm2'](gated)
        projected = processor['projection'](normed)

        return projected

    def forward(self, x):
        batch_size = x.size(0)

        # Process each field
        field_embeddings = []
        for field_idx in range(self.num_fields):
            field_input = x[:, field_idx, :, :, :]
            # Reshape to combine sentence and token dimensions
            field_input = field_input.reshape(batch_size, -1, self.input_dim)
            # Process through field-specific layers
            field_embedding = self._process_field(field_input, self.field_processors[field_idx])
            # Pool the sequence dimension
            field_embedding = field_embedding.mean(dim=1)
            field_embeddings.append(field_embedding)

        # Stack field embeddings for cross-attention
        field_tensor = torch.stack(field_embeddings, dim=1)

        # Apply cross-attention between fields
        normed = self.cross_norm(field_tensor)
        cross_attended, _ = self.cross_attention(normed)
        field_interactions = self.cross_gate(cross_attended, normed)

        # Combine and classify
        combined = field_interactions.reshape(batch_size, -1)
        hidden = self.final_combine(combined)
        logits = self.classifier(hidden)

        return logits
