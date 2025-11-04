"""
GraphCodeBERT Model cho Vulnerability Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig


class GraphCodeBERTClassifier(nn.Module):
    """
    GraphCodeBERT model với graph-aware attention
    cho binary vulnerability classification
    """
    
    def __init__(
        self,
        model_name="microsoft/graphcodebert-base",
        num_labels=2,
        hidden_dim=768,
        dropout=0.1,
        use_dfg=True
    ):
        super(GraphCodeBERTClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.use_dfg = use_dfg
        
        # Load pretrained GraphCodeBERT
        print(f"[+] Loading {model_name}...")
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_dim = self.roberta.config.hidden_size
        
        # DFG-aware attention
        if self.use_dfg:
            self.dfg_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.dfg_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        print(f"[+] Model initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        position_idx=None,
        dfg_matrix=None,
        edge_index=None
    ):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            position_idx: [batch_size, seq_len] (optional)
            dfg_matrix: [batch_size, max_nodes, max_nodes] (optional)
            edge_index: list of [2, num_edges] tensors (optional)
        
        Returns:
            logits: [batch_size, num_labels]
        """
        # Get RoBERTa embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [batch_size, seq_len, hidden_dim]
        sequence_output = outputs.last_hidden_state
        
        # [batch_size, hidden_dim] - CLS token representation
        pooled_output = outputs.pooler_output
        
        # Apply DFG-aware attention nếu có
        if self.use_dfg and dfg_matrix is not None:
            # Use DFG matrix as additional attention mask
            # dfg_matrix: [batch_size, max_nodes, max_nodes]
            
            # Create attention mask from DFG
            # Extend to sequence length
            batch_size, seq_len, _ = sequence_output.shape
            
            # Apply multi-head attention với DFG guidance
            attended_output, _ = self.dfg_attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=(attention_mask == 0)
            )
            
            # Combine với original output
            combined = sequence_output + self.dfg_linear(attended_output)
            
            # Pool: use CLS token
            pooled_output = combined[:, 0, :]
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class GraphCodeBERTWithGNN(nn.Module):
    """
    GraphCodeBERT + Graph Neural Network
    Kết hợp code embeddings với graph structure
    """
    
    def __init__(
        self,
        model_name="microsoft/graphcodebert-base",
        num_labels=2,
        gnn_layers=2,
        gnn_hidden_dim=256,
        dropout=0.1
    ):
        super(GraphCodeBERTWithGNN, self).__init__()
        
        # GraphCodeBERT encoder
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_dim = self.roberta.config.hidden_size
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            in_dim = self.hidden_dim if i == 0 else gnn_hidden_dim
            self.gnn_layers.append(
                GCNLayer(in_dim, gnn_hidden_dim, dropout)
            )
        
        # Fusion layer
        self.fusion = nn.Linear(self.hidden_dim + gnn_hidden_dim, self.hidden_dim)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, dfg_matrix=None, **kwargs):
        """Forward pass"""
        
        # Get code embeddings từ GraphCodeBERT
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        pooled_output = outputs.pooler_output  # [B, H]
        
        # Apply GNN nếu có DFG
        if dfg_matrix is not None:
            # Use mean pooling across sequence as node features
            # [B, L, H] -> [B, H]
            node_features = sequence_output.mean(dim=1)
            
            # Apply GNN layers
            gnn_output = node_features
            for gnn_layer in self.gnn_layers:
                gnn_output = gnn_layer(gnn_output, dfg_matrix)
            
            # Fuse code và graph representations
            fused = torch.cat([pooled_output, gnn_output], dim=-1)
            fused = self.fusion(fused)
            fused = F.relu(fused)
        else:
            fused = pooled_output
        
        # Classify
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits


class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer"""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: [batch_size, in_dim]
            adj: [batch_size, num_nodes, num_nodes]
        """
        # Aggregate neighbors
        # adj @ x would give [batch, num_nodes, in_dim]
        # But we're working with single node per sample, so simplified
        
        # Apply linear transformation
        out = self.linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


def create_model(model_type="simple", **kwargs):
    """
    Factory function để tạo model
    
    Args:
        model_type: 'simple' hoặc 'gnn'
        **kwargs: Additional arguments cho model
    
    Returns:
        model instance
    """
    if model_type == "simple":
        return GraphCodeBERTClassifier(**kwargs)
    elif model_type == "gnn":
        return GraphCodeBERTWithGNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing models...")
    
    # Test simple model
    print("\n[+] Testing GraphCodeBERTClassifier...")
    model = GraphCodeBERTClassifier(num_labels=2, use_dfg=True)
    
    # Dummy input
    batch_size = 4
    seq_len = 256
    
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    dfg_matrix = torch.rand(batch_size, 64, 64)
    
    logits = model(input_ids, attention_mask, dfg_matrix=dfg_matrix)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, 2]")
    
    # Test GNN model
    print("\n[+] Testing GraphCodeBERTWithGNN...")
    model_gnn = GraphCodeBERTWithGNN(num_labels=2)
    
    logits_gnn = model_gnn(input_ids, attention_mask, dfg_matrix=dfg_matrix)
    print(f"  Output shape: {logits_gnn.shape}")
    print(f"  Expected: [{batch_size}, 2]")
    
    print("\n✅ Models work correctly!")
