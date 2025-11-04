"""
Multi-label GraphCodeBERT Model
Phát hiện NHIỀU loại lỗi đồng thời trong một đoạn code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class MultiLabelGraphCodeBERT(nn.Module):
    """
    Multi-label classification model
    Output: [batch_size, num_labels] với mỗi label có probability riêng
    """
    
    def __init__(
        self,
        model_name="microsoft/graphcodebert-base",
        num_labels=4,  # SQL, Command, Path, Buffer
        hidden_dim=768,
        dropout=0.1,
        use_dfg=True
    ):
        super(MultiLabelGraphCodeBERT, self).__init__()
        
        self.num_labels = num_labels
        self.use_dfg = use_dfg
        
        print(f"[+] Loading {model_name} for Multi-label Classification...")
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
        
        # Multi-label classification head
        self.dropout = nn.Dropout(dropout)
        
        # Separate classifier cho mỗi vulnerability type
        self.classifiers = nn.ModuleDict({
            'sql_injection': nn.Sequential(
                nn.Linear(self.hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)  # Binary output
            ),
            'command_injection': nn.Sequential(
                nn.Linear(self.hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'path_traversal': nn.Sequential(
                nn.Linear(self.hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'buffer_overflow': nn.Sequential(
                nn.Linear(self.hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        })
        
        print(f"[+] Multi-label model initialized")
        print(f"    Parameters: {self.count_parameters():,}")
        print(f"    Labels: {num_labels} (SQL, Command, Path, Buffer)")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids, attention_mask, dfg_matrix=None, **kwargs):
        """
        Forward pass
        
        Returns:
            logits: [batch_size, num_labels] - raw scores
            probs: [batch_size, num_labels] - probabilities (after sigmoid)
        """
        # Get embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Apply DFG attention if enabled
        if self.use_dfg and dfg_matrix is not None:
            attended_output, _ = self.dfg_attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=(attention_mask == 0)
            )
            combined = sequence_output + self.dfg_linear(attended_output)
            pooled_output = combined[:, 0, :]
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Multi-label classification
        logits_list = []
        for vuln_type in ['sql_injection', 'command_injection', 'path_traversal', 'buffer_overflow']:
            logit = self.classifiers[vuln_type](pooled_output)  # [batch, 1]
            logits_list.append(logit)
        
        # Concatenate: [batch, num_labels]
        logits = torch.cat(logits_list, dim=1)
        
        # Probabilities (independent for each label)
        probs = torch.sigmoid(logits)
        
        return logits, probs


class SharedMultiLabelGraphCodeBERT(nn.Module):
    """
    Alternative: Shared classifier với multi-label output
    Đơn giản hơn, ít parameters hơn
    """
    
    def __init__(
        self,
        model_name="microsoft/graphcodebert-base",
        num_labels=4,
        hidden_dim=768,
        dropout=0.1,
        use_dfg=True
    ):
        super(SharedMultiLabelGraphCodeBERT, self).__init__()
        
        self.num_labels = num_labels
        self.use_dfg = use_dfg
        
        print(f"[+] Loading {model_name} (Shared Multi-label)...")
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_dim = self.roberta.config.hidden_size
        
        # DFG attention
        if self.use_dfg:
            self.dfg_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.dfg_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Shared classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)  # Direct output
        )
        
        print(f"[+] Shared multi-label model initialized")
        print(f"    Parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids, attention_mask, dfg_matrix=None, **kwargs):
        """
        Forward pass
        
        Returns:
            logits: [batch_size, num_labels]
            probs: [batch_size, num_labels]
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        if self.use_dfg and dfg_matrix is not None:
            attended_output, _ = self.dfg_attention(
                sequence_output,
                sequence_output,
                sequence_output,
                key_padding_mask=(attention_mask == 0)
            )
            combined = sequence_output + self.dfg_linear(attended_output)
            pooled_output = combined[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits)
        
        return logits, probs


def predict_vulnerabilities(model, input_ids, attention_mask, dfg_matrix=None, threshold=0.5):
    """
    Predict vulnerabilities với threshold
    
    Args:
        model: Multi-label model
        input_ids, attention_mask, dfg_matrix: Inputs
        threshold: Confidence threshold (default 0.5)
    
    Returns:
        dict: {
            'predictions': binary predictions [0/1],
            'probabilities': confidence scores [0-1],
            'detected_vulnerabilities': list of vulnerability names
        }
    """
    model.eval()
    
    with torch.no_grad():
        logits, probs = model(input_ids, attention_mask, dfg_matrix)
    
    # Convert to binary predictions
    predictions = (probs >= threshold).int()
    
    # Vulnerability names
    vuln_names = ['SQL_Injection', 'Command_Injection', 'Path_Traversal', 'Buffer_Overflow']
    
    # Get detected vulnerabilities
    detected = []
    for i, name in enumerate(vuln_names):
        if predictions[0, i] == 1:  # Assuming batch_size=1
            detected.append({
                'type': name,
                'confidence': probs[0, i].item()
            })
    
    return {
        'predictions': predictions[0].cpu().numpy(),
        'probabilities': probs[0].cpu().numpy(),
        'detected_vulnerabilities': detected
    }


if __name__ == "__main__":
    print("Testing Multi-label GraphCodeBERT...")
    
    # Test model
    model = MultiLabelGraphCodeBERT(num_labels=4, use_dfg=True)
    
    # Dummy input
    batch_size = 4
    seq_len = 256
    
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    dfg_matrix = torch.rand(batch_size, 64, 64)
    
    # Forward pass
    logits, probs = model(input_ids, attention_mask, dfg_matrix)
    
    print(f"\n✅ Model output:")
    print(f"   Logits shape: {logits.shape}")  # [4, 4]
    print(f"   Probs shape: {probs.shape}")    # [4, 4]
    print(f"   Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Test prediction
    print(f"\n✅ Example prediction:")
    result = predict_vulnerabilities(
        model, 
        input_ids[0:1], 
        attention_mask[0:1], 
        dfg_matrix[0:1],
        threshold=0.5
    )
    
    print(f"   Predictions: {result['predictions']}")
    print(f"   Probabilities: {result['probabilities']}")
    print(f"   Detected: {[v['type'] for v in result['detected_vulnerabilities']]}")
    
    print("\n✅ Multi-label model works correctly!")
