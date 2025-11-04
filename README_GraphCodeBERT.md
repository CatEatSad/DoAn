# GraphCodeBERT Vulnerability Detection

Hệ thống tokenization và training cho GraphCodeBERT phục vụ phát hiện lỗ hổng bảo mật trong code.

## 📋 Mô tả

Dự án này cải tiến việc tokenize dữ liệu từ Joern CPG để train GraphCodeBERT cho bài toán phát hiện vulnerability (Buffer Overflow, Command Injection, Path Traversal, SQL Injection).

### Các cải tiến chính:

1. **Tokenization phù hợp với GraphCodeBERT**:
   - Extract code tokens từ AST nodes
   - Build Data Flow Graph (DFG) từ CFG, PDG edges
   - Tạo mapping từ DFG nodes tới code tokens
   - Attention mask và position embeddings

2. **Graph-aware processing**:
   - DFG adjacency matrix
   - Edge index cho GNN layers
   - Variable-to-token mapping
   - Node type preservation

3. **Flexible architecture**:
   - Simple GraphCodeBERT classifier
   - GraphCodeBERT + GNN hybrid model
   - Configurable hyperparameters

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📂 Cấu trúc dữ liệu

```
train/
├── output/                          # Vulnerable samples
│   ├── Buffer_Overflow/
│   ├── Command_Injection/
│   ├── Path_Traversal/
│   └── SQL_Injection/
├── output_safe/                     # Safe samples
│   ├── Buffer_Overflow/
│   ├── Command_Injection/
│   ├── Path_Traversal/
│   └── SQL_Injection/
├── processed_graphcodebert/        # Tokenized data (generated)
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── metadata.json
└── models/                          # Trained models (generated)
    └── graphcodebert_vuln_detector/
```

## 🔧 Usage

### 1. Tokenize dữ liệu

Chạy script tokenization để xử lý JSON files từ Joern:

```bash
python graphcodebert_tokenizer.py
```

Output:
- `processed_graphcodebert/tokenized_data.pkl`: Toàn bộ data đã tokenize
- `processed_graphcodebert/train.pkl`: Training set (70%)
- `processed_graphcodebert/val.pkl`: Validation set (15%)
- `processed_graphcodebert/test.pkl`: Test set (15%)
- `processed_graphcodebert/metadata.json`: Statistics

### 2. Train model

```bash
python train_graphcodebert.py
```

Hoặc customize config:

```python
CONFIG = {
    'model_type': 'simple',  # 'simple' or 'gnn'
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'use_dfg': True,
}
```

### 3. Test dataset loading

```bash
python graphcodebert_dataset.py
```

### 4. Test models

```bash
python graphcodebert_model.py
```

## 📊 Data Format

### Input JSON (từ Joern):

```json
{
  "functions": [
    {
      "function": "main",
      "file": "example.java",
      "id": "123",
      "AST": [...],
      "CFG": [...],
      "PDG": [...]
    }
  ]
}
```

### Tokenized Output:

```python
{
    'input_ids': [101, 2023, 2003, ...],           # Token IDs [max_len]
    'attention_mask': [1, 1, 1, ...],              # Attention mask
    'position_idx': [0, 1, 2, ...],                # Position indices
    'dfg_matrix': [[0, 1, 0], [1, 0, 1], ...],    # DFG adjacency [N, N]
    'edge_index': [[0, 1], [1, 2], ...],          # Edge list [2, E]
    'label': 1,                                    # 0=safe, 1=vulnerable
    'vuln_type': 0,                                # Vulnerability type ID
    'code_tokens': ['<s>', 'public', ...],        # Original tokens
}
```

## 🏗️ Architecture

### 1. Simple GraphCodeBERT Classifier

```
Input Code → RoBERTa Encoder → DFG Attention → Classifier → Binary Output
```

### 2. GraphCodeBERT + GNN

```
Code → RoBERTa → Code Embeddings ──┐
                                    ├→ Fusion → Classifier
DFG → GNN Layers → Graph Features ──┘
```

## 📈 Training Metrics

- **Loss**: CrossEntropyLoss
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Optimizer**: AdamW với learning rate warmup
- **Scheduler**: Linear decay

## 🔍 Model Details

### GraphCodeBERTClassifier
- Base: `microsoft/graphcodebert-base`
- Hidden dim: 768
- DFG-aware multi-head attention (8 heads)
- Classification head: 768 → 384 → 2

### GraphCodeBERTWithGNN
- Base: `microsoft/graphcodebert-base`
- GNN: 2-layer GCN
- GNN hidden: 256
- Fusion: Code + Graph features

## 📝 Configuration

Trong `train_graphcodebert.py`:

```python
CONFIG = {
    # Model
    'model_type': 'simple',        # 'simple' hoặc 'gnn'
    'model_name': 'microsoft/graphcodebert-base',
    'num_labels': 2,               # Binary classification
    'use_dfg': True,               # Sử dụng DFG attention
    
    # Training
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Tokenization
    'max_code_length': 256,
    'max_dfg_length': 64,
}
```

## 🎯 Results

Model sẽ được đánh giá trên:
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Tỷ lệ true positive trong predicted positive
- **Recall**: Tỷ lệ phát hiện được vulnerable samples
- **F1 Score**: Harmonic mean của Precision và Recall
- **AUC-ROC**: Area under ROC curve

## 📦 Output Files

### Models
- `models/graphcodebert_vuln_detector/best_model.pt`: Best model checkpoint
- `models/graphcodebert_vuln_detector/checkpoint_epoch_X.pt`: Epoch checkpoints

### Results
- `models/graphcodebert_vuln_detector/test_results.json`: Test metrics
- `logs/graphcodebert/`: TensorBoard logs

## 🔧 Troubleshooting

### Memory Issues
Giảm batch_size hoặc max_code_length:
```python
CONFIG['batch_size'] = 8
CONFIG['max_code_length'] = 128
```

### Imbalanced Data
Thêm class weights:
```python
from torch.nn import CrossEntropyLoss
weights = torch.tensor([1.0, 2.0])  # Tăng trọng số cho class thiểu số
criterion = CrossEntropyLoss(weight=weights)
```

## 📚 References

- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [Microsoft GraphCodeBERT](https://github.com/microsoft/CodeBERT)
- [Joern Documentation](https://joern.io/)

## 🤝 Contributing

Contributions welcome! Các cải tiến có thể thêm:
- [ ] Multi-task learning (binary + multi-class)
- [ ] Attention visualization
- [ ] Code explanation generation
- [ ] Cross-project evaluation

## 📄 License

MIT License
