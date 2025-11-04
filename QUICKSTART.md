# Quick Start Guide - GraphCodeBERT Vulnerability Detection

## 🚀 Bắt đầu nhanh

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài đặt từng package:
```bash
pip install torch transformers numpy scikit-learn tqdm tensorboard
```

### 2. Chuẩn bị dữ liệu

Đảm bảo bạn có cấu trúc thư mục như sau:

```
train/
├── output/                    # Vulnerable code samples
│   ├── Buffer_Overflow/
│   │   ├── Buffer_Overflow_0001_vul.json
│   │   ├── Buffer_Overflow_0002_vul.json
│   │   └── ...
│   ├── Command_Injection/
│   ├── Path_Traversal/
│   └── SQL_Injection/
│
└── output_safe/               # Safe code samples
    ├── Buffer_Overflow/
    ├── Command_Injection/
    ├── Path_Traversal/
    └── SQL_Injection/
```

### 3. Tokenize dữ liệu

```bash
python graphcodebert_tokenizer.py
```

Output sẽ được lưu trong `processed_graphcodebert/`:
- `train.pkl` - Training set (70%)
- `val.pkl` - Validation set (15%)
- `test.pkl` - Test set (15%)
- `metadata.json` - Statistics

### 4. Train model

```bash
python train_graphcodebert.py
```

Hoặc chạy demo pipeline (test từng bước):
```bash
python demo_pipeline.py
```

### 5. Monitor training

```bash
tensorboard --logdir logs/graphcodebert
```

Truy cập: http://localhost:6006

## 📊 Ví dụ Output

### Tokenization Output:
```
==========================================
TOKENIZATION STATISTICS
==========================================
Total samples: 1234
  Vulnerable: 617
  Safe: 617

Detailed breakdown:
  vulnerable_Buffer_Overflow    : 150
  vulnerable_Command_Injection  : 150
  vulnerable_Path_Traversal     : 150
  vulnerable_SQL_Injection      : 167
  safe_Buffer_Overflow          : 150
  ...

Average statistics:
  Tokens per sample: 128.5
  Nodes per sample: 45.2
  Edges per sample: 67.8
```

### Training Output:
```
Epoch 1/10:
  Train - Loss: 0.5234, Acc: 0.7654, F1: 0.7432
  Val   - Loss: 0.4876, Acc: 0.7891, F1: 0.7765, AUC: 0.8234
  ✅ Saved best model (F1: 0.7765)

Epoch 2/10:
  Train - Loss: 0.4123, Acc: 0.8234, F1: 0.8156
  Val   - Loss: 0.4234, Acc: 0.8123, F1: 0.8089, AUC: 0.8567
  ✅ Saved best model (F1: 0.8089)
```

## ⚙️ Customization

### Thay đổi hyperparameters:

Edit trong `train_graphcodebert.py`:

```python
CONFIG = {
    # Model
    'model_type': 'gnn',           # Thay đổi thành 'gnn' để dùng GNN
    'use_dfg': True,               # False để tắt DFG attention
    
    # Training
    'batch_size': 32,              # Tăng nếu có GPU mạnh
    'learning_rate': 5e-5,         # Experiment với learning rate
    'num_epochs': 20,              # Tăng số epochs
    
    # Tokenization
    'max_code_length': 512,        # Tăng để xử lý code dài hơn
    'max_dfg_length': 128,         # Tăng số DFG nodes
}
```

### Train với custom data:

```python
from graphcodebert_tokenizer import GraphCodeBERTDataTokenizer

tokenizer = GraphCodeBERTDataTokenizer(
    max_code_len=256,
    max_dfg_len=64
)

# Process một file
tokenized = tokenizer.process_json_file(
    'path/to/your/file.json',
    label=1,  # 1=vulnerable, 0=safe
    vuln_type=0  # Optional: vulnerability type ID
)
```

## 🔧 Troubleshooting

### Problem: Out of Memory

**Solution 1**: Giảm batch size
```python
CONFIG['batch_size'] = 8  # hoặc 4
```

**Solution 2**: Giảm sequence length
```python
CONFIG['max_code_length'] = 128
CONFIG['max_dfg_length'] = 32
```

**Solution 3**: Gradient accumulation
```python
# Trong train_graphcodebert.py, thêm:
accumulation_steps = 4
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Problem: Imbalanced classes

**Solution**: Class weights
```python
# Trong Trainer.__init__:
from torch.nn import CrossEntropyLoss
weights = torch.tensor([1.0, 2.0])  # Tăng weight cho vulnerable class
self.criterion = CrossEntropyLoss(weight=weights.to(self.device))
```

### Problem: Low accuracy

**Solutions**:
1. Tăng số epochs: `CONFIG['num_epochs'] = 20`
2. Thử learning rate khác: `CONFIG['learning_rate'] = 1e-5` hoặc `5e-5`
3. Enable DFG attention: `CONFIG['use_dfg'] = True`
4. Thử GNN model: `CONFIG['model_type'] = 'gnn'`
5. Data augmentation (thêm code sau):

```python
# Augment bằng cách mask random tokens
def augment_tokens(tokens, mask_prob=0.1):
    import random
    return [
        '<mask>' if random.random() < mask_prob else token 
        for token in tokens
    ]
```

## 📈 Evaluation

### Load best model và evaluate:

```python
import torch
from graphcodebert_model import create_model
from graphcodebert_dataset import create_dataloader

# Load model
model = create_model(model_type='simple', num_labels=2)
checkpoint = torch.load('models/graphcodebert_vuln_detector/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
test_loader = create_dataloader('processed_graphcodebert/test.pkl', batch_size=32)

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
```

## 🎯 Best Practices

1. **Start simple**: Dùng `model_type='simple'` trước
2. **Monitor overfitting**: Theo dõi train vs val metrics
3. **Use validation set**: Để early stopping và hyperparameter tuning
4. **Save checkpoints**: Model được save mỗi epoch
5. **TensorBoard**: Luôn monitor training với TensorBoard
6. **Experiment**: Try different configs và document results

## 📚 Next Steps

1. **Hyperparameter tuning**: Grid search hoặc Optuna
2. **Ensemble models**: Combine multiple models
3. **Attention visualization**: Xem model focus vào đâu
4. **Cross-validation**: K-fold CV cho robust evaluation
5. **Multi-task learning**: Train cả binary và multi-class cùng lúc

## 💡 Tips

- **GPU**: Nếu có GPU, training sẽ nhanh hơn ~10x
- **Batch size**: Tăng batch size nếu có memory
- **Learning rate**: 2e-5 là good default cho fine-tuning
- **Epochs**: 10-20 epochs thường đủ
- **Validation**: Check validation metrics sau mỗi epoch
- **Save often**: Save checkpoints để không mất progress

## 📞 Support

Nếu gặp vấn đề:
1. Check error messages carefully
2. Verify data format với example JSON
3. Test từng component riêng (tokenizer, dataset, model)
4. Run `demo_pipeline.py` để debug từng bước

---

Happy training! 🚀
