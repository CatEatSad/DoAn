# 📦 Tổng hợp Files - GraphCodeBERT Vulnerability Detection

## 📁 Cấu trúc Project

```
train/
├── 🔧 Core Implementation
│   ├── graphcodebert_tokenizer.py      # Tokenization cho GraphCodeBERT
│   ├── graphcodebert_dataset.py        # PyTorch Dataset & DataLoader
│   ├── graphcodebert_model.py          # Model architectures
│   └── train_graphcodebert.py          # Training script
│
├── 📚 Documentation
│   ├── README_GraphCodeBERT.md         # Comprehensive README
│   ├── QUICKSTART.md                   # Quick start guide
│   ├── COMPARISON.md                   # So sánh với script cũ
│   └── SUMMARY.md                      # File này
│
├── 🧪 Testing & Demo
│   └── demo_pipeline.py                # Demo toàn bộ pipeline
│
├── ⚙️ Configuration
│   ├── requirements.txt                # Python dependencies
│   └── Script.py                       # Script gốc (reference)
│
├── 📊 Data (Input)
│   ├── output/                         # Vulnerable samples
│   │   ├── Buffer_Overflow/
│   │   ├── Command_Injection/
│   │   ├── Path_Traversal/
│   │   └── SQL_Injection/
│   └── output_safe/                    # Safe samples
│       └── (same structure)
│
└── 💾 Generated (Output)
    ├── processed_graphcodebert/        # Tokenized data
    │   ├── tokenized_data.pkl
    │   ├── train.pkl
    │   ├── val.pkl
    │   ├── test.pkl
    │   └── metadata.json
    ├── models/                          # Trained models
    │   └── graphcodebert_vuln_detector/
    │       ├── best_model.pt
    │       ├── checkpoint_epoch_X.pt
    │       └── test_results.json
    └── logs/                            # TensorBoard logs
        └── graphcodebert/
```

---

## 📄 File Descriptions

### 1️⃣ graphcodebert_tokenizer.py
**Mục đích**: Tokenize dữ liệu từ Joern JSON files

**Chức năng chính**:
- `GraphCodeBERTDataTokenizer`: Class xử lý tokenization
- `extract_code_tokens()`: Extract tokens từ AST nodes
- `extract_dfg_edges()`: Extract DFG từ graph edges
- `build_variable_mapping()`: Map variables tới token positions
- `tokenize_function()`: Tokenize một function
- `process_all_data()`: Process toàn bộ dataset
- `create_train_val_test_split()`: Split data

**Input**: JSON files từ Joern
**Output**: Pickle files với tokenized data

**Usage**:
```bash
python graphcodebert_tokenizer.py
```

---

### 2️⃣ graphcodebert_dataset.py
**Mục đích**: PyTorch Dataset và DataLoader

**Chức năng chính**:
- `GraphCodeBERTDataset`: Custom PyTorch Dataset
- `collate_batch()`: Custom collate function
- `create_dataloader()`: Factory function cho DataLoader
- `load_datasets()`: Load train/val/test sets

**Input**: Tokenized pickle files
**Output**: PyTorch DataLoader objects

**Usage**:
```python
from graphcodebert_dataset import load_datasets

dataloaders = load_datasets('processed_graphcodebert/', batch_size=32)
train_loader = dataloaders['train']
```

---

### 3️⃣ graphcodebert_model.py
**Mục đích**: Model architectures

**Models**:
1. **GraphCodeBERTClassifier** (Simple)
   - RoBERTa encoder
   - DFG-aware attention
   - Classification head
   
2. **GraphCodeBERTWithGNN** (Advanced)
   - RoBERTa encoder
   - GCN layers
   - Fusion layer
   - Classification head

**Usage**:
```python
from graphcodebert_model import create_model

model = create_model(
    model_type='simple',  # or 'gnn'
    num_labels=2,
    use_dfg=True
)
```

---

### 4️⃣ train_graphcodebert.py
**Mục đích**: Training script chính

**Chức năng**:
- `Trainer`: Training class với complete pipeline
- `train_epoch()`: Train một epoch
- `evaluate()`: Evaluate trên val/test set
- `train()`: Main training loop

**Features**:
- Learning rate scheduling
- Gradient clipping
- TensorBoard logging
- Checkpoint saving
- Best model tracking
- Metrics computation

**Usage**:
```bash
python train_graphcodebert.py
```

**Config trong file**:
```python
CONFIG = {
    'model_type': 'simple',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'use_dfg': True,
}
```

---

### 5️⃣ demo_pipeline.py
**Mục đích**: Test và demo toàn bộ pipeline

**Steps**:
1. Check requirements
2. Check data structure
3. Run tokenization
4. Load datasets
5. Create model
6. Demo training (1 epoch)

**Usage**:
```bash
python demo_pipeline.py
```

---

## 🔄 Workflow

### Step-by-step Process:

```
1. DATA PREPARATION
   ├─ Joern CPG extraction (manual)
   ├─ graph-for-funcs6.sc script
   └─ JSON files in output/

2. TOKENIZATION
   ├─ Run: graphcodebert_tokenizer.py
   ├─ Extract code tokens
   ├─ Build DFG mappings
   └─ Create train/val/test splits

3. DATASET CREATION
   ├─ GraphCodeBERTDataset class
   ├─ Custom collate function
   └─ DataLoader with batching

4. MODEL TRAINING
   ├─ Load pretrained GraphCodeBERT
   ├─ Fine-tune on vulnerability data
   ├─ Save best model
   └─ Evaluate on test set

5. EVALUATION
   ├─ Load best checkpoint
   ├─ Compute metrics
   └─ Save results
```

---

## 🎯 Quick Commands

### Complete pipeline:
```bash
# 1. Install
pip install -r requirements.txt

# 2. Tokenize
python graphcodebert_tokenizer.py

# 3. Train
python train_graphcodebert.py

# 4. Monitor
tensorboard --logdir logs/graphcodebert
```

### Test components:
```bash
# Test dataset loading
python graphcodebert_dataset.py

# Test model creation
python graphcodebert_model.py

# Demo full pipeline
python demo_pipeline.py
```

---

## 📊 Expected Output Sizes

| File | Size (approx) |
|------|---------------|
| tokenized_data.pkl | 500MB - 2GB |
| train.pkl | 350MB - 1.4GB |
| val.pkl | 75MB - 300MB |
| test.pkl | 75MB - 300MB |
| best_model.pt | 500MB |
| checkpoint_epoch_X.pt | 500MB each |

---

## ⚙️ Configuration Options

### Tokenization:
```python
MAX_CODE_LENGTH = 256      # Max tokens per code sample
MAX_DFG_LENGTH = 64        # Max DFG nodes
```

### Training:
```python
batch_size = 16            # Batch size
learning_rate = 2e-5       # Learning rate
num_epochs = 10            # Number of epochs
max_grad_norm = 1.0        # Gradient clipping
weight_decay = 0.01        # Weight decay
warmup_steps = 100         # LR warmup steps
```

### Model:
```python
model_type = 'simple'      # 'simple' or 'gnn'
use_dfg = True             # Use DFG attention
num_labels = 2             # Binary classification
dropout = 0.1              # Dropout rate
```

---

## 🔍 Key Improvements vs Script.py

| Feature | Script.py | GraphCodeBERT (New) |
|---------|-----------|---------------------|
| Tokenization | Mean pooling embeddings | Token-level with positions |
| Graph Info | Edge list only | DFG matrix + edge types |
| Model | Generic | GraphCodeBERT-specific |
| Training | Manual loop | Full Trainer class |
| Monitoring | Basic prints | TensorBoard integration |
| Checkpointing | Single save | Epoch + best model |
| Metrics | Basic | Comprehensive + confusion matrix |

---

## 💡 Tips & Tricks

### Performance:
- Use GPU: `CONFIG['device'] = 'cuda'`
- Increase batch size nếu có memory: `CONFIG['batch_size'] = 32`
- Use mixed precision: `torch.cuda.amp.autocast()`

### Memory:
- Giảm `max_code_length` nếu OOM
- Giảm `batch_size`
- Use gradient checkpointing

### Accuracy:
- Train longer: `num_epochs = 20`
- Try different learning rates: `2e-5, 5e-5, 1e-5`
- Enable DFG: `use_dfg = True`
- Try GNN model: `model_type = 'gnn'`

---

## 📚 Documentation Files

1. **README_GraphCodeBERT.md**: Comprehensive documentation
   - Architecture explanation
   - Detailed usage
   - Configuration options
   - Troubleshooting

2. **QUICKSTART.md**: Quick start guide
   - Installation steps
   - Basic usage
   - Common problems
   - Best practices

3. **COMPARISON.md**: Comparison with old script
   - Feature comparison
   - Performance comparison
   - Migration guide
   - When to use which

4. **SUMMARY.md**: This file
   - Project overview
   - File descriptions
   - Workflow
   - Quick reference

---

## 🚀 Next Steps

After completing basic training:

1. **Hyperparameter Tuning**:
   - Grid search learning rates
   - Try different batch sizes
   - Experiment with model architectures

2. **Advanced Features**:
   - Multi-task learning
   - Attention visualization
   - Model interpretation
   - Ensemble methods

3. **Deployment**:
   - Export to ONNX
   - Create inference API
   - Web interface
   - CI/CD integration

4. **Research**:
   - Cross-project evaluation
   - Few-shot learning
   - Transfer learning
   - Zero-shot detection

---

## 🤝 Contributing

To extend this project:

1. Add new vulnerability types trong tokenizer
2. Implement new model architectures
3. Add data augmentation techniques
4. Improve graph construction
5. Add visualization tools

---

## 📞 Support Resources

- **Documentation**: README_GraphCodeBERT.md
- **Quick Start**: QUICKSTART.md
- **Comparison**: COMPARISON.md
- **Demo**: Run `python demo_pipeline.py`
- **Test**: Individual component test scripts

---

## ✅ Checklist for First Run

- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Data in correct folders: `output/` and `output_safe/`
- [ ] Run tokenizer: `python graphcodebert_tokenizer.py`
- [ ] Check processed data: `processed_graphcodebert/` exists
- [ ] Start training: `python train_graphcodebert.py`
- [ ] Monitor with TensorBoard: `tensorboard --logdir logs/`
- [ ] Check results: `models/graphcodebert_vuln_detector/test_results.json`

---

**Created**: November 2024  
**Purpose**: Improved tokenization for GraphCodeBERT vulnerability detection  
**Status**: Production-ready ✅

---

Happy coding! 🚀
