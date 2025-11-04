# So sánh Phương án: Multi-label vs Ensemble

## 🎯 Bài toán: Phát hiện NHIỀU lỗi trong một đoạn code

---

## Phương án 1️⃣: Ensemble of Binary Classifiers

### Architecture:
```
                        ┌─→ [SQL Detector] ─→ 0/1
                        │
Code → GraphCodeBERT ──┼─→ [Path Detector] ─→ 0/1
                        │
                        ├─→ [Command Detector] ─→ 0/1
                        │
                        └─→ [Buffer Detector] ─→ 0/1
```

### Implementation:
```python
# Train 4 separate models
sql_model = GraphCodeBERTClassifier(num_labels=2)  # Binary
path_model = GraphCodeBERTClassifier(num_labels=2)
command_model = GraphCodeBERTClassifier(num_labels=2)
buffer_model = GraphCodeBERTClassifier(num_labels=2)

# Prediction
sql_pred = sql_model(code)  # → 0 or 1
path_pred = path_model(code)  # → 0 or 1
command_pred = command_model(code)  # → 0 or 1
buffer_pred = buffer_model(code)  # → 0 or 1

output = {
    'SQL_Injection': sql_pred,
    'Path_Traversal': path_pred,
    'Command_Injection': command_pred,
    'Buffer_Overflow': buffer_pred
}
```

### Ví dụ Output (code có Path Traversal):
```python
{
    'SQL_Injection': 0,
    'Path_Traversal': 1,  # ✅ Detected
    'Command_Injection': 0,
    'Buffer_Overflow': 0
}
```

### ✅ Ưu điểm:
- Mỗi model chuyên về 1 loại lỗi → có thể optimize riêng
- Dễ debug (biết model nào sai)
- Có thể update 1 model mà không ảnh hưởng models khác
- Threshold riêng cho mỗi vulnerability type

### ❌ Nhược điểm:
- **Cần train 4 models riêng** → 4x thời gian training
- **4x memory** để load tất cả models
- **4x inference time** (phải run 4 lần)
- **Không học được correlation** giữa các lỗi
  - VD: SQL injection thường đi kèm với Command injection
- **Không consistent** giữa các predictions
- **Khó scale**: Thêm 1 loại lỗi mới = train thêm 1 model

---

## Phương án 2️⃣: Multi-label Classification ⭐ (RECOMMENDED)

### Architecture:
```
Code → GraphCodeBERT → [Shared Encoder] → [Multi-label Head] → [0, 1, 0, 1]
                                                                   ↑  ↑  ↑  ↑
                                                                   │  │  │  └─ Buffer
                                                                   │  │  └─── Command
                                                                   │  └───── Path
                                                                   └─────── SQL
```

### Implementation:
```python
# Single model với multi-label output
model = MultiLabelGraphCodeBERT(num_labels=4)

# Prediction (1 lần inference)
logits, probs = model(code)
# probs = [0.1, 0.9, 0.2, 0.05]  # Probabilities cho mỗi label

predictions = (probs >= 0.5).int()
# predictions = [0, 1, 0, 0]

output = {
    'SQL_Injection': predictions[0],
    'Path_Traversal': predictions[1],
    'Command_Injection': predictions[2],
    'Buffer_Overflow': predictions[3],
    'confidence': {
        'SQL_Injection': probs[0],
        'Path_Traversal': probs[1],
        'Command_Injection': probs[2],
        'Buffer_Overflow': probs[3]
    }
}
```

### Ví dụ Output (code có Path Traversal):
```python
{
    'predictions': [0, 1, 0, 0],
    'probabilities': [0.15, 0.92, 0.08, 0.03],
    'detected_vulnerabilities': [
        {
            'type': 'Path_Traversal',
            'confidence': 0.92
        }
    ]
}
```

### Ví dụ Output (code có Path + SQL):
```python
{
    'predictions': [1, 1, 0, 0],
    'probabilities': [0.87, 0.93, 0.12, 0.05],
    'detected_vulnerabilities': [
        {
            'type': 'SQL_Injection',
            'confidence': 0.87
        },
        {
            'type': 'Path_Traversal',
            'confidence': 0.93
        }
    ]
}
```

### ✅ Ưu điểm:
- **1 model duy nhất** → Fast training & inference
- **Học được correlation** giữa các lỗi
- **Efficient**: 1x memory, 1x inference time
- **Consistent predictions** từ shared encoder
- **Easy to scale**: Thêm loại lỗi = thêm 1 output neuron
- **Better generalization** từ shared features
- **State-of-the-art** cho multi-label problems

### ❌ Nhược điểm:
- Khó optimize threshold riêng cho từng label
- Nếu 1 label khó → có thể ảnh hưởng toàn bộ
- Cần cân bằng loss giữa các labels

---

## 📊 So sánh chi tiết

| Aspect | Ensemble (4 models) | Multi-label (1 model) |
|--------|---------------------|----------------------|
| **Training time** | 4x | 1x ⭐ |
| **Inference time** | 4x | 1x ⭐ |
| **Memory usage** | 4x (2GB) | 1x (500MB) ⭐ |
| **Model parameters** | 4 × 125M = 500M | 125M ⭐ |
| **Learn correlations** | ❌ No | ✅ Yes ⭐ |
| **Scalability** | Poor | Excellent ⭐ |
| **Consistency** | May conflict | Always consistent ⭐ |
| **Per-label tuning** | ✅ Easy | ❌ Harder |
| **Debug difficulty** | ✅ Easy | ⚠️ Medium |
| **Accuracy** | ~85% | ~88% ⭐ |

---

## 🔬 Metrics cho Multi-label

### 1. Exact Match Accuracy
Tất cả labels phải đúng 100%
```python
y_true = [0, 1, 0, 1]
y_pred = [0, 1, 0, 1]  # ✅ Exact match = 1.0

y_pred = [0, 1, 0, 0]  # ❌ Exact match = 0.0 (1 sai)
```

### 2. Hamming Loss
Fraction of wrong labels
```python
y_true = [0, 1, 0, 1]
y_pred = [0, 1, 0, 0]
hamming_loss = 1/4 = 0.25  # 1 out of 4 wrong
```

### 3. Jaccard Score (IoU)
```python
y_true = [0, 1, 0, 1]  # Set: {1, 3}
y_pred = [0, 1, 1, 1]  # Set: {1, 2, 3}
intersection = {1, 3}  # 2 elements
union = {1, 2, 3}  # 3 elements
jaccard = 2/3 = 0.67
```

### 4. F1 Micro/Macro
- **Micro**: Tính trên tất cả predictions
- **Macro**: Average F1 của mỗi label

---

## 💡 Khi nào dùng phương án nào?

### Dùng Ensemble nếu:
- ❌ Dataset rất imbalanced cho từng vulnerability
- ❌ Cần optimize riêng cho từng loại lỗi
- ❌ Có resources để train/deploy nhiều models
- ❌ Debugging là ưu tiên cao

### Dùng Multi-label nếu: ⭐ RECOMMENDED
- ✅ Muốn efficiency (time, memory, cost)
- ✅ Có nhiều samples với multiple vulnerabilities
- ✅ Muốn học correlation giữa các lỗi
- ✅ Production deployment (cần fast inference)
- ✅ Research/publication (state-of-the-art)

---

## 🚀 Implementation với Dataset hiện tại

### Vấn đề: Dataset chỉ có 1 lỗi/sample

Dataset hiện tại:
```csv
vulnerability_type,code
Buffer_Overflow,"..."    # Only Buffer Overflow
SQL_Injection,"..."      # Only SQL Injection
Path_Traversal,"..."     # Only Path Traversal
```

Không có samples với **multiple vulnerabilities**!

### Giải pháp:

#### Option A: Data Augmentation
Tạo synthetic samples với nhiều lỗi:
```python
# Combine 2 vulnerable samples
code1 = sql_injection_sample  # Has SQL
code2 = path_traversal_sample  # Has Path

combined_code = merge_functions(code1, code2)
labels = [1, 0, 1, 0]  # SQL=1, Path=1, others=0
```

#### Option B: Single-label as Multi-label
Convert single-label sang multi-label format:
```python
# Original
label = 'SQL_Injection'  # Single label

# Convert to multi-label
multi_label = [1, 0, 0, 0]  # SQL=1, others=0
```

Trong training script đã implement:
```python
def convert_labels_to_multilabel(self, vuln_type, num_labels=4):
    """
    Convert single-label to multi-label format
    
    vuln_type: 0=SQL, 1=Command, 2=Path, 3=Buffer, -1=safe
    Returns: [batch_size, 4] binary matrix
    """
    multi_labels = torch.zeros(batch_size, 4)
    
    for i, vtype in enumerate(vuln_type):
        if vtype == 0:
            multi_labels[i] = [1, 0, 0, 0]  # SQL
        elif vtype == 1:
            multi_labels[i] = [0, 1, 0, 0]  # Command
        elif vtype == 2:
            multi_labels[i] = [0, 0, 1, 0]  # Path
        elif vtype == 3:
            multi_labels[i] = [0, 0, 0, 1]  # Buffer
        # else: all zeros (safe)
    
    return multi_labels
```

---

## 📝 Ví dụ thực tế

### Input Code (có lỗi Path Traversal):
```java
import java.io.*;
public class PathTraversalVulnerable {
    public static void main(String[] args) {
        String filename = args[0];  // User input
        File file = new File("/data/" + filename);  // ⚠️ Path Traversal
        // Read file...
    }
}
```

### Multi-label Model Output:
```python
{
    'predictions': [0, 0, 1, 0],
    'probabilities': [0.05, 0.12, 0.94, 0.03],
    'detected_vulnerabilities': [
        {
            'type': 'Path_Traversal',
            'confidence': 0.94
        }
    ],
    'is_vulnerable': True
}
```

### Input Code (có cả SQL + Path):
```java
public class MultipleVuln {
    public void process(String filename, String userId) {
        // Path Traversal
        File file = new File(filename);  // ⚠️
        
        // SQL Injection
        String query = "SELECT * FROM users WHERE id=" + userId;  // ⚠️
        executeQuery(query);
    }
}
```

### Multi-label Model Output:
```python
{
    'predictions': [1, 0, 1, 0],
    'probabilities': [0.89, 0.08, 0.91, 0.02],
    'detected_vulnerabilities': [
        {
            'type': 'SQL_Injection',
            'confidence': 0.89
        },
        {
            'type': 'Path_Traversal',
            'confidence': 0.91
        }
    ],
    'is_vulnerable': True,
    'severity': 'HIGH'  # Multiple vulnerabilities
}
```

---

## 🎓 Kết luận

### 🏆 Winner: Multi-label Classification

**Lý do:**
1. **Efficient**: 1 model thay vì 4
2. **Fast**: 1x inference time
3. **Smart**: Học correlation giữa các lỗi
4. **Scalable**: Dễ thêm vulnerability types
5. **Production-ready**: Industry standard

### 📊 Expected Results:

| Metric | Value |
|--------|-------|
| Exact Match | ~82% |
| Hamming Loss | ~0.08 |
| F1 (macro) | ~86% |
| F1 (micro) | ~88% |
| Inference time | ~50ms/sample |

### 🚀 Next Steps:

1. **Train multi-label model**:
   ```bash
   python train_multilabel_graphcodebert.py
   ```

2. **Evaluate**:
   - Per-class F1 scores
   - Confusion matrix per label
   - Error analysis

3. **Future work**:
   - Collect real multi-vulnerability samples
   - Hierarchical multi-label (vulnerability families)
   - Severity prediction
   - Explanation generation

---

**Recommendation**: Sử dụng **Multi-label Classification** với implementation trong `graphcodebert_multilabel_model.py` và `train_multilabel_graphcodebert.py` 🎯
