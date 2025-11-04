"""
Demo Multi-label Prediction
Test model trên code examples
"""

import torch
from graphcodebert_multilabel_model import MultiLabelGraphCodeBERT, predict_vulnerabilities
from transformers import RobertaTokenizer


# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")


def tokenize_code(code, max_length=256):
    """Tokenize code snippet"""
    tokens = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask']


def demo_prediction():
    """Demo predictions trên các code examples"""
    
    print("="*70)
    print("MULTI-LABEL VULNERABILITY DETECTION - DEMO")
    print("="*70)
    
    # Load model
    print("\n[+] Loading model...")
    model = MultiLabelGraphCodeBERT(num_labels=4, use_dfg=True)
    model.eval()
    
    # Example codes
    examples = [
        {
            'name': 'SQL Injection',
            'code': '''
import java.sql.*;
public class VulnerableCode {
    public void login(String username, String password) {
        String query = "SELECT * FROM users WHERE username='" + username + "'";
        executeQuery(query);
    }
}
            '''
        },
        {
            'name': 'Path Traversal',
            'code': '''
import java.io.*;
public class FileHandler {
    public void readFile(String filename) {
        File file = new File("/data/" + filename);
        FileInputStream fis = new FileInputStream(file);
    }
}
            '''
        },
        {
            'name': 'Command Injection',
            'code': '''
public class SystemCommand {
    public void execute(String cmd) {
        Runtime.getRuntime().exec("bash -c " + cmd);
    }
}
            '''
        },
        {
            'name': 'Multiple Vulnerabilities (SQL + Path)',
            'code': '''
import java.sql.*;
import java.io.*;
public class MultipleVulns {
    public void process(String userId, String filename) {
        // SQL Injection
        String query = "SELECT * FROM users WHERE id=" + userId;
        executeQuery(query);
        
        // Path Traversal
        File file = new File(filename);
        readFile(file);
    }
}
            '''
        },
        {
            'name': 'Safe Code',
            'code': '''
import java.sql.*;
public class SecureCode {
    public void login(String username, String password) {
        String query = "SELECT * FROM users WHERE username=? AND password=?";
        PreparedStatement stmt = connection.prepareStatement(query);
        stmt.setString(1, username);
        stmt.setString(2, password);
        stmt.executeQuery();
    }
}
            '''
        }
    ]
    
    # Predict on each example
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}] {example['name']}")
        print("-" * 70)
        
        # Tokenize
        input_ids, attention_mask = tokenize_code(example['code'])
        
        # Dummy DFG (in real scenario, extract from Joern)
        dfg_matrix = torch.rand(1, 64, 64)
        
        # Predict
        result = predict_vulnerabilities(
            model,
            input_ids,
            attention_mask,
            dfg_matrix,
            threshold=0.5
        )
        
        # Display results
        print(f"Code snippet:")
        code_lines = example['code'].strip().split('\n')[:5]
        for line in code_lines:
            print(f"  {line}")
        if len(example['code'].strip().split('\n')) > 5:
            print(f"  ...")
        
        print(f"\n✅ Predictions:")
        print(f"  Binary:       {result['predictions']}")
        print(f"  Probabilities: {result['probabilities'].round(3)}")
        
        if len(result['detected_vulnerabilities']) > 0:
            print(f"\n⚠️  Detected {len(result['detected_vulnerabilities'])} vulnerability(ies):")
            for vuln in result['detected_vulnerabilities']:
                confidence_pct = vuln['confidence'] * 100
                print(f"    • {vuln['type']:20s} (confidence: {confidence_pct:.1f}%)")
        else:
            print(f"\n✅ No vulnerabilities detected (code appears safe)")
        
        print()
    
    print("="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print()
    print("💡 Note: Probabilities are random in this demo.")
    print("   Train the model first with:")
    print("   python train_multilabel_graphcodebert.py")
    print()


def explain_multi_label():
    """Giải thích multi-label approach"""
    print("\n" + "="*70)
    print("MULTI-LABEL CLASSIFICATION EXPLAINED")
    print("="*70)
    
    print("""
Multi-label classification cho phép phát hiện NHIỀU loại lỗi đồng thời:

┌─────────────────────────────────────────────────────────────┐
│  Code Sample                                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ String query = "SELECT * FROM users WHERE id=" + id;  │  │
│  │ File file = new File(filename);                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│                   GraphCodeBERT Model                        │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Output: [1, 0, 1, 0]                                  │  │
│  │          ↑  ↑  ↑  ↑                                   │  │
│  │          │  │  │  └─ Buffer Overflow = 0 (No)         │  │
│  │          │  │  └─── Command Injection = 1 (Yes)       │  │
│  │          │  └───── Path Traversal = 0 (No)            │  │
│  │          └─────── SQL Injection = 1 (Yes)             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Các metrics quan trọng:

1. Exact Match Accuracy: Tất cả labels phải đúng
   Example: [1,0,1,0] vs [1,0,1,0] → Exact match ✅
            [1,0,1,0] vs [1,0,0,0] → Not exact match ❌

2. Hamming Loss: Số lượng labels sai / tổng số labels
   Lower is better (0 = perfect)

3. F1 Macro: Average F1 score của mỗi vulnerability type
   Treats each vulnerability equally

4. F1 Micro: F1 tính trên tất cả predictions
   Affected more by frequent vulnerabilities

So với Ensemble approach:
• Multi-label: 1 model → 1x time, 1x memory ✅
• Ensemble: 4 models → 4x time, 4x memory ❌
• Multi-label learns correlations between vulnerabilities ✅
• Multi-label gives consistent predictions ✅
    """)


def usage_guide():
    """Hướng dẫn sử dụng"""
    print("\n" + "="*70)
    print("USAGE GUIDE")
    print("="*70)
    
    print("""
📚 Workflow:

1. Tokenize data:
   python graphcodebert_tokenizer.py

2. Train multi-label model:
   python train_multilabel_graphcodebert.py

3. Load và predict:
   ```python
   from graphcodebert_multilabel_model import MultiLabelGraphCodeBERT
   import torch
   
   # Load trained model
   model = MultiLabelGraphCodeBERT(num_labels=4)
   checkpoint = torch.load('models/multilabel_graphcodebert/best_model.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   
   # Predict
   with torch.no_grad():
       logits, probs = model(input_ids, attention_mask, dfg_matrix)
   
   # Get predictions
   predictions = (probs >= 0.5).int()
   
   # Interpret
   vuln_names = ['SQL_Injection', 'Command_Injection', 
                 'Path_Traversal', 'Buffer_Overflow']
   
   for i, name in enumerate(vuln_names):
       if predictions[0, i] == 1:
           print(f"⚠️  {name}: {probs[0, i]:.2%}")
   ```

🎯 Threshold tuning:

Default threshold = 0.5, nhưng có thể adjust:

• High precision (ít false positive): threshold = 0.7
• High recall (phát hiện nhiều): threshold = 0.3
• Balanced: threshold = 0.5

Per-class threshold:
   thresholds = {
       'SQL_Injection': 0.6,
       'Command_Injection': 0.5,
       'Path_Traversal': 0.4,
       'Buffer_Overflow': 0.7
   }

📊 Evaluate results:

   from sklearn.metrics import classification_report
   
   print(classification_report(
       y_true, 
       y_pred,
       target_names=vuln_names
   ))

🚀 Production deployment:

1. Export to ONNX for faster inference
2. Use batch prediction cho multiple samples
3. Cache embeddings cho repeated code
4. Monitor predictions và retrain periodically
    """)


if __name__ == "__main__":
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     MULTI-LABEL GRAPHCODEBERT VULNERABILITY DETECTION DEMO        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    # Demo predictions
    demo_prediction()
    
    # Explain multi-label
    explain_multi_label()
    
    # Usage guide
    usage_guide()
    
    print("\n" + "="*70)
    print("For more information, see: MULTILABEL_COMPARISON.md")
    print("="*70 + "\n")
