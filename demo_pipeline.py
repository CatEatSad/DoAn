"""
Demo script - Test toàn bộ pipeline từ tokenization đến training
"""

import os
import sys

def check_requirements():
    """Kiểm tra các dependencies"""
    print("🔍 Checking requirements...")
    
    required = ['torch', 'transformers', 'numpy', 'sklearn', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_data_structure():
    """Kiểm tra cấu trúc thư mục data"""
    print("\n🔍 Checking data structure...")
    
    required_dirs = [
        'output/Buffer_Overflow',
        'output/Command_Injection',
        'output/Path_Traversal',
        'output/SQL_Injection',
        'output_safe/Buffer_Overflow',
        'output_safe/Command_Injection',
        'output_safe/Path_Traversal',
        'output_safe/SQL_Injection'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
            print(f"  ✅ {dir_path} ({len(json_files)} files)")
        else:
            print(f"  ❌ {dir_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def demo_tokenization():
    """Demo tokenization"""
    print("\n" + "="*70)
    print("STEP 1: TOKENIZATION")
    print("="*70)
    
    try:
        from graphcodebert_tokenizer import process_all_data
        
        print("\n[+] Starting tokenization...")
        all_data, stats = process_all_data()
        
        if len(all_data) > 0:
            print(f"\n✅ Tokenization successful!")
            print(f"   Processed {len(all_data)} samples")
            return True
        else:
            print("\n❌ No data processed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during tokenization: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_dataset():
    """Demo dataset loading"""
    print("\n" + "="*70)
    print("STEP 2: DATASET LOADING")
    print("="*70)
    
    try:
        from graphcodebert_dataset import load_datasets
        
        print("\n[+] Loading datasets...")
        dataloaders = load_datasets('processed_graphcodebert', batch_size=8)
        
        if 'train' in dataloaders:
            train_loader = dataloaders['train']
            print(f"\n✅ Dataset loading successful!")
            print(f"   Train batches: {len(train_loader)}")
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Input IDs shape: {batch['input_ids'].shape}")
            print(f"   Labels: {batch['label'].tolist()}")
            
            return True
        else:
            print("\n❌ Training data not found")
            return False
            
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_model():
    """Demo model creation"""
    print("\n" + "="*70)
    print("STEP 3: MODEL CREATION")
    print("="*70)
    
    try:
        import torch
        from graphcodebert_model import create_model
        
        print("\n[+] Creating model...")
        model = create_model(
            model_type='simple',
            num_labels=2,
            use_dfg=True
        )
        
        # Test forward pass
        batch_size = 4
        seq_len = 256
        
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        dfg_matrix = torch.rand(batch_size, 64, 64)
        
        print("\n[+] Testing forward pass...")
        logits = model(input_ids, attention_mask, dfg_matrix=dfg_matrix)
        
        print(f"\n✅ Model creation successful!")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected: [{batch_size}, 2]")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_training():
    """Demo training (1 epoch only)"""
    print("\n" + "="*70)
    print("STEP 4: TRAINING (DEMO - 1 EPOCH)")
    print("="*70)
    
    try:
        print("\n[+] Starting demo training...")
        print("    (For full training, run: python train_graphcodebert.py)")
        
        # Import và setup
        import torch
        from train_graphcodebert import Trainer, CONFIG
        
        # Modify config for quick demo
        demo_config = CONFIG.copy()
        demo_config['num_epochs'] = 1
        demo_config['batch_size'] = 8
        
        print("\n[+] Creating trainer...")
        trainer = Trainer(demo_config)
        
        print("\n[+] Training 1 epoch...")
        trainer.train()
        
        print(f"\n✅ Demo training successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo workflow"""
    print("="*70)
    print("GRAPHCODEBERT VULNERABILITY DETECTION - DEMO")
    print("="*70)
    print()
    print("This demo will:")
    print("  1. Check requirements")
    print("  2. Check data structure")
    print("  3. Tokenize data")
    print("  4. Load datasets")
    print("  5. Create model")
    print("  6. Run demo training (1 epoch)")
    print()
    
    # Step 0: Check requirements
    if not check_requirements():
        print("\n⚠️  Please install missing requirements first")
        print("   pip install -r requirements.txt")
        return
    
    # Step 1: Check data
    if not check_data_structure():
        print("\n⚠️  Data directories not found or incomplete")
        print("   Please ensure 'output/' and 'output_safe/' directories exist")
        return
    
    # Step 2: Tokenization
    print("\n" + "="*70)
    print("Would you like to run tokenization?")
    print("(This may take several minutes depending on data size)")
    response = input("Continue? [y/N]: ").strip().lower()
    
    if response == 'y':
        if not demo_tokenization():
            print("\n⚠️  Tokenization failed. Please check errors above.")
            return
    else:
        print("\n⏭️  Skipping tokenization")
        print("   Assuming processed_graphcodebert/ already exists")
    
    # Step 3: Dataset loading
    if not demo_dataset():
        print("\n⚠️  Dataset loading failed")
        print("   Please run tokenization first")
        return
    
    # Step 4: Model creation
    if not demo_model():
        print("\n⚠️  Model creation failed")
        return
    
    # Step 5: Training demo
    print("\n" + "="*70)
    print("Would you like to run demo training (1 epoch)?")
    print("(This will take a few minutes)")
    response = input("Continue? [y/N]: ").strip().lower()
    
    if response == 'y':
        demo_training()
    else:
        print("\n⏭️  Skipping demo training")
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETED!")
    print("="*70)
    print()
    print("📁 Generated files:")
    print("   - processed_graphcodebert/   (tokenized data)")
    print("   - models/                    (trained models)")
    print("   - logs/                      (TensorBoard logs)")
    print()
    print("🚀 Next steps:")
    print("   1. Full training: python train_graphcodebert.py")
    print("   2. TensorBoard: tensorboard --logdir logs/graphcodebert")
    print("   3. Customize config in train_graphcodebert.py")
    print()
    print("📚 See README_GraphCodeBERT.md for more details")
    print("="*70)


if __name__ == "__main__":
    main()
