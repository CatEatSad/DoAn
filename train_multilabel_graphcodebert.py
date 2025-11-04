"""
Training Script cho Multi-label GraphCodeBERT
Phát hiện nhiều loại lỗi đồng thời
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    multilabel_confusion_matrix, hamming_loss,
    jaccard_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from graphcodebert_multilabel_model import MultiLabelGraphCodeBERT, SharedMultiLabelGraphCodeBERT
from graphcodebert_dataset import load_datasets


# Configuration
CONFIG = {
    'model_type': 'multi_label',  # 'multi_label' or 'shared_multi_label'
    'model_name': 'microsoft/graphcodebert-base',
    'num_labels': 4,  # SQL, Command, Path, Buffer
    'use_dfg': True,
    
    # Training
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 15,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Multi-label specific
    'pos_weight': [2.0, 2.0, 2.0, 2.0],  # Weight cho positive class (nếu imbalanced)
    'threshold': 0.5,  # Decision threshold
    
    # Paths
    'processed_dir': 'processed_graphcodebert',
    'output_dir': 'models/multilabel_graphcodebert',
    'log_dir': 'logs/multilabel_graphcodebert',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0
}

# Vulnerability names
VULN_NAMES = ['SQL_Injection', 'Command_Injection', 'Path_Traversal', 'Buffer_Overflow']


class MultiLabelTrainer:
    """Trainer cho multi-label classification"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        print(f"\n{'='*70}")
        print(f"MULTI-LABEL GRAPHCODEBERT TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Labels: {', '.join(VULN_NAMES)}")
        
        # Create directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Load datasets
        print(f"\n[+] Loading datasets...")
        self.dataloaders = load_datasets(
            config['processed_dir'],
            batch_size=config['batch_size']
        )
        
        # Create model
        print(f"\n[+] Creating multi-label model...")
        if config['model_type'] == 'multi_label':
            self.model = MultiLabelGraphCodeBERT(
                model_name=config['model_name'],
                num_labels=config['num_labels'],
                use_dfg=config['use_dfg']
            )
        else:
            self.model = SharedMultiLabelGraphCodeBERT(
                model_name=config['model_name'],
                num_labels=config['num_labels'],
                use_dfg=config['use_dfg']
            )
        
        self.model.to(self.device)
        
        # Loss function: BCEWithLogitsLoss for multi-label
        pos_weight = torch.tensor(config['pos_weight']).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        num_training_steps = len(self.dataloaders['train']) * config['num_epochs']
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Tracking
        self.global_step = 0
        self.best_val_f1 = 0.0
        
        print(f"\n[+] Trainer initialized")
        print(f"    Training samples: {len(self.dataloaders['train'].dataset)}")
        if 'val' in self.dataloaders:
            print(f"    Validation samples: {len(self.dataloaders['val'].dataset)}")
    
    def convert_labels_to_multilabel(self, vuln_type, num_labels=4):
        """
        Convert single-label to multi-label format
        
        Args:
            vuln_type: [batch_size] with values 0,1,2,3 or -1 for safe
        
        Returns:
            multi_labels: [batch_size, num_labels] binary matrix
        """
        batch_size = vuln_type.size(0)
        multi_labels = torch.zeros(batch_size, num_labels, device=vuln_type.device)
        
        for i in range(batch_size):
            vtype = vuln_type[i].item()
            if vtype >= 0 and vtype < num_labels:
                multi_labels[i, vtype] = 1.0
        
        return multi_labels
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            vuln_type = batch['vuln_type'].to(self.device)
            dfg_matrix = batch['dfg_matrix'].to(self.device) if self.config['use_dfg'] else None
            
            # Convert to multi-label format
            labels = self.convert_labels_to_multilabel(vuln_type, self.config['num_labels'])
            
            # Forward pass
            logits, probs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dfg_matrix=dfg_matrix
            )
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = (probs >= self.config['threshold']).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            self.global_step += 1
        
        # Compute metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.compute_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.dataloaders['train'])
        
        return metrics
    
    def evaluate(self, split='val'):
        """Evaluate on val/test set"""
        if split not in self.dataloaders:
            return None
        
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc=f"Evaluating {split}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                vuln_type = batch['vuln_type'].to(self.device)
                dfg_matrix = batch['dfg_matrix'].to(self.device) if self.config['use_dfg'] else None
                
                # Convert to multi-label
                labels = self.convert_labels_to_multilabel(vuln_type, self.config['num_labels'])
                
                # Forward
                logits, probs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dfg_matrix=dfg_matrix
                )
                
                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Predictions
                preds = (probs >= self.config['threshold']).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)
        
        metrics = self.compute_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.dataloaders[split])
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i, name in enumerate(VULN_NAMES):
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
            )
            metrics['per_class'][name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    
    def compute_multilabel_metrics(self, y_true, y_pred):
        """Compute multi-label metrics"""
        
        # Exact match accuracy (all labels must match)
        exact_match = accuracy_score(y_true, y_pred)
        
        # Hamming loss (fraction of wrong labels)
        hamming = hamming_loss(y_true, y_pred)
        
        # Jaccard score (IoU for multi-label)
        jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        
        # Micro/Macro averaged metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        return {
            'exact_match': exact_match,
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = None
            if 'val' in self.dataloaders:
                val_metrics = self.evaluate('val')
            
            # Log
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1(macro): {train_metrics['f1_macro']:.4f}, "
                  f"Exact Match: {train_metrics['exact_match']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"F1(macro): {val_metrics['f1_macro']:.4f}, "
                      f"Exact Match: {val_metrics['exact_match']:.4f}")
                
                # Per-class F1
                print(f"  Per-class F1:")
                for name, scores in val_metrics['per_class'].items():
                    print(f"    {name:20s}: {scores['f1']:.4f}")
                
                # TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/f1_macro', val_metrics['f1_macro'], epoch)
                self.writer.add_scalar('val/exact_match', val_metrics['exact_match'], epoch)
                
                # Save best model
                if val_metrics['f1_macro'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1_macro']
                    
                    save_path = os.path.join(self.config['output_dir'], 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'config': self.config
                    }, save_path)
                    
                    print(f"  ✅ Saved best model (F1: {val_metrics['f1_macro']:.4f})")
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Validation F1 (macro): {self.best_val_f1:.4f}")
        
        # Test evaluation
        if 'test' in self.dataloaders:
            print(f"\n[+] Evaluating on test set...")
            
            # Load best model
            best_model_path = os.path.join(self.config['output_dir'], 'best_model.pt')
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            test_metrics = self.evaluate('test')
            
            print(f"\nTest Results:")
            print(f"  Exact Match:   {test_metrics['exact_match']:.4f}")
            print(f"  Hamming Loss:  {test_metrics['hamming_loss']:.4f}")
            print(f"  Jaccard Score: {test_metrics['jaccard_score']:.4f}")
            print(f"  F1 (micro):    {test_metrics['f1_micro']:.4f}")
            print(f"  F1 (macro):    {test_metrics['f1_macro']:.4f}")
            
            print(f"\n  Per-class Results:")
            for name, scores in test_metrics['per_class'].items():
                print(f"    {name:20s}: P={scores['precision']:.4f}, "
                      f"R={scores['recall']:.4f}, F1={scores['f1']:.4f}")
            
            # Save results
            results_path = os.path.join(self.config['output_dir'], 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            print(f"\n✅ Saved test results to {results_path}")
        
        self.writer.close()


def main():
    """Main entry point"""
    trainer = MultiLabelTrainer(CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()
