import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import json
import os
from datetime import datetime

class BGMHANPipeline:
    def __init__(self, data_processor, model, test_size=0.2, lr=2e-5, ep=100):
        """
        Initialize the BGMHAN Pipeline.
        
        Args:
            data_processor: BGMHANDataProcessor instance
            model: BGMHAN model instance
        """
        self.data_processor = data_processor
        self.model = model
        self.test_size = test_size
        self.learning_rate = lr
        self.epochs=ep
        
    def _prepare_embeddings(self, X_embeddings, indices):
        """
        Prepare embeddings for model input by stacking and reshaping.
        
        Args:
            X_embeddings: Dictionary of embeddings for each text field
            indices: Indices to select from embeddings
            
        Returns:
            torch.Tensor: Prepared embeddings tensor
        """
        # Stack embeddings for each field maintaining the proper shape for BGMHAN
        stacked_embeddings = []
        for col in self.data_processor.text_columns:
            # Reshape embeddings to (batch_size, 1, 1, embedding_dim)
            field_embeddings = X_embeddings[col][indices]
            reshaped = field_embeddings.reshape(len(indices), 1, 1, -1)
            stacked_embeddings.append(reshaped)
            
        # Stack along field dimension
        return torch.FloatTensor(np.stack(stacked_embeddings, axis=1))
        
    def run(self):
        """
        Execute the complete training and evaluation pipeline.
        
        Args:
            test_size: Proportion of data to use for validation/testing
            
        Returns:
            dict: Dictionary containing all training and evaluation metrics
        """
        # Load and prepare data
        X_embeddings, y, classes = self.data_processor.prepare_data()
        
        # Split indices with stratification for train/val/test
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=self.test_size, random_state=42, stratify=y)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])
        
        # Prepare data splits
        X_train = self._prepare_embeddings(X_embeddings, train_idx)
        X_val = self._prepare_embeddings(X_embeddings, val_idx)
        X_test = self._prepare_embeddings(X_embeddings, test_idx)
        
        y_train = torch.LongTensor(y[train_idx])
        y_val = torch.LongTensor(y[val_idx])
        y_test = torch.LongTensor(y[test_idx])
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        
        # Setup device and move data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
        y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)
        class_weights = class_weights.to(device)
        self.model = self.model.to(device)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate schedulers
        cycle_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,
            epochs=200,
            steps_per_epoch=len(X_train) // 16 + 1,
            pct_start=0.2,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Training parameters
        batch_size = 16
        num_epochs = self.epochs
        patience = 10
        best_val_acc = 0
        best_state = None
        no_improve = 0
        min_lr_reached = False
        training_history = []

        print(f"Starting training on {device}")
        print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            train_correct = 0
            
            # Shuffle training data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Batch training
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                cycle_scheduler.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / len(y_train)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_x = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / len(y_val)
            
            # Update plateau scheduler and check learning rate
            plateau_scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= plateau_scheduler.min_lrs[0]:
                min_lr_reached = True
            
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': total_loss / (len(X_train) // batch_size),
                'train_acc': train_acc,
                'val_loss': val_loss / max(1, len(X_val) // batch_size),
                'val_acc': val_acc,
                'learning_rate': current_lr
            })
            # Print progress with current learning rate
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {training_history[-1]['train_loss']:.4f}, "
                    f"Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {training_history[-1]['val_loss']:.4f}, "
                    f"Val Acc: {val_acc:.4f}")
                print(f"Learning Rate: {current_lr:.2e}")
                print("-" * 60)
                
            # Model checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict()
                no_improve = 0
                print(f"Epoch {epoch+1} - Validation accuracy: {val_acc:.4f} (best)")
            else:
                no_improve += 1
            
            # Early stopping checks
            if no_improve >= patience and min_lr_reached and epoch > 100:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
            elif no_improve >= patience * 2:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
        
        
        # Final evaluation
        print("\nRunning final evaluation")
        self.model.load_state_dict(best_state)
        self.model.eval()
        
        test_results = self._evaluate(X_test, y_test, criterion, batch_size, classes)
        test_results.update({
            'training_history': training_history,
            'best_val_acc': best_val_acc
        })
        
        # Save results
        self._save_results(test_results)
        
        return test_results
    
    def _evaluate(self, X_test, y_test, criterion, batch_size, classes):
        """Run final evaluation and return metrics."""
        test_loss = 0
        test_correct = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_x = X_test[i:i+batch_size]
                batch_y = y_test[i:i+batch_size]
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        test_accuracy = test_correct/len(y_test)
        test_loss = test_loss/len(y_test)
        cm = confusion_matrix(all_labels, all_preds)
        test_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        test_ap = average_precision_score(all_labels, np.array(all_probs)[:, 1])
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                all_labels, all_preds, target_names=classes, digits=4, output_dict=True
            ),
            'predictions': {
                'true_labels': np.array(all_labels).tolist(),
                'predicted_labels': np.array(all_preds).tolist(),
                'probabilities': np.array(all_probs).tolist()
            }
        }
    
    def _save_results(self, results):
        """Save results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = '../results/bgmhan'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/bgmhan_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
    