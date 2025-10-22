"""
Gaming Console Sentiment Analysis - DistilBERT Model Training
Clean implementation focused on DistilBERT only
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import json
import os
from tqdm import tqdm
import pandas as pd

class GamingConsoleSentimentDataset(Dataset):
    """Custom dataset for gaming console sentiment analysis with DistilBERT."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def setup_distilbert_model(num_labels: int = 3):
    """Initialize DistilBERT model and tokenizer for gaming console sentiment analysis."""
    print("Setting up DistilBERT model for gaming console sentiment analysis...")
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    
    model.to(device)
    
    return model, tokenizer, device

def train_distilbert_gaming_console(
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    output_dir: str = "./results/models/distilbert_gaming_console",
    max_length: int = 256,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5
) -> Dict[str, Any]:
    """
    Train DistilBERT model specifically for gaming console sentiment analysis.
    """
    print("="*60)
    print("TRAINING DISTILBERT FOR GAMING CONSOLE SENTIMENT ANALYSIS")
    print("="*60)
    
    # Setup model
    model, tokenizer, device = setup_distilbert_model(num_labels=3)
    
    # Create datasets
    train_dataset = GamingConsoleSentimentDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = GamingConsoleSentimentDataset(X_val, y_val, tokenizer, max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=learning_rate,
        save_total_limit=2,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print(f"Training on {len(X_train):,} gaming console reviews...")
    print(f"Validation on {len(X_val):,} gaming console reviews...")
    
    training_result = trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions, val_probabilities = predict_with_distilbert(
        model, tokenizer, X_val, device, max_length, batch_size
    )
    
    # Calculate validation metrics
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_report = classification_report(y_val, val_predictions, output_dict=True, zero_division=0)
    val_cm = confusion_matrix(y_val, val_predictions)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'training_result': training_result,
        'val_predictions': val_predictions,
        'val_probabilities': val_probabilities,
        'val_accuracy': val_accuracy,
        'val_report': val_report,
        'val_confusion_matrix': val_cm,
        'training_loss': training_result.training_loss,
        'training_steps': training_result.global_step
    }

def predict_with_distilbert(
    model,
    tokenizer,
    texts: List[str],
    device,
    max_length: int = 256,
    batch_size: int = 16
) -> Tuple[List[int], List[List[float]]]:
    """Make predictions using trained DistilBERT model."""
    
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities and predictions
            batch_probabilities = torch.softmax(logits, dim=-1)
            batch_predictions = torch.argmax(logits, dim=-1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            probabilities.extend(batch_probabilities.cpu().numpy())
    
    return predictions, probabilities

def evaluate_on_test_set(
    model,
    tokenizer,
    X_test: List[str],
    y_test: List[int],
    device,
    max_length: int = 256,
    batch_size: int = 16
) -> Dict[str, Any]:
    """Evaluate DistilBERT model on test set."""
    
    print("\nEvaluating DistilBERT on test set...")
    
    # Make predictions
    test_predictions, test_probabilities = predict_with_distilbert(
        model, tokenizer, X_test, device, max_length, batch_size
    )
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(y_test, test_predictions)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return {
        'test_predictions': test_predictions,
        'test_probabilities': test_probabilities,
        'test_accuracy': test_accuracy,
        'test_report': test_report,
        'test_confusion_matrix': test_cm
    }

def cross_validate_distilbert(
    X: List[str],
    y: List[int],
    k: int = 5,
    max_length: int = 256,
    batch_size: int = 16,
    num_epochs: int = 2
) -> Dict[str, Any]:
    """Perform k-fold cross-validation with DistilBERT."""
    
    print(f"\nPerforming {k}-fold cross-validation with DistilBERT...")
    
    # Convert to numpy for indexing
    X_array = np.array(X)
    y_array = np.array(y)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array)):
        print(f"\nTraining fold {fold + 1}/{k}...")
        
        # Split data for this fold
        X_train_fold = X_array[train_idx].tolist()
        X_val_fold = X_array[val_idx].tolist()
        y_train_fold = y_array[train_idx].tolist()
        y_val_fold = y_array[val_idx].tolist()
        
        # Setup model for this fold
        model, tokenizer, device = setup_distilbert_model(num_labels=3)
        
        # Create datasets
        train_dataset = GamingConsoleSentimentDataset(X_train_fold, y_train_fold, tokenizer, max_length)
        val_dataset = GamingConsoleSentimentDataset(X_val_fold, y_val_fold, tokenizer, max_length)
        
        # Training arguments for CV (reduced epochs)
        training_args = TrainingArguments(
            output_dir=f'./temp_cv_fold_{fold}',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            learning_rate=2e-5,
            report_to=None,
        )
        
        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        # Evaluate on validation fold
        val_predictions, _ = predict_with_distilbert(
            model, tokenizer, X_val_fold, device, max_length, batch_size
        )
        
        # Calculate metrics
        fold_accuracy = accuracy_score(y_val_fold, val_predictions)
        fold_report = classification_report(y_val_fold, val_predictions, output_dict=True, zero_division=0)
        fold_f1 = fold_report['macro avg']['f1-score']
        
        fold_accuracies.append(fold_accuracy)
        fold_f1_scores.append(fold_f1)
        
        print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}")
        
        # Clean up
        del model, trainer, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'accuracy_mean': float(np.mean(fold_accuracies)),
        'accuracy_std': float(np.std(fold_accuracies)),
        'f1_mean': float(np.mean(fold_f1_scores)),
        'f1_std': float(np.std(fold_f1_scores)),
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores
    }
