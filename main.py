"""
Gaming Console Sentiment Analysis - Main Execution Script
Clean implementation with proper DistilBERT training and verification
"""

import os
import json
import numpy as np
from typing import Dict, Any


from data_preprocessing import load_and_process_gaming_console_data
from model_training import (
    train_distilbert_gaming_console,
    evaluate_on_test_set,
    cross_validate_distilbert
)
from visualization import create_all_visualizations

def save_verification_samples(verification_samples, output_path: str):
    """Save verification samples to text file for manual inspection."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("GAMING CONSOLE SENTIMENT ANALYSIS - DATA VERIFICATION\n")
        f.write("="*60 + "\n\n")
        f.write("Sample of filtered reviews to verify gaming console focus:\n\n")
        
        for i, sample in enumerate(verification_samples, 1):
            f.write(f"SAMPLE {i} (Rating: {sample['rating']}/5)\n")
            f.write("-" * 40 + "\n")
            f.write(f"ASIN: {sample['asin']}\n")
            f.write(f"Summary: {sample['summary']}\n")
            f.write(f"Review: {sample['reviewText']}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Verification samples saved to: {output_path}")

def save_distilbert_results(
    train_result: Dict,
    val_result: Dict,
    test_result: Dict,
    cv_result: Dict,
    output_path: str
):
    """Save DistilBERT results in clean format."""
    

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = {
        "model_type": "DistilBERT",
        "task": "Gaming Console Sentiment Analysis",
        "num_classes": 3,
        "class_labels": {
            "0": "Negative",
            "1": "Neutral", 
            "2": "Positive"
        },
        "training_phase": {
            "training_loss": float(train_result['training_loss']),
            "training_steps": int(train_result['training_steps'])
        },
        "validation_phase": {
            "accuracy": float(val_result['val_accuracy']),
            "classification_report": convert_numpy_types(val_result['val_report']),
            "confusion_matrix": convert_numpy_types(val_result['val_confusion_matrix'])
        },
        "test_phase": {
            "accuracy": float(test_result['test_accuracy']),
            "classification_report": convert_numpy_types(test_result['test_report']),
            "confusion_matrix": convert_numpy_types(test_result['test_confusion_matrix'])
        },
        "cross_validation": {
            "folds": len(cv_result['fold_accuracies']),
            "accuracy_mean": float(cv_result['accuracy_mean']),
            "accuracy_std": float(cv_result['accuracy_std']),
            "f1_mean": float(cv_result['f1_mean']),
            "f1_std": float(cv_result['f1_std']),
            "fold_accuracies": convert_numpy_types(cv_result['fold_accuracies']),
            "fold_f1_scores": convert_numpy_types(cv_result['fold_f1_scores'])
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"DistilBERT results saved to: {output_path}")

def save_dataset_summary(df, output_path: str):
    """Save dataset summary statistics."""
    
    summary = {
        "dataset_info": {
            "total_reviews": len(df),
            "gaming_console_focus": "Strict filtering applied for gaming console hardware only",
            "sentiment_distribution": {
                "negative": int(df[df['sentiment'] == 0].shape[0]),
                "neutral": int(df[df['sentiment'] == 1].shape[0]),
                "positive": int(df[df['sentiment'] == 2].shape[0])
            },
            "rating_distribution": df['overall'].value_counts().sort_index().to_dict(),
            "average_review_length": float(df['clean_text'].str.len().mean()),
            "median_review_length": float(df['clean_text'].str.len().median())
        },
        "filtering_criteria": {
            "primary_keywords": [
                "playstation", "ps4", "ps5", "ps3", "ps2",
                "xbox", "xbox one", "xbox series", "xbox 360",
                "nintendo switch", "nintendo wii", "wii u",
                "gaming console", "video game console", "game console"
            ],
            "hardware_terms_required": [
                "console", "system", "hardware", "controller", "gamepad",
                "hdmi", "storage", "hard drive", "ssd", "graphics"
            ],
            "exclusion_terms": [
                "alexa", "echo", "fire tv", "fire stick", "kindle",
                "tablet", "phone", "mobile", "pc game", "computer game",
                "board game", "card game", "toy", "action figure"
            ]
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset summary saved to: {output_path}")

def main():
    """Main execution function for gaming console sentiment analysis."""
    
    print("="*80)
    print("GAMING CONSOLE SENTIMENT ANALYSIS - CLEAN IMPLEMENTATION")
    print("="*80)
    
    # Setup paths
    data_path = "../../Video_Games_5.json.gz"  # Adjust path as needed
    results_dir = "../results"
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    verification_dir = os.path.join(results_dir, "verification")
    
    # Create directories
    for directory in [results_dir, plots_dir, metrics_dir, verification_dir]:
        os.makedirs(directory, exist_ok=True)

    print("\nStep 1: Loading and processing gaming console data...")
    X_train, X_val, X_test, y_train, y_val, y_test, df, verification_samples = load_and_process_gaming_console_data(data_path)

    verification_path = os.path.join(verification_dir, "data_verification.txt")
    save_verification_samples(verification_samples, verification_path)
    
    dataset_summary_path = os.path.join(metrics_dir, "dataset_summary.json")
    save_dataset_summary(df, dataset_summary_path)
    
    X_train_list = X_train.tolist()
    X_val_list = X_val.tolist()
    X_test_list = X_test.tolist()
    y_train_list = y_train.tolist()
    y_val_list = y_val.tolist()
    y_test_list = y_test.tolist()
    

    print("\nStep 2: Training DistilBERT for gaming console sentiment analysis...")
    models_dir = os.path.join(results_dir, "models")
    
    train_result = train_distilbert_gaming_console(
        X_train_list, y_train_list,
        X_val_list, y_val_list,
        output_dir=models_dir,
        max_length=256,
        batch_size=16,
        num_epochs=3,
        learning_rate=2e-5
    )
    
    print("\nStep 3: Evaluating DistilBERT on test set...")
    test_result = evaluate_on_test_set(
        train_result['model'],
        train_result['tokenizer'],
        X_test_list,
        y_test_list,
        train_result['device'],
        max_length=256,
        batch_size=16
    )
    
    print("\nStep 4: Performing cross-validation...")
    cv_result = cross_validate_distilbert(
        df['clean_text'].tolist(),
        df['sentiment'].tolist(),
        k=5,
        max_length=256,
        batch_size=16,
        num_epochs=2
    )
    
    print("\nStep 5: Saving DistilBERT results...")
    results_path = os.path.join(metrics_dir, "distilbert_gaming_console_results.json")
    save_distilbert_results(train_result, train_result, test_result, cv_result, results_path)
    
    print("\nStep 6: Creating visualizations...")
    visualization_files = create_all_visualizations(
        df, train_result, train_result, test_result, cv_result, plots_dir
    )
    
    print("\n" + "="*80)
    print("GAMING CONSOLE SENTIMENT ANALYSIS - RESULTS SUMMARY")
    print("="*80)
    print(f"Dataset: {len(df):,} gaming console reviews")
    print(f"Training Accuracy: {1 - train_result['training_loss']:.4f}")
    print(f"Validation Accuracy: {train_result['val_accuracy']:.4f}")
    print(f"Test Accuracy: {test_result['test_accuracy']:.4f}")
    print(f"Cross-Validation: {cv_result['accuracy_mean']:.4f} ± {cv_result['accuracy_std']:.4f}")
    print(f"Cross-Validation F1: {cv_result['f1_mean']:.4f} ± {cv_result['f1_std']:.4f}")
    
    print(f"\nResults saved to: {results_dir}")
    print("Generated files:")
    print(f"  - Verification: {verification_path}")
    print(f"  - Dataset Summary: {dataset_summary_path}")
    print(f"  - DistilBERT Results: {results_path}")
    print(f"  - Visualizations: {len(visualization_files)} files in {plots_dir}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - GAMING CONSOLE SENTIMENT ANALYSIS")
    print("="*80)

if __name__ == "__main__":
    main()
