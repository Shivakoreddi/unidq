"""
UNIDQ Quickstart Example

This script demonstrates basic usage of the UNIDQ package for multi-task data quality.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from unidq import UNIDQ, UNIDQConfig, MultiTaskDataset, UNIDQTrainer
from unidq.utils import set_seed, get_device, create_synthetic_errors
from unidq.evaluation import evaluate_all_tasks


def main():
    """Run quickstart example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("=" * 60)
    print("UNIDQ Quickstart Example")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating sample dataset...")
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
        'age': [25, 30, 35, 40, 28, 33, 29, 45],
        'city': ['NYC', 'LA', 'Chicago', 'Houston', 'NYC', 'LA', 'Chicago', 'Boston'],
        'salary': [50000, 60000, 70000, 80000, 55000, 65000, 58000, 90000],
        'department': ['IT', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales'],
    })
    
    print(f"   Created dataframe with {len(df)} rows and {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Create synthetic labels for demonstration
    print("\n2. Creating synthetic task labels...")
    np.random.seed(42)
    
    task_labels = {
        'error_detection': pd.Series(np.random.randint(0, 2, len(df))),
        'duplicate_detection': pd.Series(np.random.randint(0, 2, len(df))),
        'outlier_detection': pd.Series(np.random.randint(0, 2, len(df))),
    }
    
    print(f"   Created labels for {len(task_labels)} tasks")
    
    # Step 3: Create dataset and dataloaders
    print("\n3. Creating datasets and dataloaders...")
    
    dataset = MultiTaskDataset(
        data=df,
        task_labels=task_labels,
        max_length=128,
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=MultiTaskDataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=MultiTaskDataset.collate_fn,
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Step 4: Initialize model
    print("\n4. Initializing UNIDQ model...")
    
    config = UNIDQConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        max_seq_length=128,
        vocab_size=256,  # Character-level
    )
    
    model = UNIDQ(config)
    device = get_device()
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Device: {device}")
    
    # Step 5: Create trainer
    print("\n5. Setting up trainer...")
    
    trainer = UNIDQTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
    )
    
    print("   Trainer initialized")
    
    # Step 6: Train the model
    print("\n6. Training model...")
    print("   Training for 2 epochs (demo)...")
    
    history = trainer.train(
        num_epochs=2,
        save_dir='./checkpoints',
        save_best=True,
    )
    
    print("\n   Training complete!")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Step 7: Evaluate model
    print("\n7. Evaluating model...")
    
    metrics = evaluate_all_tasks(
        model=model,
        dataloader=val_loader,
        device=device,
    )
    
    print("\n   Evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Step 8: Make predictions
    print("\n8. Making predictions on sample data...")
    
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
        
        outputs = model(
            sample_batch['input_ids'],
            sample_batch['attention_mask']
        )
        
        print("\n   Prediction outputs:")
        for task_name, task_output in outputs.items():
            print(f"   {task_name}: shape {task_output.shape}")
    
    # Step 9: Save model
    print("\n9. Saving model...")
    
    model.save_pretrained('./saved_model')
    print("   Model saved to './saved_model'")
    
    # Step 10: Load model
    print("\n10. Loading model...")
    
    loaded_model = UNIDQ.from_pretrained('./saved_model')
    print("   Model loaded successfully!")
    
    print("\n" + "=" * 60)
    print("Quickstart complete! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("- Check out the tutorial notebook in examples/notebooks/")
    print("- Customize the model configuration for your use case")
    print("- Train on your own data quality datasets")
    print("- Explore individual task evaluation functions")
    print("=" * 60)


if __name__ == '__main__':
    main()
