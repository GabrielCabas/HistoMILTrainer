"""Training functions"""
import torch
from torch import nn, optim
from histomil.utils import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, classification_report, f1_score
import numpy as np
from tqdm import tqdm
import os
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, results_dir, learning_rate, fold, epochs, patience = 2,
    stop_epoch = 2, class_weights = None, model_name = None, params = None):
    """
    Train function
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"Starting training for fold {fold}")
    logger.info("=" * 60)
    logger.info(f"Training parameters:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Patience: {patience}")
    logger.info(f"  - Stop epoch: {stop_epoch}")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Train batches: {len(train_loader)}")
    logger.info(f"  - Val batches: {len(val_loader)}")
    
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights_tensor)
        logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    logger.debug(f"Optimizer: AdamW with lr={learning_rate}")
    
    early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch, verbose=True)
    logger.info("Start training")

    best_metrics = {
        "epoch": 0,
        "train_loss": float('inf'),
        "train_auc": 0.0,
        "train_acc": 0.0,
        "val_loss": float('inf'),
        "val_auc": 0.0,
        "val_acc": 0.0,
    }

    # Generate checkpoint name once, outside the loop
    if params is not None:
        string_params = "_".join([f"{k}={v}" for k, v in params.items()])
    else:
        string_params = ""
    output_name = os.path.abspath(f"{results_dir}/{fold}-{string_params}-checkpoint.pt")
    logger.info(f"Checkpoint will be saved to: {output_name}")

    for epoch in range(epochs):
        logger.info("-" * 60)
        logger.info(f"Epoch {epoch+1}/{epochs}")
        # Training
        model.train()
        train_loss = 0.
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Handle variable patches: batch is a list of (features, label) tuples
            # Variable patches mode: process each slide individually
            optimizer.zero_grad()
            batch_loss = 0.
            
            # Process each slide and accumulate gradients
            for features, label in batch:
                features = features.to(device)  # Shape: (num_patches, feature_dim)
                label = label.to(device).long().unsqueeze(0)  # Shape: (1,)
                
                # Add batch dimension: (1, num_patches, feature_dim)
                features = features.unsqueeze(0)
                
                # Forward pass
                if model_name in ["clam", "dftd"]:
                    logits, attn = model(features, label, criterion)  # Shape: (1, 2)
                else:
                    logits, attn = model(features)
                loss = criterion(logits["logits"], label)
                
                # Scale loss by batch size for proper averaging
                loss = loss / len(batch)
                batch_loss += loss.item() * len(batch)  # Store unscaled for reporting
                
                # Backward pass (accumulates gradients)
                loss.backward()
                
                # Collect predictions
                probs = torch.softmax(logits["logits"], dim=1)
                train_preds.append(probs[0, 1].detach().cpu().item())
                train_labels.append(label[0].detach().cpu().item())
            
            # Update weights once after processing all slides in batch
            optimizer.step()
            
            # Average loss across slides in batch
            total_loss = batch_loss / len(batch)
            train_loss += total_loss
        train_loss /= len(train_loader)

        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, np.array(train_preds) > 0.5)
        logger.debug(f"Training metrics computed: {len(train_labels)} samples")

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad(): #No grad
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle variable patches: batch is a list of (features, label) tuples
                # Variable patches mode: process each slide individually
                batch_loss = 0.
                for features, label in batch:
                    features = features.to(device)  # Shape: (num_patches, feature_dim)
                    label = label.to(device).long().unsqueeze(0)  # Shape: (1,)
                    
                    # Add batch dimension: (1, num_patches, feature_dim)
                    features = features.unsqueeze(0)
                    
                    # Forward pass
                    if model_name in ["clam", "dftd"]:
                        logits, attn = model(features, label, criterion)  # Shape: (1, 2)
                    else:
                        logits, attn = model(features)
                    loss = criterion(logits["logits"], label)
                    
                    batch_loss += loss.item()
                    
                    # Collect predictions
                    probs = torch.softmax(logits["logits"], dim=1)
                    val_preds.append(probs[0, 1].cpu().item())
                    val_labels.append(label[0].cpu().item())
                
                # Average loss across slides in batch
                total_loss = batch_loss / len(batch)
                val_loss += total_loss

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
        logger.debug(f"Validation metrics computed: {len(val_labels)} samples")

        logger.info(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        early_stopping(epoch, val_loss, model, ckpt_name=output_name)

        # Save best epoch metrics
        if early_stopping.best_epoch == epoch:
            logger.info(f"✓ New best model at epoch {epoch+1} (val_loss: {val_loss:.4f})")
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_acc": val_acc,
            }

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Loading best model from checkpoint: {output_name}")
    model.load_state_dict(torch.load(output_name))
    logger.info("=" * 60)
    logger.info(f"✓ Training completed for fold {fold}")
    logger.info(f"Best epoch: {best_metrics['epoch']}, Best val AUC: {best_metrics['val_auc']:.4f}")
    logger.info("=" * 60)
    return model, best_metrics, output_name

def test(model, test_loader, class_weights = None, model_name = None):
    """Test function: Evaluates clam model with optimal threshold selection by F1 macro."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting model evaluation on test set")
    logger.info("=" * 60)
    logger.info(f"Test batches: {len(test_loader)}")
    
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights_tensor)
        logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")
    
    model.eval()
    all_labels, all_outputs = [], []
    correct = 0
    total = 0

    logger.info("Processing test batches")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Handle variable patches: batch is a list of (features, label) tuples
            # Variable patches mode: process each slide individually
            for features, label in batch:
                features = features.to(device)  # Shape: (num_patches, feature_dim)
                label = label.to(device).long().unsqueeze(0)  # Shape: (1,)
                
                # Add batch dimension: (1, num_patches, feature_dim)
                features = features.unsqueeze(0)
                
                # Forward pass
                if model_name in ["clam", "dftd"]:
                    logits, attn = model(features, label, criterion)  # Shape: (1, 2)
                else:
                    logits, attn = model(features)
                probs = torch.softmax(logits["logits"], dim=1)
                predicted = torch.argmax(probs, dim=1)  # predicted: [1]
                
                correct += (predicted == label).sum().item()
                total += label.size(0)
                
                all_outputs.append(probs[0, 1].cpu().item())  # prob. clase 1
                all_labels.append(label[0].cpu().item())

    # Convert lists to numpy arrays (handles both scalars and arrays)
    # List contains scalars (variable patches mode)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    logger.info(f"Computing metrics on {len(all_labels)} test samples")
    auc = roc_auc_score(all_labels, all_outputs)
    pred_labels = (all_outputs >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    f1_macro = f1_score(all_labels, pred_labels, average='macro')
    
    logger.info(f"Test accuracy: {correct}/{total} = {correct/total:.4f}")

    metrics = {
        "test_auc": auc,
        "test_acc": accuracy,
        "f1_macro": f1_macro
    }
    
    logger.info("=" * 60)
    logger.info("✓ Test evaluation completed")
    logger.info(f"Test metrics: AUC={auc:.4f}, Acc={accuracy:.4f}, F1={f1_macro:.4f}")
    logger.info("=" * 60)
    
    return metrics, all_outputs, all_labels
