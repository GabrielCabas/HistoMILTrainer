"""Training functions"""
import torch
from torch import nn, optim
from histomil.utils import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, classification_report,f1_score
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, results_dir, learning_rate, fold, epochs, patience = 2, stop_epoch = 2, class_weights = None, model_name = None):
    """
    Train function
    """
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights_tensor)
        print(criterion)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch, verbose=True)
    print("Start training")

    best_metrics = {
        "epoch": 0,
        "train_loss": float('inf'),
        "train_auc": 0.0,
        "train_acc": 0.0,
        "val_loss": float('inf'),
        "val_auc": 0.0,
        "val_acc": 0.0,
    }

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}.Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

        early_stopping(epoch, val_loss, model, ckpt_name=f"{results_dir}/{fold}-checkpoint.pt")

        # Save best epoch metrics
        if early_stopping.best_epoch == epoch:
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
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(f"{results_dir}/{fold}-checkpoint.pt"))
    return model, best_metrics

def test(model, test_loader, class_weights = None, model_name = None):
    """Test function: Evaluates clam model with optimal threshold selection by F1 macro."""
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights_tensor)
        print(criterion)
    else:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    all_labels, all_outputs = [], []
    correct = 0
    total = 0

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

    auc = roc_auc_score(all_labels, all_outputs)

    # Finding threshold
    best_f1_macro = 0
    best_threshold = 0.5
    thresholds = np.linspace(0, 1, 101)

    for thresh in thresholds:
        preds = (all_outputs >= thresh).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = thresh

    print(f"Optimal threshold for max F1 macro: {best_threshold:.3f} with F1 macro: {best_f1_macro:.3f}")

    pred_labels = (all_outputs >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels, pred_labels)
    accuracy = accuracy_score(all_labels, pred_labels)

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy (with optimal threshold): {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(all_labels, pred_labels, digits=4))

    metrics = {
        "test_auc": auc,
        "test_acc": accuracy,
        "optimal_threshold": best_threshold,
        "f1_macro": best_f1_macro
    }
    return metrics
