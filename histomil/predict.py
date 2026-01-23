"""Prediction and Heatmap functions"""
import json
import os
from pathlib import Path
import h5py
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import logging

from histomil import (
    H5DatasetPredict,
    import_model,
    variable_patches_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor:
    SEED = 2
    BATCH_SIZE = 16
    
    def __init__(self, csv_path, weights_path, features_folder, feature_extractor, results_dir, mil, params_path):
        self.logger = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.weights_path = weights_path
        self.features_folder = features_folder
        self.feature_extractor = feature_extractor
        self.results_dir = results_dir
        self.batch_size = 1 if mil == "clam" else self.BATCH_SIZE
        self.mil = mil
        self.params_path = params_path
        
        self.logger.info(f"Initializing Predictor with parameters:")
        self.logger.info(f"  - CSV path: {csv_path}")
        self.logger.info(f"  - Weights path: {weights_path}")
        self.logger.info(f"  - Features folder: {features_folder}")
        self.logger.info(f"  - Feature extractor: {feature_extractor}")
        self.logger.info(f"  - Results directory: {results_dir}")
        self.logger.info(f"  - MIL model: {mil}")
        self.logger.info(f"  - Params path: {params_path}")
        self.logger.info(f"  - Batch size: {self.batch_size}")
        self.logger.info(f"  - Device: {device}")

    def _load_data(self, csv_path):
        """Load data from csv file."""
        self.logger.info(f"Loading dataset from CSV: {csv_path}")
        dataset_csv = pd.read_csv(csv_path)
        self.logger.info(f"Dataset loaded: {len(dataset_csv)} slides")
        return dataset_csv
    
    def _create_loader(self, dataset_csv):
        """Create DataLoader for a given split."""
        self.logger.debug(f"Creating DataLoader with batch_size={self.batch_size}")
        loader = DataLoader(
            H5DatasetPredict(self.features_folder, dataset_csv, variable_patches=True),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=variable_patches_collate_fn,
        )
        self.logger.debug(f"DataLoader created: {len(loader)} batches")
        return loader

    @staticmethod
    def drop_extension(filepath):
        filename = Path(filepath)
        return filename.stem

    def predict(self, model, test_loader):
        """Predict function: Predicts the class of a set of slides."""
        self.logger.info("Starting prediction process")
        model.eval()
        all_outputs = []
        all_attentions = []
        total_slides = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Handle variable patches: batch is a list of (features, label) tuples
                # Variable patches mode: process each slide individually
                for features in batch:
                    features = features.to(device)  # Shape: (num_patches, feature_dim)
                    # Add batch dimension: (1, num_patches, feature_dim)
                    features = features.unsqueeze(0)
                    num_patches = features.shape[1]
                    self.logger.debug(f"Processing slide with {num_patches} patches")
                    
                    if self.mil in ["clam", "dftd"]:
                        logits, attn = model(features, 
                                          torch.tensor([1]).to(device),
                                          CrossEntropyLoss().to(device),
                                          return_attention=True)
                    else:
                        logits, attn = model(features, return_attention=True)
                    probs = torch.softmax(logits["logits"], dim=1)
                    all_outputs.append(probs[0, 1].cpu().item())  # prob. clase 1
                    all_attentions.append(attn["attention"].squeeze().cpu().numpy())
                    total_slides += 1
        # Convert lists to numpy arrays (handles both scalars and arrays)
        # List contains scalars (variable patches mode)
        all_outputs = np.array(all_outputs)
        self.logger.info(f"✓ Prediction completed: {total_slides} slides processed")
        self.logger.debug(f"Output shape: {all_outputs.shape}, Attention scores: {len(all_attentions)}")
        return all_outputs, all_attentions

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting prediction pipeline")
        self.logger.info("=" * 60)
        
        #Import model and load weights
        self.logger.info(f"Loading model parameters from: {self.params_path}")
        with open(self.params_path, "r") as f:
            params_dict = json.load(f)
        self.logger.debug(f"Model parameters: {params_dict}")
        
        self.logger.info(f"Importing {self.mil} model with {self.feature_extractor} feature extractor")
        model = import_model(self.mil, self.feature_extractor, **params_dict).to(device)
        
        self.logger.info(f"Loading model weights from: {self.weights_path}")
        model.load_state_dict(torch.load(self.weights_path))
        self.logger.info("✓ Model loaded successfully")
        
        dataset_df = self._load_data(self.csv_path)
        test_loader = self._create_loader(dataset_df)
        
        #Predict and extract attention scores
        y_prob, all_attentions = self.predict(model, test_loader)
        
        #Export predictions and attention scores
        self.logger.info("Exporting predictions and attention scores")
        results_df = pd.DataFrame({"slide_id": dataset_df["slide_id"], "prob": y_prob})
        results_df["pred"] = results_df["prob"].apply(lambda x: 1 if x > 0.5 else 0)
        
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        self.logger.info(f"Saving predictions to: {predictions_file}")
        results_df.to_csv(predictions_file, index=False)
        self.logger.info(f"✓ Predictions saved: {len(results_df)} slides")
        
        attention_dir = os.path.join(self.results_dir, "attention_scores")
        self.logger.info(f"Creating attention scores directory: {attention_dir}")
        os.makedirs(attention_dir, exist_ok=True)
        
        self.logger.info(f"Saving attention scores for {len(all_attentions)} slides")
        for slide_id, attn_score in zip(dataset_df["slide_id"], all_attentions):
            attn_file = os.path.join(attention_dir, self.drop_extension(slide_id) + ".h5")
            self.logger.debug(f"Saving attention scores for {slide_id} to {attn_file}")
            with h5py.File(attn_file, "w") as f:
                f.create_dataset("attention_scores", data=attn_score)
        
        self.logger.info("=" * 60)
        self.logger.info("✓ Prediction pipeline completed successfully")
        self.logger.info("=" * 60)
