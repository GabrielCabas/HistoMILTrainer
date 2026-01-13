
"""Extracted from CLAM and TRIDENT"""
from src.builder import create_model

def import_model(model_name, pretrained_model, **kwargs):
    return create_model(f"{model_name}.base.{pretrained_model}", **kwargs)
