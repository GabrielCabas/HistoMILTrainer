"""Prediction and Heatmap functions"""
import os
from pathlib import Path
import h5py
from trident import OpenSlideWSI, visualize_heatmap
import logging


class HeatmapVisualizer:
    def __init__(self, slide_id, slide_folder, features_folder, attn_scores_folder, results_dir):
        self.logger = logging.getLogger(__name__)
        self.slide_id = slide_id #Slide filename
        self.slide_folder = slide_folder
        self.features_folder = features_folder
        self.results_dir = results_dir
        self.attn_scores_folder = attn_scores_folder #Attention scores for one slide
        
        self.logger.info(f"Initializing HeatmapVisualizer for slide: {slide_id}")
        self.logger.info(f"  - Slide folder: {slide_folder}")
        self.logger.info(f"  - Features folder: {features_folder}")
        self.logger.info(f"  - Attention scores folder: {attn_scores_folder}")
        self.logger.info(f"  - Results directory: {results_dir}")

    @staticmethod
    def drop_extension(filepath):
        filename = Path(filepath)
        return filename.stem

    def _load_slide(self):
        slide_path = os.path.join(self.slide_folder, self.slide_id)
        self.logger.info(f"Loading slide from: {slide_path}")
        slide = OpenSlideWSI(slide_path=slide_path, lazy_init=False)
        self.logger.debug(f"Slide loaded: dimensions={slide.dimensions if hasattr(slide, 'dimensions') else 'N/A'}")
        return slide

    def _load_attention_scores(self):
        attention_scores_path = os.path.join(self.attn_scores_folder, self.drop_extension(self.slide_id) + ".h5")
        self.logger.info(f"Loading attention scores from: {attention_scores_path}")
        with h5py.File(attention_scores_path, 'r') as f:
            attention_scores = f['attention_scores'][:]
        self.logger.debug(f"Attention scores loaded: shape={attention_scores.shape}")
        return attention_scores

    def _load_coords(self):
        features_path = os.path.join(self.features_folder, self.drop_extension(self.slide_id) + ".h5")
        self.logger.info(f"Loading coordinates from: {features_path}")
        with h5py.File(features_path, 'r') as f:
            coords = f['coords'][:]
            coords_attrs = dict(f['coords'].attrs)
        self.logger.debug(f"Coordinates loaded: shape={coords.shape}, patch_size_level0={coords_attrs.get('patch_size_level0', 'N/A')}")
        return coords, coords_attrs

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info(f"Starting heatmap visualization for slide: {self.slide_id}")
        self.logger.info("=" * 60)
        
        slide = self._load_slide()
        coords, coords_attrs = self._load_coords()
        attention_scores = self._load_attention_scores()
        
        self.logger.info(f"Creating output directory: {self.results_dir}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        output_filename = "heatmap_" + self.drop_extension(self.slide_id) + ".png"
        self.logger.info(f"Generating heatmap visualization")
        self.logger.debug(f"  - Visualization level: 1")
        self.logger.debug(f"  - Patch size level 0: {coords_attrs['patch_size_level0']}")
        self.logger.debug(f"  - Normalize: True")
        self.logger.debug(f"  - Top patches to save: 20")
        self.logger.debug(f"  - Output filename: {output_filename}")
        
        visualize_heatmap(
            wsi=slide,
            scores=attention_scores,
            coords=coords,
            vis_level=1,
            patch_size_level0=coords_attrs['patch_size_level0'],
            normalize=True,
            num_top_patches_to_save=20,
            output_dir=self.results_dir,
            filename=output_filename
        )
        
        output_path = os.path.join(self.results_dir, output_filename)
        self.logger.info("=" * 60)
        self.logger.info(f"✓ Heatmap visualization completed: {output_path}")
        self.logger.info("=" * 60)