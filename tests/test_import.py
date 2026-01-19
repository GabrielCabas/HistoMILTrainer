"""Test that the histomil library can be imported."""


def test_import_histomil():
    """Test that histomil can be imported successfully."""
    import histomil  # noqa: F401

    # Verify that the module has the expected attributes
    assert hasattr(histomil, "SplitManager")
    assert hasattr(histomil, "H5Dataset")
    assert hasattr(histomil, "variable_patches_collate_fn")
    assert hasattr(histomil, "seed_torch")
    assert hasattr(histomil, "get_weights")
    assert hasattr(histomil, "train")
    assert hasattr(histomil, "test")
    assert hasattr(histomil, "EarlyStopping")
    assert hasattr(histomil, "import_model")

