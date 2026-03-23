def test_import_model_pipeline():
    import model_pipeline as mp

    assert mp.ModelType.LGB.value == "lgb"
    assert hasattr(mp, "train")
    assert hasattr(mp, "load_dataset")
