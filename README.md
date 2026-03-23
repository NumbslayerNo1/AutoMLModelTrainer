# TrainLGBMexico

Offline modeling utilities: `model_pipeline` package for load → split → train (LGB) → predict → tune.

See `specs/model_pipeline/quickstart.md` for a minimal example.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=model_pipeline --cov-report=term-missing
```

**macOS:** if `import lightgbm` fails with missing `libomp.dylib`, run `brew install libomp`. Parquet I/O needs **pyarrow** (already listed in `pyproject.toml`).
# AutoMLModelTrainer
