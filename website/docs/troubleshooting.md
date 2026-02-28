---
sidebar_position: 11
title: Troubleshooting
---

# Troubleshooting

## Common Issues

### "gcc: command not found"

Timber needs a C compiler to build shared libraries.

**macOS:**
```bash
xcode-select --install
```

**Ubuntu/Debian:**
```bash
sudo apt install build-essential
```

**Verify:**
```bash
gcc --version
```

### "model 'xyz' not loaded"

The model isn't in the store. Load it first:

```bash
timber load model.json --name xyz
timber list  # Verify it's there
timber serve xyz
```

### "expected N features, got M"

The input array has the wrong number of features. Check your model:

```bash
timber list
# Look at the FEATURES column
```

Ensure your input has exactly that many features per sample.

### Predictions Don't Match Python

1. **Check precision:** Timber uses double-precision accumulation, which may differ from single-precision frameworks at the 6th+ decimal place
2. **Check base_score:** XGBoost converts `base_score` from probability to logit space. Verify the conversion is correct
3. **Validate numerically:**
   ```bash
   timber validate --artifact ./dist/ --reference model.json --data test.csv --tolerance 1e-5
   ```

### Compilation Takes Too Long

Large models (>1000 trees) may produce very large C files. Options:
- Ensure `gcc -O3` is being used (the default)
- Check disk space in `~/.timber/`
- Use `--format` to skip auto-detection overhead

### Server Returns 500 Error

Check the terminal output where `timber serve` is running for stack traces. Common causes:
- Corrupted shared library (re-run `timber load`)
- Missing shared library (check `~/.timber/models/<name>/libtimber_model.*`)

### Model Auto-Detection Fails

If Timber picks the wrong format:

```bash
# Force the format explicitly
timber load model.json --format catboost
timber load model.json --format xgboost
```

CatBoost and XGBoost both use `.json` — Timber distinguishes them by looking for `oblivious_trees` (CatBoost) vs `learner` (XGBoost) keys.

## Getting Help

- **GitHub Issues:** [github.com/kossisoroyce/timber/issues](https://github.com/kossisoroyce/timber/issues)
- **Email:** kossi@electricsheep.africa

When filing a bug, include:
1. Timber version (`timber --version`)
2. OS and Python version
3. Model framework and format
4. Full error message
5. Minimal reproduction steps
