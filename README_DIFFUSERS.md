# Diffusers layout (FD-Loss)

This repository keeps the image generators (JiT, iMF, pMF) and tokenizer helpers in a tree that matches the Hugging Face **diffusers** package boundaries, following the same approach as [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git).

## Layout

- `src/diffusers/models/commons.py` — shared RoPE and layer utilities.
- `src/diffusers/models/autoencoder.py` — `DiffusersAutoencoderKL` Hub VAE wrapper.
- `src/diffusers/models/transformers/` — JiT backbone, MiT backbone, and the three denoiser modules.

## Install

Editable install registers the Python package **`fdloss_diffusers`** (mapped from `src/diffusers` in `pyproject.toml`) so the PyPI **`diffusers`** library remains importable for `AutoencoderKL` and other Hub types.

```bash
pip install -e .
```

Training code imports `fdloss_diffusers.models` instead of a top-level `models` package.

## Upstreaming

To propose these classes inside `huggingface/diffusers`, copy the files under `src/diffusers` into the matching paths in the diffusers repository and register lazy imports in that project’s public API tables.
