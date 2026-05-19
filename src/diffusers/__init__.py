# Copyright 2026 FD-Loss authors.
"""
FD-Loss generation stack in a Diffusers repository layout.

The on-disk paths under ``src/diffusers`` mirror the Hugging Face ``diffusers``
package so these modules can be copied into ``huggingface/diffusers`` when
upstreaming, following the same pattern as `NiT-diffusers
<https://github.com/Bili-Sakura/NiT-diffusers>`_.

Install with ``pip install -e .`` and import the Python package ``fdloss_diffusers``
(see ``pyproject.toml``): the install name avoids shadowing the PyPI ``diffusers``
library required for ``AutoencoderKL`` and other Hub components.
"""

from .models import (
    DiffusersAutoencoderKL,
    JiTDenoiser_models,
    VAE_models,
    iMFDenoiser_models,
    pMFDenoiser_models,
)

__all__ = [
    "DiffusersAutoencoderKL",
    "JiTDenoiser_models",
    "VAE_models",
    "iMFDenoiser_models",
    "pMFDenoiser_models",
]
