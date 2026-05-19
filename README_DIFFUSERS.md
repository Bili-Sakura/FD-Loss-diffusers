FD-Loss Diffusers integration
=============================

Generation models now live under `src/diffusers`, mirroring the layout used for upstream
[Hugging Face Diffusers](https://github.com/huggingface/diffusers) integration (see
[NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git)).

Layout
------

- `models/transformers/transformer_jit.py` — `JiTTransformer2DModel` (`ModelMixin` / `ConfigMixin`)
- `models/transformers/transformer_mit.py` — `MiTTransformer2DModel` for pMF and iMF backbones
- `models/denoisers/` — training and sampling wrappers (`JiTDenoiser`, `pMFDenoiser`, `iMFDenoiser`)
- `models/autoencoders/autoencoder_fd.py` — VAE tokenizer wrapper around `diffusers.AutoencoderKL`
- `schedulers/scheduling_flow_match_fd.py` — `FDLossFlowMatchScheduler`
- `pipelines/jit/pipeline_jit.py` — `JiTPipeline` for class-conditional sampling

Runtime imports
---------------

Training and evaluation register the local tree as `fd_diffusers` so it does not shadow the
installed `diffusers` package (which still provides `AutoencoderKL` and other core types):

```python
from utils.fd_diffusers_bootstrap import register_fd_diffusers

register_fd_diffusers()
from fd_diffusers import JiTDenoiser_models, pMFDenoiser_models
```

Upstreaming
-----------

To contribute these modules to `huggingface/diffusers`, copy the files under `src/diffusers`
into the matching locations in the Diffusers repository and register the public classes in
Diffusers' lazy import tables.
