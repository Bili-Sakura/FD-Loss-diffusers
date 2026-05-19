import pytest

torch = pytest.importorskip("torch")

from utils.fd_diffusers_bootstrap import register_fd_diffusers

register_fd_diffusers()

from fd_diffusers import (  # noqa: E402
    FDLossFlowMatchScheduler,
    JiTTransformer2DModel,
    MiTTransformer2DModel,
)


def test_jit_transformer_forward():
    model = JiTTransformer2DModel(
        input_size=32,
        patch_size=8,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
        bottleneck_dim=16,
        in_context_len=0,
        in_context_start=0,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.5, 0.25])
    y = torch.tensor([1, 2])
    out = model(x, t, y)
    assert out.sample.shape == x.shape


def test_mit_transformer_forward():
    model = MiTTransformer2DModel(
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=64,
        depth=4,
        num_heads=4,
        num_classes=10,
        aux_head_depth=2,
        num_class_tokens=2,
        num_time_tokens=2,
        num_cfg_tokens=2,
        num_interval_tokens=1,
        output_type="v",
        disable_v_head=False,
    )
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([0.5, 0.25])
    h = torch.tensor([0.1, 0.05])
    omega = torch.tensor([2.0, 3.0])
    t_min = torch.tensor([0.0, 0.0])
    t_max = torch.tensor([1.0, 1.0])
    y = torch.tensor([1, 2])
    out = model(x, t, h, omega=omega, t_min=t_min, t_max=t_max, y=y)
    assert out.sample.shape == x.shape
    assert out.aux_sample.shape == x.shape


def test_flow_match_scheduler_step():
    scheduler = FDLossFlowMatchScheduler()
    scheduler.set_timesteps(4)
    sample = torch.ones(1, 3, 2, 2)
    velocity = torch.full_like(sample, 2.0)
    output = scheduler.step(velocity, torch.tensor([1.0]), sample, torch.tensor([0.75]))
    assert torch.allclose(output.prev_sample, torch.full_like(sample, 0.5))
