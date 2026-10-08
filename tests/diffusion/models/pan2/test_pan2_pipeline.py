# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import dataclasses

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StopAtLatentsError(Exception):
    pass


class _FakeTransformer:
    """The attributes `PAN2Pipeline.forward` reads from the transformer before latent preparation.

    Not an `nn.Module`: the pipeline under test is built without `nn.Module.__init__`, so it cannot hold submodules.
    """

    patch_size = (1, 2, 2)

    def __init__(self):
        self.x_embedder = nn.Linear(1, 1)


def _resolved_num_frames(request_id="pan2-test", **sampling_overrides) -> int:
    """Run `PAN2Pipeline.forward` up to latent preparation and return the number of frames it resolved."""
    from vllm_omni.diffusion.cache.cachedit import RequestScopedCacheDiTRuntime
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2QualityPolicy

    pipeline = object.__new__(PAN2Pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor_spatial = 16
    pipeline.vae_scale_factor_temporal = 4
    pipeline.sequential_cfg_branches = False
    pipeline._quality_policy = PAN2QualityPolicy(OmniDiffusionConfig(cache_backend="none"))
    pipeline._cache_dit_runtime = RequestScopedCacheDiTRuntime(pipeline)
    pipeline.transformer = _FakeTransformer()
    pipeline.encode_prompt = lambda _prompt, _device: torch.zeros(1, 1, 4)

    def prepare_latents(height, width, num_frames, device, generator, latents):
        pipeline.num_frames = num_frames
        raise _StopAtLatentsError

    pipeline.prepare_latents = prepare_latents

    sampling_params = OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=2, **sampling_overrides)
    request = OmniDiffusionRequest(prompt="a cat", sampling_params=sampling_params, request_id=request_id)
    with pytest.raises(_StopAtLatentsError):
        pipeline.forward(DiffusionRequestBatch([request]))
    return pipeline.num_frames


def test_omitted_num_frames_uses_the_pan2_default():
    from vllm_omni.diffusion.models.pan2.pipeline_pan2 import PAN2_DEFAULT_NUM_FRAMES

    assert OmniDiffusionSamplingParams().num_frames == 1
    assert _resolved_num_frames() == PAN2_DEFAULT_NUM_FRAMES


def test_requested_num_frames_is_kept():
    assert _resolved_num_frames(num_frames=9) == 9


def test_dummy_run_keeps_a_single_frame():
    assert _resolved_num_frames(request_id=DUMMY_DIFFUSION_REQUEST_ID, num_frames=1) == 1


def test_request_output_type_latent_returns_the_latents():
    from vllm_omni.diffusion.cache.cachedit import RequestScopedCacheDiTRuntime
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2QualityPolicy

    pipeline = object.__new__(PAN2Pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor_spatial = 16
    pipeline.vae_scale_factor_temporal = 4
    pipeline.num_channels_latents = 4
    pipeline.sequential_cfg_branches = False
    pipeline._quality_policy = PAN2QualityPolicy(OmniDiffusionConfig(cache_backend="none"))
    pipeline._cache_dit_runtime = RequestScopedCacheDiTRuntime(pipeline)
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    pipeline.encode_prompt = lambda _prompt, _device: torch.zeros(1, 1, 4)
    # One denoising step that leaves the latents unchanged; the VAE is never set, so decoding would fail.
    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: torch.zeros_like(
        kwargs["positive_kwargs"]["hidden_states"]
    )
    pipeline.scheduler_step_maybe_with_cfg = lambda noise_pred, t, latents, do_true_cfg: latents

    sampling_params = OmniDiffusionSamplingParams(
        height=64, width=64, num_frames=5, num_inference_steps=1, output_type="latent"
    )
    request = OmniDiffusionRequest(prompt="a cat", sampling_params=sampling_params, request_id="pan2-test")
    output = pipeline.forward(DiffusionRequestBatch([request])).output

    assert output.shape == (1, 4, 2, 4, 4)


class _FakeComponent:
    """A loaded tokenizer, text encoder or scheduler: only `.to` is used at construction."""

    def to(self, *args, **kwargs):
        return self


@dataclasses.dataclass
class _FakeVAEConfig:
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    z_dim: int = 48


class _FakeVAE(_FakeComponent):
    config = _FakeVAEConfig()


class _FakePAN2Transformer:
    def __init__(self, **kwargs):
        from vllm_omni.diffusion.models.pan2 import PAN2Transformer3DModel

        self._cache_dit_adapter_config = PAN2Transformer3DModel._cache_dit_adapter_config


def test_every_component_loads_the_requested_revision(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline, pipeline_pan2

    calls: list[tuple[str, str | None]] = []

    class _Loader:
        def __init__(self, name: str, component: _FakeComponent):
            self.name = name
            self.component = component

        def from_pretrained(self, *args, revision: str | None = None, **kwargs) -> _FakeComponent:
            calls.append((self.name, revision))
            return self.component

    def prefetch_subfolders(model, subfolders, *, revision=None, **kwargs) -> None:
        calls.append(("prefetch", revision))

    def from_pretrained_with_prefetch(factory, model, *, subfolder, revision=None, **kwargs) -> _FakeComponent:
        calls.append((subfolder, revision))
        return _FakeVAE() if subfolder == "vae" else _FakeComponent()

    monkeypatch.setattr(pipeline_pan2, "prefetch_subfolders", prefetch_subfolders)
    monkeypatch.setattr(pipeline_pan2, "from_pretrained_with_prefetch", from_pretrained_with_prefetch)
    monkeypatch.setattr(pipeline_pan2, "AutoTokenizer", _Loader("tokenizer", _FakeComponent()))
    monkeypatch.setattr(pipeline_pan2, "FlowMatchEulerDiscreteScheduler", _Loader("scheduler", _FakeComponent()))
    monkeypatch.setattr(pipeline_pan2, "PAN2Transformer3DModel", _FakePAN2Transformer)

    pipeline = PAN2Pipeline(od_config=OmniDiffusionConfig(model="IFM/PAN2", revision="abc123"))

    assert sorted(calls) == sorted(
        (name, "abc123") for name in ("prefetch", "tokenizer", "text_encoder", "vae", "scheduler")
    )
    assert [source.revision for source in pipeline.weights_sources] == ["abc123"]


@pytest.mark.parametrize("degree", ["ring_degree", "allgather_degree"])
def test_ring_and_allgather_sequence_parallel_are_rejected_before_loading(degree: str):
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline

    od_config = OmniDiffusionConfig(model="unused", parallel_config=DiffusionParallelConfig(**{degree: 2}))
    with pytest.raises(NotImplementedError, match="ring or all-gather sequence parallel"):
        PAN2Pipeline(od_config=od_config)
