# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
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


def test_ring_sequence_parallel_is_rejected_before_loading():
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline

    od_config = OmniDiffusionConfig(model="unused", parallel_config=DiffusionParallelConfig(ring_degree=2))
    with pytest.raises(NotImplementedError, match="ring sequence parallel"):
        PAN2Pipeline(od_config=od_config)
