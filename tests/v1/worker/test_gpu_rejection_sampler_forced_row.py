# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thinking-budget forced rows must survive rejection sampling.

``_thinking_budget_kernel`` forces the reasoning-end marker by writing a
1e9 sentinel logit on the forced token. In production (DFlash drafts,
vocab-parallel resample, adaptive verification, temperature > 0) the
committed token was observed to be a degenerate repeat cycle instead of
the forced marker; the resample kernels now bypass the stochastic path
deterministically whenever the sentinel is present.

Note: with well-formed standalone inputs the unfixed kernels also commit
the forced token (the token assertions here pass on the unpatched tree);
the guards pin that contract and remove the sentinel's dependence on
stochastic numerics. The production loss mode involves the
adaptive-verification compacted layout and is validated at engine level
after the image rebuild.
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for forced-row rejection sampler tests",
        allow_module_level=True,
    )

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

DEVICE = torch.device("cuda")
VOCAB_SIZE = 1024
NUM_SPECULATIVE_STEPS = 2
LOGITS_PER_REQ = NUM_SPECULATIVE_STEPS + 1
FORCED_TOKEN = 7


def _run(
    target_logits: torch.Tensor,
    draft_tokens: tuple[int, int] = (11, 12),
) -> tuple[torch.Tensor, torch.Tensor]:
    """One request, k=2, draft logits present, temperature 0.9."""
    num_reqs = 1
    num_logits = num_reqs * LOGITS_PER_REQ
    gen = torch.Generator(device="cpu").manual_seed(42)
    draft_logits = torch.randn(
        1,
        NUM_SPECULATIVE_STEPS,
        VOCAB_SIZE,
        dtype=torch.float32,
        generator=gen,
    ).to(DEVICE)
    # Draft tokens deliberately != FORCED_TOKEN unless a test opts in.
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=DEVICE)
    draft_sampled[1] = draft_tokens[0]
    draft_sampled[2] = draft_tokens[1]
    cu_num_logits = torch.tensor(
        [0, num_logits], dtype=torch.int32, device=DEVICE
    )
    idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = idx_mapping.repeat_interleave(LOGITS_PER_REQ)
    expanded_local_pos = (
        torch.arange(LOGITS_PER_REQ, dtype=torch.int32, device=DEVICE)
        .repeat(num_reqs)
        .contiguous()
    )
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE)
    temperature = torch.full((1,), 0.9, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((1,), 42, dtype=torch.int64, device=DEVICE)
    return rejection_sample(
        target_logits.to(DEVICE),
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
        NUM_SPECULATIVE_STEPS,
    )


def test_forced_row_commits_forced_token_at_temperature():
    gen = torch.Generator(device="cpu").manual_seed(7)
    target = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.float32, generator=gen
    )
    # Force at the first draft position: sentinel on the forced token.
    target[0, FORCED_TOKEN] = 1.0e9
    sampled, num_sampled = _run(target)

    # The draft token at the forced position is rejected with certainty
    # (its target probability underflows against the sentinel), so the
    # resample commits the first token: it must be the forced marker,
    # not a stochastic draw.
    assert num_sampled[0].item() == 1
    assert sampled[0, 0].item() == FORCED_TOKEN


def test_unforced_row_resamples_normally():
    gen = torch.Generator(device="cpu").manual_seed(7)
    target = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.float32, generator=gen
    )
    sampled, num_sampled = _run(target)

    # No sentinel: the first draft may be accepted or rejected; either
    # way a valid in-vocab token commits and the guard must not fire.
    assert 0 <= sampled[0, 0].item() < VOCAB_SIZE
    assert num_sampled[0].item() >= 1


def test_forced_bonus_row_commits_forced_token():
    """Budget tripping exactly at the bonus position (both draft tokens
    accepted, sentinel on the target-only row): the forced token must
    commit at the bonus slot (review #5, gap 2)."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    target = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.float32, generator=gen
    )
    # Make both draft tokens certain winners so the bonus row is reached.
    target[0, 11] = 50.0
    target[1, 12] = 50.0
    target[2, FORCED_TOKEN] = 1.0e9  # bonus row

    sampled, num_sampled = _run(target)

    assert num_sampled[0].item() == 3
    assert sampled[0, 0].item() == 11
    assert sampled[0, 1].item() == 12
    assert sampled[0, 2].item() == FORCED_TOKEN


def test_draft_proposing_forced_token_is_accepted():
    """If the draft itself proposes the forced token on a sentinel row,
    verification accepts it directly (p/q >= 1 against the sentinel):
    the correct token commits without resampling (review #5, gap 3)."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    target = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.float32, generator=gen
    )
    target[0, FORCED_TOKEN] = 1.0e9

    sampled, num_sampled = _run(target, draft_tokens=(FORCED_TOKEN, 12))

    assert sampled[0, 0].item() == FORCED_TOKEN
    assert num_sampled[0].item() >= 1


def test_vp_resample_guard_commits_forced_token_deterministically():
    """The vocab-parallel resample path (DFlash drafts) must commit the
    forced token through the sentinel guard, storing the raw sentinel
    value instead of a noised residual.

    On the unpatched tree the token assertion still passes (the residual
    math defends the sentinel with well-formed inputs) but the stored
    value is a small noised residual, not the ~1e9 sentinel: that is the
    guarded behavior this test pins.
    """
    from vllm.v1.worker.gpu.spec_decode import vocab_parallel as vp_sampling

    gen = torch.Generator(device="cpu").manual_seed(7)
    target = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.float32, generator=gen
    )
    target[0, FORCED_TOKEN] = 1.0e9

    draft = torch.randn(
        1, NUM_SPECULATIVE_STEPS, VOCAB_SIZE, dtype=torch.float32,
        generator=gen,
    ).to(DEVICE)
    cache = vp_sampling.VPDraftCache(
        draft, vocab_start=0, vocab_size=VOCAB_SIZE, tp_group=None
    )
    temp = 0.9
    # Per-step draft logsumexp, the way the draft phase fills the cache.
    for step in range(NUM_SPECULATIVE_STEPS):
        cache.lse[0, step] = torch.logsumexp(
            draft[0, step] / temp, dim=-1
        )

    draft_sampled = torch.zeros(
        LOGITS_PER_REQ, dtype=torch.int64, device=DEVICE
    )
    draft_sampled[1] = 11
    draft_sampled[2] = 12
    cu_num_logits = torch.tensor(
        [0, LOGITS_PER_REQ], dtype=torch.int32, device=DEVICE
    )
    expanded_idx_mapping = torch.zeros(
        LOGITS_PER_REQ, dtype=torch.int32, device=DEVICE
    )
    rejected_step = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    temperature = torch.full((1,), temp, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((1,), 42, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(LOGITS_PER_REQ, dtype=torch.int64, device=DEVICE)
    target_rejected_lse = (
        torch.logsumexp(target[0], dim=-1).reshape(1).to(DEVICE)
    )

    out = vp_sampling.resample_local(
        target.to(DEVICE),
        target_rejected_lse,
        rejected_step,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seeds,
        pos,
        cache,
    )

    # Winner (value, token): the sentinel row must win with the raw ~1e9
    # sentinel value, committing the forced marker deterministically.
    assert out[0, 1].item() == FORCED_TOKEN
    assert out[0, 0].item() > 1.0e8
