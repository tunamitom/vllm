# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


class MockDistinctEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor(
        [3] * len(input_ids), dtype=torch.int32, device=DEVICE
    )
    idx_mapping_np = idx_mapping.cpu().numpy()
    state.apply(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.tensor(local_pos, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_v2_thinking_budget_forces_end_after_budget_reached():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE).view(1, -1)
    expected = logits.cpu()
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    expected[0, END] = 1.0e9
    torch.testing.assert_close(out, expected)


def test_v2_thinking_budget_restores_masked_end_token():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    logits[0, END] = -float("inf")
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_allows_tokens_before_budget():
    req_states = _make_req_states([1, START, 10, 11], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[11], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_multi_token_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_uses_distinct_forced_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_stops_after_natural_end_marker():
    req_states = _make_req_states(
        [1, START, 10, END, 20, 21, 22],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[22], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_ignores_plain_request():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams())
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_greedy_sampling_applies_thinking_budget():
    """Greedy-only requests must not bypass thinking-budget processing."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    sampler = Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )
    sampler.add_request(
        req_idx=3,
        prompt_len=1,
        sampling_params=SamplingParams(
            temperature=0.0,
            thinking_token_budget=3,
        ),
    )
    sampler.apply_staged_writes()

    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    idx_mapping_np = idx_mapping.cpu().numpy()
    expanded_idx_mapping = idx_mapping.clone()
    input_ids = torch.tensor([12], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        input_ids,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
    )

    assert out[0, END].item() == pytest.approx(1.0e9)


def test_v2_thinking_budget_latest_prefill_end_disables_forcing():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[13], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_uses_latest_prefill_start_boundary():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13, START, 14, 15, 16],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[16], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_incrementally_scans_long_generation():
    """Guard against rescanning the full token history on every decode step."""
    tokens = [1, START, *([10] * 16382)]
    req_states = _make_req_states(tokens)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=32768))
    state.apply_staged_writes()

    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])
    assert state.cached_scan_pos[3].item() == len(tokens)

    req_states.all_token_ids.stage_write(3, len(tokens), [10])
    req_states.total_len.stage_write_elem(3, len(tokens) + 1)
    req_states.apply_staged_writes()
    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])

    assert state.cached_scan_pos[3].item() == len(tokens) + 1


def test_v2_thinking_budget_clamps_oversized_budget():
    """Budgets beyond int32 must not crash and behave as unlimited."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=2**40))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_ignores_resumed_prompt_reasoning():
    """Prompt-embedded reasoning no longer counts against the budget.

    Upstream used to count reasoning tokens from the last unclosed
    think-open anywhere in the token stream, including the prompt. A
    replayed history containing an aborted reasoning turn (unclosed
    think-open early in a long prompt) pre-exhausted the budget before
    the first sampled token, force-firing on step 1 (2026-09-27
    production incident). Only GENERATED reasoning counts now.
    """
    req_states = _make_req_states([1, START, 10, 11, END_A], prompt_len=5)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[END_A], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_end_prefix_from_template_suffix():
    """A generation prompt ending with the think-open marker (MiMo/Qwen3
    templates end the generation prompt with the think-open marker)
    still opens the reasoning window, and a partial forced-end marker at
    the end of the generated tail continues from the next marker token."""
    req_states = _make_req_states(
        [1, 2, 3, 4, 5, START, 10, 11, END_A], prompt_len=6
    )
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[END_A], local_pos=[0])

    assert out[0, END_B] == pytest.approx(1.0e9)
    assert out[0, END_A] == 0


def test_v2_thinking_budget_ignores_unclosed_prompt_think_open():
    """Regression (2026-09-27 production incident, arm B of the A/B/C
    proof): an unclosed think-open deep inside the prompt (replayed
    aborted reasoning) must NOT pre-exhaust the budget."""
    tokens = [START, 10, 11] + [20] * 17
    req_states = _make_req_states(tokens, prompt_len=20)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[20], local_pos=[0])

    assert torch.all(out == 0)
    # Second apply exercises the incremental scan path with the floor.
    out2 = _apply(state, logits, input_ids=[20, 20], local_pos=[0, 1])
    assert torch.all(out2 == 0)


def test_v2_thinking_budget_counts_template_suffix_think_open():
    """The think-open marker at prompt_len - START_LEN (chat-template
    generation prompt) is within the allowed window and must count."""
    req_states = _make_req_states(
        [1, 2, 3, 4, 5, START, 10, 11, 12], prompt_len=6
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_ignores_prompt_natural_ends():
    """Natural-end markers inside the prompt (closed reasoning turns in
    replayed history) must not close the CURRENT turn's reasoning window:
    the end-marker scan shares the prompt floor, so the template
    think-open at the prompt tail still opens the window and generated
    reasoning still forces at the budget (review #5, gap 1; the
    generated-region counterpart — a natural end after a generated
    think-open disables the force via last_start <= last_end — is pinned
    by test_v2_thinking_budget_latest_prefill_end_disables_forcing)."""
    tokens = [
        1, START, 10, 11, END, 20, 21, END, 30, 31, 32,
        START, 40, 41, 42,
    ]
    req_states = _make_req_states(tokens, prompt_len=12)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[42], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)
