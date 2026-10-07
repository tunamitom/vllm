# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-family loader contract tests (Codex §12.2 / §9.3 requests).

The strict geometry + source-index identity pin applies ONLY to the
``mimo_v26_pro_mxfp4`` family. Upstream families (deepseek_v41, kimi_k3,
deepseek_v4_flash) keep stock pass-through behavior for geometry-mismatched
configs and are covered by their own upstream fixtures.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.mxfp4_csf_loader import (
    CODEC,
    SCHEMA,
    Mxfp4CsfModelLoader,
)


def _mimo_container(tmp_path, *, with_identity=True, source_index=None):
    """A minimal valid MiMo CSF container + matching model root."""
    tensor_dir = tmp_path / "container"
    tensor_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    name = "model-00001.safetensors"
    tensors = {
        "model.layers.5.mlp.experts.0.w1.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        "model.layers.5.mlp.experts.0.w1.scale.mxfp4_csf_fixed": torch.ones(
            8, dtype=torch.uint8
        ),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / name)
    common = {
        "schema": SCHEMA,
        "codec": CODEC,
        "family": "mimo_v26_pro_mxfp4",
    }
    contract = {
        **common,
        "shards": [{"file": name}],
        "source_names": {k: name for k in tensors},
    }
    if with_identity:
        contract["source_index_sha256"] = (
            source_index
            if source_index is not None
            else hashlib.sha256(b"model.safetensors.index.json").hexdigest()
        )
    (tensor_dir / "build-contract.json").write_text(json.dumps(contract))
    (tensor_dir / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": name}]})
    )
    # model root with the index the identity pin checks
    index_bytes = b"model.safetensors.index.json"
    (model_dir / "model.safetensors.index.json").write_bytes(index_bytes)
    default_sha = hashlib.sha256(index_bytes).hexdigest()
    return tensor_dir, model_dir, default_sha


def _mimo_model_config(container, model_dir, *, geometry=(384, 6144, 2048)):
    e, h, n = geometry
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            num_hidden_layers=70,
        ),
        hf_text_config=SimpleNamespace(
            model_type="mimo_v2",
            n_routed_experts=e,
            hidden_size=h,
            moe_intermediate_size=n,
        ),
        model=str(model_dir),
    )


def test_mimo_identity_pin_accepts_matching_source_index(tmp_path):
    """Positive MiMo case: container + matching source index resolves."""
    container, model_dir, sha = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = sha
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    resolved = loader._resolve_csf_root(
        contract_like := {
            "quant_method": "mxfp4_csf",
            "checkpoint_root": str(container),
        },
        _mimo_model_config(container, model_dir).hf_text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "mimo_v26_pro_mxfp4"


def test_mimo_identity_pin_rejects_foreign_source_index(tmp_path):
    """Foreign source checkpoint (index sha mismatch) must refuse."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = (
        hashlib.sha256(b"some-other-checkpoints-index").hexdigest()
    )
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="built from a different source"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(container, model_dir).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_identity_pin_requires_recorded_sha(tmp_path):
    """MiMo container without source_index_sha256 must refuse (no bypass)."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path, with_identity=False)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="lacks source_index_sha256"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(container, model_dir).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_geometry_mismatch_refuses_csf_load(tmp_path):
    """Explicit CSF intent on a geometry-mismatched MiMo config refuses."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    with pytest.raises(ValueError, match="geometry does not match"):
        loader._resolve_csf_root(
            {
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(container),
            },
            _mimo_model_config(
                container, model_dir, geometry=(256, 6144, 2048)
            ).hf_text_config,
            model_root=str(model_dir),
        )


def test_mimo_plain_source_checkpoint_passes_through(tmp_path):
    """Hybrid MiMo source WITHOUT CSF intent: plain FP8 launch, no raise.

    The hybrid intent gate (fp8 + store_dtype mxfp4 + model_type mimo_v2*)
    requires MIMO_CSF_ROOT or checkpoint_root; absent that, this is the plain
    source checkpoint served as FP8 — must return None (probe contract).
    """
    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        {"quant_method": "fp8"}, text_config, model_root=str(model_dir)
    )
    assert resolved is None


def test_mimo_dflash_draft_passes_through(tmp_path):
    """DFlash draft model (no quantization_config at all) passes through."""
    container, model_dir, _ = _mimo_container(tmp_path)
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        None, text_config, model_root=str(model_dir)
    )
    assert resolved is None


def test_non_mimo_family_keeps_stock_geometry_pass_through(tmp_path):
    """Codex §12.2 regression: upstream family + geometry-mismatched config
    must return None (pass-through), NOT raise — stock upstream behavior."""
    import pytest

    container, model_dir, _ = _mimo_container(tmp_path)
    # rewrite container as deepseek_v41 family
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["family"] = "deepseek_v41"
    (container / "build-contract.json").write_text(json.dumps(contract))
    manifest = json.loads((container / "manifest.json").read_text())
    manifest["family"] = "deepseek_v41"
    (container / "manifest.json").write_text(json.dumps(manifest))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    # MiMo geometry config against a deepseek_v41 container with explicit
    # mxfp4_csf intent: stock resolves any valid container (no geometry gate
    # for non-MiMo families); the MiMo-scoped gate must not raise here.
    resolved = loader._resolve_csf_root(
        {
            "quant_method": "mxfp4_csf",
            "checkpoint_root": str(container),
        },
        _mimo_model_config(container, model_dir).hf_text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "deepseek_v41"


def test_mimo_hybrid_intent_with_csf_root_resolves(tmp_path):
    """Hybrid intent (fp8 + store_dtype mxfp4 + mimo_v2*) + MIMO_CSF_ROOT
    resolves the CSF container (this is the production fast profile)."""
    import os

    container, model_dir, sha = _mimo_container(tmp_path)
    contract = json.loads(
        (container / "build-contract.json").read_text()
    )
    contract["source_index_sha256"] = sha
    (container / "build-contract.json").write_text(json.dumps(contract))

    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    text_config = _mimo_model_config(container, model_dir).hf_text_config
    resolved = loader._resolve_csf_root(
        {
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "checkpoint_root": str(container),
        },
        text_config,
        model_root=str(model_dir),
    )
    assert resolved is not None
    assert resolved[1]["family"] == "mimo_v26_pro_mxfp4"


# ---------------------------------------------------------------------------
# MXFP4/X4T scale prefetch (Codex §9.2/§9.3/§9.5): MiMo-owned orchestration.
# ---------------------------------------------------------------------------


def _stub_method(owner, index, *, backend=None):
    """A Mxfp4CsfMoEMethod with the prefetch fields but no weights.

    ``owner`` is the prefetch STATE (MimoMxfp4CsfScalePrefetch); the owner
    namespace mirrors MimoMxfp4CsfConfig's capability contract.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import Mxfp4CsfMoEMethod

    owner_ns = SimpleNamespace(
        x4t_scale_prefetch=True, scale_prefetch_state=owner
    )
    if owner.scale_stream is None:
        # register() is never called on stubs; give the state a fake side stream.
        owner.scale_stream = _FakeStream()
    method = object.__new__(Mxfp4CsfMoEMethod)
    method.owner = owner_ns
    method.layer_index = index
    method.scale_prefetch_owner = owner
    method.backend = backend if backend is not None else SimpleNamespace(
        x4t_scales_expanded=False
    )
    method.moe_done = SimpleNamespace(record=lambda stream: None)
    method.scales_ready = SimpleNamespace(record=lambda stream: None)
    method.prepared = f"layer-{index}"
    method.moe_kernel = SimpleNamespace()
    return method


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class _FakeStream:
    """Minimal torch.cuda.Stream stand-in (records waits, no CUDA)."""

    def __init__(self, device=None):
        self.waits = []
        self.entered = 0

    def wait_event(self, event):
        self.waits.append(event)

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False


def _arm_stub_cuda(monkeypatch):
    """Replace the CUDA stream/event surface with CPU stubs."""
    main = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: main)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: _FakeStream(*a))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    # torch.cuda.is_current_stream_capturing raises on CPU-only hosts; the
    # production helper's contract is "False when not capturing".
    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.mxfp4_csf._is_current_stream_capturing",
        lambda: False,
    )
    return main


def _step(monkeypatch, *, starts, spec_tokens=3, gdn_prefills=None):
    """One forward-context step with a decodes-first batch (Codex 9.5 shape)."""
    import vllm.config as vllm_config
    import vllm.forward_context as forward_context

    # Production width source: the speculative config (B12xPagedMetadata does
    # not carry num_spec_decode_tokens; KDA/GDN metadata refines it).
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=spec_tokens)
        ),
    )
    metadata = {
        "attn": SimpleNamespace(
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            num_actual_tokens=starts[-1],
            max_query_len=max(b - a for a, b in zip(starts, starts[1:])),
        )
    }
    if gdn_prefills is not None:
        metadata["kda"] = SimpleNamespace(
            num_prefills=gdn_prefills,
            num_spec_decode_tokens=starts[-1],
            max_query_len=spec_tokens,
        )
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    return context


def test_mimo_owner_carries_prefetch_capability():
    """MiMo owner is the ONLY owner that arms (Codex 9.1-3)."""
    from vllm.model_executor.layers.quantization.mxfp4_csf import (
        MimoMxfp4CsfConfig,
    )
    from vllm.models.deepseek_v4_1.mxfp4_csf import DeepseekV41Mxfp4CsfConfig

    assert getattr(MimoMxfp4CsfConfig, "x4t_scale_prefetch", False) is True
    # Non-MiMo owners never set the capability attribute: unarmed even with
    # the env on (blast radius).
    assert not getattr(DeepseekV41Mxfp4CsfConfig, "x4t_scale_prefetch", False)


def test_mxfp4_scale_prefetch_env_defaults_off():
    """VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH defaults OFF (unlike the NVFP4 twin)."""
    import vllm.envs as envs

    assert envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH is False
    # The NVFP4 twin keeps its own default; this diff must not touch it.
    assert envs.VLLM_B12X_CSF_SCALE_PREFETCH is True


def test_row_type_eligibility_rejects_dflash_verification_and_decode(monkeypatch):
    """Codex 9.5: arm only for actual eligible prefill rows.

    DFlash verification batches (uniform 1+spec-token queries) and pure decode
    batches must NOT arm, whatever their row count — a total-row threshold
    would arm for exactly these cases.
    """
    from vllm.models.deepseek_v4_1.mxfp4_csf import _batch_has_prefill_rows

    # Pure decode: 4 requests, uniform_query_len=4 rows each (MTP=3 ->
    # 1 + 3 spec tokens). No prefill rows -> must NOT arm.
    _step(monkeypatch, starts=[0, 4, 8, 12, 16], spec_tokens=3)
    assert _batch_has_prefill_rows() is False

    # DFlash verification: uniform 1+spec-token queries, 8 rows each
    # (DFlash=7 -> uniform_query_len=8). Many rows, but ALL decode-class.
    _step(monkeypatch, starts=[0, 8, 16, 24, 32], spec_tokens=7, gdn_prefills=0)
    assert _batch_has_prefill_rows() is False

    # A real prefill request (query 512) among decodes: arm.
    _step(monkeypatch, starts=[0, 8, 520])
    assert _batch_has_prefill_rows() is True

    # Missing metadata: never arm (fail closed).
    import vllm.forward_context as forward_context

    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    assert _batch_has_prefill_rows() is False


def test_prefetch_off_is_unarmed(monkeypatch):
    """With the env OFF the method runs today's path exactly: no skip flag,
    no side stream, no expansion."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", False)
    state = MimoMxfp4CsfScalePrefetch()
    calls = []
    backend = SimpleNamespace(x4t_scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)
    # env OFF -> the method never registers (scale_prefetch_owner is None).
    method.scale_prefetch_owner = None
    method.moe_kernel = SimpleNamespace(
        apply=lambda **_: calls.append(backend.x4t_scales_expanded) or "out"
    )
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    result = method.apply(layer, torch.zeros(8, 4), None, None, None, None)
    assert result == "out"
    assert calls == [False], "prefetch OFF must never set the skip flag"
    assert state.scale_prefetch is None


def test_stale_generation_is_awaited_and_never_reused(monkeypatch):
    """Codex §9.3 point 3/4: a stale/mismatched generation still needs a wait
    before an inline expansion overwrites the shared scratch, and its scales
    are NEVER consumed."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import (
        MimoMxfp4CsfScalePrefetch,
        _forward_token,
    )

    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    backend = SimpleNamespace(x4t_scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)
    seen = []
    method.moe_kernel = SimpleNamespace(
        apply=lambda **_: seen.append(backend.x4t_scales_expanded)
    )
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = torch.zeros(512, 4)

    # A pending prefetch for a DIFFERENT generation (wrong token, wrong layer).
    stale_event = _FakeEvent()
    state.scale_prefetch = (3, 999999, stale_event)
    _step(monkeypatch, starts=[0, 8, 520])
    method.apply(layer, x, None, None, None, None)

    # The stale event was awaited (scratch race protection) ...
    assert stale_event in main.waits, (
        "a stale prefetch must still be awaited before the inline expansion"
    )
    # ... but NOT consumed as this call's scales.
    assert seen == [False], "stale generation must never be reused"
    assert state.scale_prefetch is None
    # The skip flag is cleared after the call (no leakage into the next one).
    assert backend.x4t_scales_expanded is False

    # A matching generation IS consumed. The token must be created for the
    # SAME forward context the apply runs in (each step gets its own token).
    fresh = _FakeEvent()
    _step(monkeypatch, starts=[0, 8, 520])
    token = _forward_token()
    state.scale_prefetch = (3, token, fresh)
    method.apply(layer, x, None, None, None, None)
    assert seen[-1] is True, "matching (layer, token) must consume the prefetch"
    assert fresh in main.waits
    assert backend.x4t_scales_expanded is False, "flag must be cleared in finally"


def test_cancellation_clears_skip_flag_and_capture_drains(monkeypatch):
    """A failed forward clears the per-call skip flag; capture drains."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    main = _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    backend = SimpleNamespace(x4t_scales_expanded=False)
    method = _stub_method(state, 3, backend=backend)

    def boom(**_):
        raise RuntimeError("forward failed")

    method.moe_kernel = SimpleNamespace(apply=boom)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = torch.zeros(512, 4)
    _step(monkeypatch, starts=[0, 8, 520])
    with pytest.raises(RuntimeError, match="forward failed"):
        method.apply(layer, x, None, None, None, None)
    assert backend.x4t_scales_expanded is False, (
        "the skip flag must be cleared in finally even on a failed forward"
    )

    # Graph transition: capture disables new prefetch AND drains in-flight
    # side-stream writes before captured work can use the shared buffers.
    inflight = _FakeEvent()
    state.scale_prefetch = (3, 1, inflight)
    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.mxfp4_csf._is_current_stream_capturing",
        lambda: True,
    )
    method.moe_kernel = SimpleNamespace(apply=lambda **_: "captured")
    assert method.apply(layer, x, None, None, None, None) == "captured"
    assert inflight in main.waits, (
        "capture must drain prior side-stream writes, not just early-return"
    )
    assert state.scale_prefetch is None


def test_mixed_decode_prefill_arms_for_prefill_rows_only(monkeypatch):
    """Mixed decode/prefill splits: the producer arms only for a call with
    actual prefill rows (row-type eligibility), never by total row count."""
    import vllm.models.deepseek_v4_1.mxfp4_csf as mxfp4_module
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.b12x._w4a16_a4_prefill_enabled",
        lambda: True,
    )
    expanded = []

    def fake_expand(prepared):
        expanded.append(prepared)
        return True

    import b12x.moe.fused_moe as fused_moe

    monkeypatch.setattr(fused_moe, "expand_scales", fake_expand)

    state = MimoMxfp4CsfScalePrefetch()
    layers = {i: _stub_method(state, i) for i in (3, 4)}
    for method in layers.values():
        state.scale_layers[method.layer_index] = method
        method.moe_kernel = SimpleNamespace(apply=lambda **_: None)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    # Mixed batch: 8 decode rows + 512 prefill rows. _num_leading_decode_tokens
    # reports 8 -> the call is prefill-eligible -> arm for layer 4.
    monkeypatch.setattr(
        mxfp4_module, "_num_leading_decode_tokens", lambda tokens: 8
    )
    _step(monkeypatch, starts=[0, 8, 520])
    x = torch.zeros(520, 4)
    for method in layers.values():
        method.apply(layer, x, None, None, None, None)
    assert expanded == [layers[4].prepared], (
        "mixed decode/prefill must arm for the prefill rows' next layer"
    )

    # Pure-decode batch (DFlash verification shape): 32 rows, 0 prefill rows.
    # Even with many rows, no arming (Codex 9.5).
    expanded.clear()
    monkeypatch.setattr(
        mxfp4_module, "_num_leading_decode_tokens", lambda tokens: tokens
    )
    _step(monkeypatch, starts=[0, 8, 16, 24, 32], spec_tokens=7, gdn_prefills=0)
    x = torch.zeros(32, 4)
    for method in layers.values():
        method.apply(layer, x, None, None, None, None)
    assert expanded == [], (
        "DFlash verification / pure decode must never arm the prefetch"
    )


def test_first_and_last_layer_handled(monkeypatch):
    """First layer (nothing pending) and last layer (nothing to arm) are safe."""
    from vllm.models.deepseek_v4_1.mxfp4_csf import MimoMxfp4CsfScalePrefetch

    _arm_stub_cuda(monkeypatch)
    monkeypatch.setattr("vllm.envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH", True)
    state = MimoMxfp4CsfScalePrefetch()
    calls = []
    last = _stub_method(state, 3)
    last.moe_kernel = SimpleNamespace(
        apply=lambda **_: calls.append(last.backend.x4t_scales_expanded)
    )
    state.scale_layers[3] = last
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    _step(monkeypatch, starts=[0, 8, 520])
    # Layer 3 is the last registered layer: no layer 4 to arm, and no pending
    # prefetch (first call of the forward).
    last.apply(layer, torch.zeros(512, 4), None, None, None, None)
    assert calls == [False], "first/last layer must run the inline path"
    assert state.scale_prefetch is None
