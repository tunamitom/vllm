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
