# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly compressed DS4.1 routed experts with native dense tensors."""

import itertools

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import (
    B12xExperts,
    _is_current_stream_capturing,
    _num_leading_decode_tokens,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.utils.torch_utils import set_default_torch_num_threads

from .quant_config import DeepseekV41FP8Config

logger = init_logger(__name__)

# Per-forward token counter (shared with nvfp4_csf.py via the same forward
# context attribute; see _forward_token).
_FORWARD_TOKENS = itertools.count()


def _spec_decode_query_width() -> int:
    """Rows per request in a uniform decode/verification batch.

    ``1 + num_speculative_tokens`` (the runner's ``uniform_decode_query_len``):
    MTP=3 gives 4, DFlash=7 gives 8. Queries at or below this width are
    decode or speculative-verification rows, not prefill rows.
    """
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    spec = getattr(config, "speculative_config", None) if config is not None else None
    return 1 + int(getattr(spec, "num_speculative_tokens", 0) or 0)


def _batch_has_prefill_rows() -> bool:
    """Whether the current batch contains actual prefill rows (Codex 9.5).

    Row-type eligibility, never a total-row threshold: a batch qualifies only
    when some request's query exceeds the decode/verification width
    (``1 + num_speculative_tokens``, taken from the speculative config and
    from any metadata that reports its own spec-token count). DFlash
    verification batches (uniform 1+spec-token queries) and pure decode
    batches never qualify, whatever their row count. Host-side metadata only
    (``max_query_len``): no device read, no synchronization. Missing metadata
    means no arming.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    metadata = getattr(get_forward_context(), "attn_metadata", None)
    if metadata is None:
        return False
    values: list = []
    if isinstance(metadata, dict):
        values = list(metadata.values())
    elif isinstance(metadata, (list, tuple)):
        for entry in metadata:
            if isinstance(entry, dict):
                values.extend(entry.values())
    width = _spec_decode_query_width()
    for value in values:
        # Backends that know their own spec-token count (KDA/GDN) refine the
        # config-derived width; B12xPagedMetadata relies on the config alone.
        spec_tokens = getattr(value, "num_spec_decode_tokens", None)
        if isinstance(spec_tokens, int):
            width = max(width, 1 + int(spec_tokens))
    return any(
        isinstance(getattr(value, "max_query_len", None), int)
        and int(value.max_query_len) > width
        for value in values
    )


def _b12x_x4t_prefetch_supported(experts) -> bool:
    """Whether b12x exposes prepared full-X4T expansion + consumer skip.

    Requires both halves of the coordinated contract: ``expand_scales`` must
    handle the paired-program X4T payload (prepared full-X4T expansion) and
    ``bind`` must accept ``x4t_scales_expanded`` (consumer skip). Older b12x
    builds have neither; arming there would duplicate work (a side-stream
    kernel with the inline decoder still running), so prefetch stays off.
    Fails closed on any missing piece.
    """
    if getattr(experts._impl, "x4t_prefetch", None) is None:
        return False
    try:
        import inspect

        from b12x.moe import fused_moe
        from b12x.moe.fused_moe._impl import TPMoEScratchPlan

        if not hasattr(fused_moe, "expand_scales"):
            return False
        return "x4t_scales_expanded" in inspect.signature(
            TPMoEScratchPlan.bind
        ).parameters
    except Exception:
        return False


class MimoMxfp4CsfScalePrefetch:
    """Owner-carried MXFP4/X4T scale-prefetch state for one MiMo expert owner.

    Mirrors ``Nvfp4CsfConfig``'s scale_layers/scale_stream/scale_prefetch +
    moe_done/scales_ready pattern (nvfp4_csf.py), with Codex §9.3 corrections
    for the shared 54 MiB per-rank X4T scratch:

    * The overlap window is layer L+1's ATTENTION: after L's MoE completes
      (``moe_done``), the side stream expands L+1's scales into the shared
      scratch while L+1's attention runs; L+1's MoE consumes them.
    * A pending prefetch is always awaited before the consumer runs OR before
      any inline expansion overwrites the same scratch — a stale/mismatched
      generation still needs that wait (§9.3 point 3/4).
    * The per-call consumer skip flag is set only for the matching
      (layer, forward-token) and cleared in ``finally``.
    * CUDA-graph capture: new prefetch is disabled AND any pending side-stream
      write is drained (waited) before captured work can touch the shared
      buffers — early-return alone is not a sync policy.
    * Eligibility is per-row-type (``_batch_has_prefill_rows``): DFlash
      verification and pure-decode batches never arm (Codex 9.5).

    PP1, no ubatching, no concurrent forwards sharing the owner (enforced by
    the shared method's PP1/no-ubatching requirement). IDs/events/streams are
    allocated at preparation; nothing is allocated per request.
    """

    def __init__(self):
        self.scale_layers: dict[int, "Mxfp4CsfMoEMethod"] = {}
        self.scale_stream: torch.cuda.Stream | None = None
        # (destination layer index, forward token, ready event)
        self.scale_prefetch: tuple[int, int, torch.cuda.Event] | None = None

    def register(self, method: "Mxfp4CsfMoEMethod", device: torch.device) -> None:
        self.scale_layers[method.layer_index] = method
        if self.scale_stream is None:
            self.scale_stream = torch.cuda.Stream(device)

    def drain(self) -> None:
        """Wait out any in-flight side-stream expansion (capture transition)."""
        pending = self.scale_prefetch
        self.scale_prefetch = None
        if pending is not None:
            torch.cuda.current_stream().wait_event(pending[2])


def _forward_token() -> int | None:
    """An id of the current forward pass, or None outside of one.

    Duplicated from nvfp4_csf.py's tiny helper with the same context attribute
    (``_b12x_csf_token``), so the MXFP4 and NVFP4 prefetch mechanisms share one
    per-forward identity. The import is avoided deliberately: nvfp4_csf imports
    the modelopt graph, and this module is imported by the shared CSF config.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    token = getattr(context, "_b12x_csf_token", None)
    if token is None:
        token = next(_FORWARD_TOKENS)
        context._b12x_csf_token = token
    return token


class DeepseekV41Mxfp4CsfConfig(DeepseekV41FP8Config):
    """MXFP4-CSF main experts; unchanged native dense, shared and draft weights."""

    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None)
            in ("deepseek_v41", "deepseek_v41_text")
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        result = cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
        )
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("DS4.1 routed experts require a numbered layer")
            config = get_current_vllm_config()
            if int(match.group(1)) < config.model_config.hf_config.num_hidden_layers:
                if (
                    config.parallel_config.pipeline_parallel_size != 1
                    or config.parallel_config.use_ubatching
                ):
                    raise NotImplementedError(
                        "MXFP4-CSF shared scale scratch requires PP1 without ubatching"
                    )
                if config.load_config.load_format not in ("mxfp4_csf",):
                    raise ValueError("MXFP4-CSF requires --load-format mxfp4_csf ")
                return Mxfp4CsfMoEMethod(
                    layer.moe_config,
                    self,
                    activation_mode="a16" if envs.VLLM_B12X_MOE_FP4_FORCE_A16 else "a8",
                )
        return super().get_quant_method(layer, prefix)


class Mxfp4CsfMoEMethod(FusedMoEMethodBase):
    """Expand scales for routed experts into serialized, model-owned scratch."""

    def __init__(self, moe, owner, *, activation_mode="a16"):
        super().__init__(moe)
        self.owner = owner
        if activation_mode not in ("a16", "a8"):
            raise ValueError("MXFP4-CSF requires A16 or MXFP8 activations")
        self.activation_mode = activation_mode
        # Set at weight-load time when the owner arms the X4T scale prefetch.
        self.scale_prefetch_owner = None
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError("MXFP4-CSF experts support TP without EP/DP")
        if (
            moe.activation not in (MoEActivation.SILU, MoEActivation.SITU)
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
            or (
                moe.activation == MoEActivation.SITU
                and (
                    moe.activation_situ_beta != 4.0
                    or moe.activation_situ_linear_beta != 25.0
                )
            )
        ):
            raise ValueError(
                "MXFP4-CSF requires bias-free BF16 SwiGLU or SiTU(4,25) experts"
            )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None or params_dtype != torch.bfloat16:
            raise ValueError("MXFP4-CSF requires a numbered BF16 expert layer")
        self.layer_index = int(match.group(1))
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.local_intermediate = intermediate_size_per_partition
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        activation = "mxfp8" if self.activation_mode == "a8" else None
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(dtype=activation),
            _a2=FusedMoEQuantDesc(dtype=activation),
            _w1=FusedMoEQuantDesc(dtype="mxfp4"),
            _w2=FusedMoEQuantDesc(dtype="mxfp4"),
        )

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe

        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            read_mxfp4_csf_layer,
        )

        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        device = layer.w13_weight.device
        e, h, n = self.num_experts, self.hidden_size, self.local_intermediate
        shapes = ((e, h // 32, 2 * n), (e, n // 32, h))
        if self.owner.scale_scratch is None:
            # MXFP4-CSF disallows ubatching: every layer consumes these scale grids
            # before its successor may overwrite them on the same stream.
            self.owner.scale_scratch = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )
        scratch = self.owner.scale_scratch
        if any(
            t.shape != shape or t.device != device for t, shape in zip(scratch, shapes)
        ):
            raise ValueError("MXFP4-CSF shared scale scratch geometry/device mismatch")
        # Per-expert CPU slices are too small to amortize intra-op barriers.
        # Restore the serving thread policy before kernel preparation.
        with set_default_torch_num_threads(1):
            weights = read_mxfp4_csf_layer(
                self.owner.checkpoint_root,
                self.layer_index,
                num_experts=e,
                hidden_size=h,
                intermediate_size=n * tp,
                tp_rank=rank,
                tp_size=tp,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
            )
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=fused_moe.ActivationSpec(
                mode=self.activation_mode,
                nonlinearity="situ"
                if self.moe.activation == MoEActivation.SITU
                else "silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=self.moe.swiglu_limit,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=e, hidden_size=h, intermediate_size=n
            ),
        )
        prepared = fused_moe.prepare_weights(plan=plan, weights=weights)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )
        self.backend, self.prepared = backend, prepared
        # MiMo-owned X4T scale prefetch (Codex §9.2/§9.3): default OFF, owner
        # capability attribute only (never an isinstance check against the
        # MiMo config, which would be a circular import for non-MiMo owners).
        self.scale_prefetch_owner = None
        self.moe_done = torch.cuda.Event()
        self.scales_ready = torch.cuda.Event()
        if (
            getattr(self.owner, "x4t_scale_prefetch", False)
            and self.activation_mode == "a16"
            and envs.VLLM_B12X_MXFP4_CSF_SCALE_PREFETCH
            and _b12x_x4t_prefetch_supported(prepared)
        ):
            state = getattr(self.owner, "scale_prefetch_state", None)
            if state is not None:
                self.scale_prefetch_owner = state
                state.register(self, device)
        logger.info(
            "MXFP4-CSF lossless MXFP4 layer %d rank %d/%d: %s activations, "
            "compressed scales, shared scratch %d bytes, x4t-scale-prefetch=%s",
            self.layer_index,
            rank,
            tp,
            "MXFP8" if self.activation_mode == "a8" else "BF16",
            sum(t.numel() for t in scratch),
            self.scale_prefetch_owner is not None,
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        workspace=None,
    ):
        assert self.moe_kernel is not None

        def run():
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
                workspace=workspace,
            )

        owner = self.scale_prefetch_owner
        if owner is None:
            return run()
        stream = torch.cuda.current_stream()
        if _is_current_stream_capturing():
            # Codex §9.3 point 4: during CUDA-graph capture disable new
            # prefetch AND drain prior side-stream writes before the graph can
            # use the shared buffers. Early-return alone is not a sync policy.
            owner.drain()
            return run()
        token = _forward_token()
        # Codex §9.3 point 3: consume-or-drop the pending prefetch, but always
        # WAIT it before this call: whether or not the generation matches, the
        # side-stream expansion may be in flight over the same 54 MiB scratch
        # this call's inline expansion would overwrite.
        pending, owner.scale_prefetch = owner.scale_prefetch, None
        if pending is not None:
            stream.wait_event(pending[2])
        self.backend.x4t_scales_expanded = (
            token is not None
            and pending is not None
            and (pending[:2] == (self.layer_index, token))
        )
        try:
            result = run()
        finally:
            self.backend.x4t_scales_expanded = False
        following = owner.scale_layers.get(self.layer_index + 1)
        if (
            token is not None
            and following is not None
            and self._expands(x)
            and _batch_has_prefill_rows()
        ):
            # The next layer's MoE would expand its scales into the same
            # scratch: do it now on a side stream, overlapping that layer's
            # attention (NOT this layer's: it is already done).
            from b12x.moe import fused_moe as b12x_fused_moe

            self.moe_done.record(stream)
            side = owner.scale_stream
            side.wait_event(self.moe_done)
            with torch.cuda.stream(side):
                b12x_fused_moe.expand_scales(following.prepared)
                following.scales_ready.record(side)
            owner.scale_prefetch = (
                following.layer_index,
                token,
                following.scales_ready,
            )
        return result

    def _expands(self, x: torch.Tensor) -> bool:
        """Whether this call's consumers expand X4T scales inline.

        Codex §9.5: arm only for actual eligible prefill rows, never a
        total-row threshold — DFlash verification batches and mixed
        decode/prefill splits carry many decode rows and must not arm for
        them. The consumer's skip is per whole call (the X4T runners skip
        their inline decode for the entire call), so the producer only arms
        when the call is row-type eligible (checked by the caller).
        """
        from vllm.model_executor.layers.fused_moe.b12x import (
            _w4a16_a4_prefill_enabled,
        )

        tokens = int(x.shape[0])
        if _w4a16_a4_prefill_enabled():
            return _num_leading_decode_tokens(tokens) < tokens
        return False
