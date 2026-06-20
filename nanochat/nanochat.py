"""Training module.


Exposes:
  train_model(tokenizer, make_train_dataloader, max_seq_len, time_budget,
              device) -> (model, info_dict)

"""

import gc
import math
import time
import queue as _queue
import threading as _threading
from dataclasses import dataclass, asdict

import torch


class _Prefetcher:
    """Overlap the evaluator dataloader's heavy pure-Python best-fit packing with
    GPU compute. The evaluator's make_dataloader couples (a) Python packing, (b) an
    H2D copy into a SINGLE reused gpu_buffer, and (c) yielding aliased views of that
    buffer. We run next() on ONE background thread, so every CUDA enqueue it issues
    (the H2D copy, then a clone) is in program order on the default stream: clone(N)
    is enqueued before the next H2D(N+1) that overwrites gpu_buffer, so the clone
    always reads valid data. Each batch is cloned into a DISTINCT tensor (no buffer
    reuse race -- this is what broke the windowed-timing attempt), handed off via a
    bounded Queue. The packing (GIL-bound Python) runs while the main thread is
    blocked in the per-step .item() CUDA sync (GIL released), hiding ~the host cost.
    """

    def __init__(self, loader, depth=4):
        self._q = _queue.Queue(maxsize=depth)
        self._done = object()
        self._t = _threading.Thread(target=self._run, args=(loader,), daemon=True)
        self._t.start()

    def _run(self, loader):
        try:
            for x, y, ep in loader:
                self._q.put((x.clone(), y.clone(), ep))
        except BaseException as e:  # surface dataloader errors on the main thread
            self._q.put(e)
        else:
            self._q.put(self._done)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if isinstance(item, BaseException):
            raise item
        if item is self._done:
            raise StopIteration
        return item
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# The per-parameter fused optimizer steps (adamw/rmsprop/muon) are each
# torch.compile(dynamic=False, fullgraph=True), so they specialize one cached
# graph per distinct (shape, dtype). With n_layer=8 the AdamW group alone routes
# 9 distinct shapes (lm_head fp32 [V,d]; wte/value bf16 [V,d]; scalar vecs [8],
# [7], [3], [2]; gate matrices [1,d], [3,d]; output_gate bias [1]) which exceeds
# PyTorch's default recompile_limit of 8 -> FailOnRecompileLimitHit at step 0.
# These are tiny elementwise kernels and compile only during warmup (steps <=10,
# excluded from the training-time budget).
torch._dynamo.config.recompile_limit = 128
if hasattr(torch._dynamo.config, "cache_size_limit"):
    torch._dynamo.config.cache_size_limit = 128

# Coordinate-descent autotuning searches block-size/num-warps/etc and speeds up
# the memory-bound pointwise/reduction kernels. It replays
# mathematically-identical kernels (different fp reduction tiling stays within
# bf16 tol). The only cost is extra warmup-compile time, outside the training
# budget.
try:
    import torch._inductor.config as _inductor_config
    _inductor_config.coordinate_descent_tuning = True
except Exception as _e:
    print(f"coordinate_descent_tuning unavailable ({_e}); using max-autotune defaults")

if torch.cuda.get_device_capability() >= (10, 0):
    _SDPA_BACKEND_ORDER = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
else:
    _SDPA_BACKEND_ORDER = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
_SDPA_BACKEND_NAME = {
    SDPBackend.CUDNN_ATTENTION: "cudnn",
    SDPBackend.FLASH_ATTENTION: "flash",
    SDPBackend.EFFICIENT_ATTENTION: "efficient",
    SDPBackend.MATH: "math",
}


def _sdpa_kernel_ctx(backend):
    try:
        return sdpa_kernel([backend], set_priority=True)
    except TypeError:
        try:
            return sdpa_kernel([backend], set_priority_order=True)
        except TypeError:
            return sdpa_kernel([backend])

_cap = torch.cuda.get_device_capability()
# FlashAttention-3 is Hopper-only (sm90).
_USE_FA3 = _cap == (9, 0)
if _USE_FA3:
    from kernels import get_kernel

    fa3 = get_kernel("varunneal/flash-attention-3").flash_attn_interface

# FlashAttention-4 (CuTe-DSL, Blackwell sm100-native). The CuTe forward and
# backward entry points are plain kernels with no autograd hook of their own.
# We make a single differentiable Python callable out of them by registering
# the forward as a torch.library custom op, giving it a shape-only meta rule,
# and stitching the backward (itself a second custom op) onto it through
# register_autograd. Because the forward op IS the autograd node, Dynamo
# captures each attention call -- forward and backward together -- as one
# opaque operator, with no per-layer graph break to throw away the speedup.
_USE_FA4 = _cap[0] >= 10 and not _USE_FA3
if _USE_FA4:
    from flash_attn.cute import flash_attn_func as _cute_attn_fwd
    from flash_attn.cute.interface import _flash_attn_bwd as _cute_attn_bwd

    # A custom-op schema cannot express an Optional[tuple] window, so we collapse
    # the sliding window down to one integer `edge`. Zero is the sentinel for
    # "no window -> strict causal"; any positive value is the left edge. The
    # right edge is fixed at 0. The forward kernel wants a (left, 0)/(None, None)
    # tuple and the backward wants a bare left/None, both rebuilt per call.
    _NO_WINDOW = 0

    def _fa4_left_edge(window_size, seq_len):
        edge = window_size[0] if isinstance(window_size, tuple) else window_size
        # Unset, non-positive, or sequence-spanning -> behaves as full causal.
        if edge is None or edge <= 0 or edge >= seq_len:
            return _NO_WINDOW
        return int(edge)

    @torch.library.custom_op("fa4cute::attend", mutates_args=())
    def _fa4_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    edge: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (edge, 0) if edge > _NO_WINDOW else (None, None)
        out, lse = _cute_attn_fwd(
            q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_attend.register_fake
    def _fa4_attend_meta(q, k, v, edge):
        B, T, H, _D = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
        return out, lse

    @torch.library.custom_op("fa4cute::attend_grad", mutates_args=())
    def _fa4_attend_grad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         out: torch.Tensor, dout: torch.Tensor,
                         lse: torch.Tensor,
                         edge: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = edge if edge > _NO_WINDOW else None
        dq, dk, dv = _cute_attn_bwd(
            q, k, v, out, dout, lse,
            causal=True, window_size_left=left, window_size_right=0)
        return dq, dk, dv

    @_fa4_attend_grad.register_fake
    def _fa4_attend_grad_meta(q, k, v, out, dout, lse, edge):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_setup(ctx, inputs, output):
        q, k, v, edge = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.edge = edge

    def _fa4_vjp(ctx, dout, _dlse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4cute.attend_grad(
            q, k, v, out, dout, lse, ctx.edge)
        return dq, dk, dv, None

    _fa4_attend.register_autograd(_fa4_vjp, setup_context=_fa4_setup)

    def fa4_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        edge = _fa4_left_edge(window_size, q.shape[1])
        out, _lse = torch.ops.fa4cute.attend(q, k, v, edge)
        return out

    print(f"Using flash-attn-4 as custom op (GPU capability {_cap})")

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    mlp_ratio: int = 8  # widen MLP
    window_pattern: str = "SSSL"
    ngram_vocab: int = 1048576
    trigram_vocab: int = 1048576
    fourgram_vocab: int = 262144
    ve_ngram_vocab: int = 1048576
    ve_trigram_vocab: int = 1048576


# ---------------------------------------------------------------------------
_USE_FUSED_QKNORM = _USE_FA4
if _USE_FUSED_QKNORM:
    try:
        import triton
        import triton.language as tl

        _QKN_EPS = 1e-6
        # Autotune the row-blocking/occupancy. BLOCK only groups independent
        # (b,t,h) rows, so every config produces bit-identical output.
        # key=['N']: N is fixed across steps, so this benchmarks once during warmup
        # (outside the budget). The config set is restricted to combinations that
        # are guaranteed to fit in registers (<=16 elems/thread/buffer: BLOCK*HALF /
        # (num_warps*32) <= 16, HALF=64) so every config compiles+runs -- it does NOT
        # depend on the autotuner pruning an OutOfResources config (that would surface
        # at the runtime kernel call, outside the import-time eager fallback guard).
        _qkn_configs = [
            triton.Config({'BLOCK': 16}, num_warps=4),
            triton.Config({'BLOCK': 16}, num_warps=8),
            triton.Config({'BLOCK': 32}, num_warps=4),
            triton.Config({'BLOCK': 32}, num_warps=8),
            triton.Config({'BLOCK': 64}, num_warps=8),
        ]

        @triton.autotune(configs=_qkn_configs, key=['N'])
        @triton.jit
        def _rope_norm_fwd_kernel(X, COS, SIN, Y, N, H, Tseq, eps,
                                  D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            mask = rows < N
            safe = tl.where(mask, rows, 0)
            t = (safe // H) % Tseq
            cols = tl.arange(0, HALF)
            xb = safe[:, None] * D
            cb = t[:, None] * HALF
            o1 = xb + cols[None, :]
            o2 = xb + HALF + cols[None, :]
            co = cb + cols[None, :]
            m2 = mask[:, None]
            x1 = tl.load(X + o1, mask=m2, other=0.0).to(tl.float32)
            x2 = tl.load(X + o2, mask=m2, other=0.0).to(tl.float32)
            cos = tl.load(COS + co, mask=m2, other=0.0).to(tl.float32)
            sin = tl.load(SIN + co, mask=m2, other=0.0).to(tl.float32)
            r1 = x1 * cos + x2 * sin
            r2 = -x1 * sin + x2 * cos
            ms = (tl.sum(r1 * r1, axis=1) + tl.sum(r2 * r2, axis=1)) / D
            inv = 1.0 / tl.sqrt(ms + eps)
            tl.store(Y + o1, (r1 * inv[:, None]).to(Y.dtype.element_ty), mask=m2)
            tl.store(Y + o2, (r2 * inv[:, None]).to(Y.dtype.element_ty), mask=m2)

        @triton.autotune(configs=_qkn_configs, key=['N'])
        @triton.jit
        def _rope_norm_bwd_kernel(X, COS, SIN, GY, DX, N, H, Tseq, eps,
                                  D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            mask = rows < N
            safe = tl.where(mask, rows, 0)
            t = (safe // H) % Tseq
            cols = tl.arange(0, HALF)
            xb = safe[:, None] * D
            cb = t[:, None] * HALF
            o1 = xb + cols[None, :]
            o2 = xb + HALF + cols[None, :]
            co = cb + cols[None, :]
            m2 = mask[:, None]
            x1 = tl.load(X + o1, mask=m2, other=0.0).to(tl.float32)
            x2 = tl.load(X + o2, mask=m2, other=0.0).to(tl.float32)
            cos = tl.load(COS + co, mask=m2, other=0.0).to(tl.float32)
            sin = tl.load(SIN + co, mask=m2, other=0.0).to(tl.float32)
            gy1 = tl.load(GY + o1, mask=m2, other=0.0).to(tl.float32)
            gy2 = tl.load(GY + o2, mask=m2, other=0.0).to(tl.float32)
            r1 = x1 * cos + x2 * sin
            r2 = -x1 * sin + x2 * cos
            ms = (tl.sum(r1 * r1, axis=1) + tl.sum(r2 * r2, axis=1)) / D
            inv = 1.0 / tl.sqrt(ms + eps)
            n1 = r1 * inv[:, None]
            n2 = r2 * inv[:, None]
            # rms-norm backward (no weight): dr = inv * (gy - n * mean(gy*n))
            dot = (tl.sum(gy1 * n1, axis=1) + tl.sum(gy2 * n2, axis=1)) / D
            dr1 = inv[:, None] * (gy1 - n1 * dot[:, None])
            dr2 = inv[:, None] * (gy2 - n2 * dot[:, None])
            # RoPE^T (rotate by -theta): transpose of [[cos, sin], [-sin, cos]]
            dx1 = cos * dr1 - sin * dr2
            dx2 = sin * dr1 + cos * dr2
            tl.store(DX + o1, dx1.to(DX.dtype.element_ty), mask=m2)
            tl.store(DX + o2, dx2.to(DX.dtype.element_ty), mask=m2)

        @torch.library.custom_op("qknorm::rope_norm", mutates_args=())
        def _rope_norm_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            B, Tseq, H, D = x.shape
            half = D // 2
            N = B * Tseq * H
            y = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
            _rope_norm_fwd_kernel[grid](x, cos, sin, y, N, H, Tseq, _QKN_EPS,
                                        D=D, HALF=half)
            return y

        @_rope_norm_op.register_fake
        def _(x, cos, sin):
            return torch.empty_like(x)

        @torch.library.custom_op("qknorm::rope_norm_bwd", mutates_args=())
        def _rope_norm_bwd_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              gy: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            gy = gy.contiguous()
            B, Tseq, H, D = x.shape
            half = D // 2
            N = B * Tseq * H
            dx = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
            _rope_norm_bwd_kernel[grid](x, cos, sin, gy, dx, N, H, Tseq, _QKN_EPS,
                                        D=D, HALF=half)
            return dx

        @_rope_norm_bwd_op.register_fake
        def _(x, cos, sin, gy):
            return torch.empty_like(x)

        def _rope_norm_setup(ctx, inputs, output):
            x, cos, sin = inputs
            ctx.save_for_backward(x, cos, sin)

        def _rope_norm_backward(ctx, grad_y):
            x, cos, sin = ctx.saved_tensors
            dx = torch.ops.qknorm.rope_norm_bwd(x, cos, sin, grad_y)
            return dx, None, None

        _rope_norm_op.register_autograd(_rope_norm_backward, setup_context=_rope_norm_setup)

        # q-only variant that FOLDS the per-layer attention temperature into the
        # normalization epilogue. y = temp * norm(rope(x)); folding temp into inv (=
        # temp/sqrt(ms+eps)) is a single extra scalar multiply -- it removes the separate
        # eager `q*attn_temp` [B,T,H,D] read+write (+ its backward) entirely. Backward is
        # exact: grad into norm(rope(x)) is temp*grad_y -> reuse the unscaled rope_norm_bwd
        # on (temp*grad_y) for grad_x; grad_temp = sum(grad_y*y)/temp (eager). Bit-identical
        # within bf16 tol to the eager rope_norm-then-multiply path.
        @triton.autotune(configs=_qkn_configs, key=['N'])
        @triton.jit
        def _rope_norm_scaled_fwd_kernel(X, COS, SIN, TEMP, Y, N, H, Tseq, eps,
                                         D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            mask = rows < N
            safe = tl.where(mask, rows, 0)
            t = (safe // H) % Tseq
            cols = tl.arange(0, HALF)
            xb = safe[:, None] * D
            cb = t[:, None] * HALF
            o1 = xb + cols[None, :]
            o2 = xb + HALF + cols[None, :]
            co = cb + cols[None, :]
            m2 = mask[:, None]
            temp = tl.load(TEMP)
            x1 = tl.load(X + o1, mask=m2, other=0.0).to(tl.float32)
            x2 = tl.load(X + o2, mask=m2, other=0.0).to(tl.float32)
            cos = tl.load(COS + co, mask=m2, other=0.0).to(tl.float32)
            sin = tl.load(SIN + co, mask=m2, other=0.0).to(tl.float32)
            r1 = x1 * cos + x2 * sin
            r2 = -x1 * sin + x2 * cos
            ms = (tl.sum(r1 * r1, axis=1) + tl.sum(r2 * r2, axis=1)) / D
            inv = temp / tl.sqrt(ms + eps)  # fold temperature into the rms-norm scale
            tl.store(Y + o1, (r1 * inv[:, None]).to(Y.dtype.element_ty), mask=m2)
            tl.store(Y + o2, (r2 * inv[:, None]).to(Y.dtype.element_ty), mask=m2)

        @torch.library.custom_op("qknorm::rope_norm_scaled", mutates_args=())
        def _rope_norm_scaled_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                                 temp: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            B, Tseq, H, D = x.shape
            half = D // 2
            N = B * Tseq * H
            y = torch.empty_like(x)
            tflat = temp.reshape(1).to(torch.float32)
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
            _rope_norm_scaled_fwd_kernel[grid](x, cos, sin, tflat, y, N, H, Tseq, _QKN_EPS,
                                               D=D, HALF=half)
            return y

        @_rope_norm_scaled_op.register_fake
        def _(x, cos, sin, temp):
            return torch.empty_like(x)

        def _rope_norm_scaled_setup(ctx, inputs, output):
            x, cos, sin, temp = inputs
            ctx.save_for_backward(x, cos, sin, temp, output)

        def _rope_norm_scaled_backward(ctx, grad_y):
            x, cos, sin, temp, y = ctx.saved_tensors
            gy = grad_y.contiguous()
            tf = temp.to(torch.float32)
            grad_x = torch.ops.qknorm.rope_norm_bwd(x, cos, sin, gy * temp.to(gy.dtype))
            grad_temp = (gy.to(torch.float32) * y.to(torch.float32)).sum() / tf.clamp_min(1e-8)
            grad_temp = grad_temp.reshape(temp.shape).to(temp.dtype)
            return grad_x, None, None, grad_temp

        _rope_norm_scaled_op.register_autograd(_rope_norm_scaled_backward,
                                               setup_context=_rope_norm_scaled_setup)

        def _fused_rope_norm(x, cos, sin):
            half = x.shape[-1] // 2
            return torch.ops.qknorm.rope_norm(x, cos.reshape(-1, half), sin.reshape(-1, half))

        def _fused_rope_norm_scaled(x, cos, sin, temp):
            half = x.shape[-1] // 2
            return torch.ops.qknorm.rope_norm_scaled(
                x, cos.reshape(-1, half), sin.reshape(-1, half), temp)

        # FUSED PER-HEAD ATTENTION-OUTPUT EPILOGUE. One Triton op for the whole
        # epilogue: out = g * y / sqrt(mean(y^2 over head_dim) + eps), i.e. an fp32 RMS
        # magnitude-decoupling over the head dimension TIMES a per-(b,t,h) scalar gain g
        # (= exp(tanh(W_route x)), supplied pre-computed). Eager did this in >=2 passes
        # over the [B,T,H,D] output (reduce, then two broadcast multiplies); this kernel
        # is a single load+reduce+store pass -> ~half the HBM traffic on the biggest
        # epilogue tensor. Bit-equivalent within bf16 tol (reduction in fp32). Backward
        # is exact: with r=rsqrt(ms+eps), n=y*r, out=g*n ->
        #   grad_y = g*r*(grad_out - n*mean_D(n*grad_out)),  grad_g = sum_D(grad_out*n).
        # Config set sized for D=head_dim=128 so every config keeps <=16 elems/thread/
        # buffer (BLOCK*D/(num_warps*32) <= 16) -> all compile+run; autotune keys on N
        # (fixed) so it benchmarks once during warmup (outside the budget). g is a
        # per-row scalar so every config is bit-identical.
        _epi_configs = [
            triton.Config({'BLOCK': 8}, num_warps=4),
            triton.Config({'BLOCK': 16}, num_warps=4),
            triton.Config({'BLOCK': 16}, num_warps=8),
            triton.Config({'BLOCK': 32}, num_warps=8),
        ]

        @triton.autotune(configs=_epi_configs, key=['N'])
        @triton.jit
        def _attn_epi_fwd_kernel(X, G, Y, N, eps, D: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            mask = rows < N
            safe = tl.where(mask, rows, 0)
            cols = tl.arange(0, D)
            off = safe[:, None] * D + cols[None, :]
            m2 = mask[:, None]
            x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
            g = tl.load(G + safe, mask=mask, other=0.0).to(tl.float32)
            ms = tl.sum(x * x, axis=1) / D
            inv = g / tl.sqrt(ms + eps)
            tl.store(Y + off, (x * inv[:, None]).to(Y.dtype.element_ty), mask=m2)

        @triton.autotune(configs=_epi_configs, key=['N'])
        @triton.jit
        def _attn_epi_bwd_kernel(X, G, GY, DX, DG, N, eps,
                                 D: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            mask = rows < N
            safe = tl.where(mask, rows, 0)
            cols = tl.arange(0, D)
            off = safe[:, None] * D + cols[None, :]
            m2 = mask[:, None]
            x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
            g = tl.load(G + safe, mask=mask, other=0.0).to(tl.float32)
            gy = tl.load(GY + off, mask=m2, other=0.0).to(tl.float32)
            ms = tl.sum(x * x, axis=1) / D
            r = 1.0 / tl.sqrt(ms + eps)
            n = x * r[:, None]
            s = tl.sum(gy * n, axis=1)          # grad_g (per row) = sum_D(grad_out * n)
            dot = s / D                          # mean_D(n * grad_out)
            dx = g[:, None] * r[:, None] * (gy - n * dot[:, None])
            tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=m2)
            tl.store(DG + safe, s.to(DG.dtype.element_ty), mask=mask)

        @torch.library.custom_op("qknorm::attn_epi", mutates_args=())
        def _attn_epi_op(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            y = y.contiguous()
            B, Tseq, H, D = y.shape
            N = B * Tseq * H
            out = torch.empty_like(y)
            gflat = g.contiguous().reshape(N)
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
            _attn_epi_fwd_kernel[grid](y, gflat, out, N, _QKN_EPS, D=D)
            return out

        @_attn_epi_op.register_fake
        def _(y, g):
            return torch.empty_like(y)

        @torch.library.custom_op("qknorm::attn_epi_bwd", mutates_args=())
        def _attn_epi_bwd_op(y: torch.Tensor, g: torch.Tensor,
                             gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = y.contiguous()
            gy = gy.contiguous()
            B, Tseq, H, D = y.shape
            N = B * Tseq * H
            dx = torch.empty_like(y)
            dg = torch.empty(N, dtype=torch.float32, device=y.device)
            gflat = g.contiguous().reshape(N)
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
            _attn_epi_bwd_kernel[grid](y, gflat, gy, dx, dg, N, _QKN_EPS, D=D)
            return dx, dg.reshape(B, Tseq, H).to(g.dtype)

        @_attn_epi_bwd_op.register_fake
        def _(y, g, gy):
            return torch.empty_like(y), torch.empty_like(g)

        def _attn_epi_setup(ctx, inputs, output):
            y, g = inputs
            ctx.save_for_backward(y, g)

        def _attn_epi_backward(ctx, grad_out):
            y, g = ctx.saved_tensors
            dx, dg = torch.ops.qknorm.attn_epi_bwd(y, g, grad_out)
            return dx, dg

        _attn_epi_op.register_autograd(_attn_epi_backward, setup_context=_attn_epi_setup)

        def _fused_attn_epi(y, g):
            return torch.ops.qknorm.attn_epi(y, g)

        print("Using fused RoPE+QK-RMSNorm Triton custom op")
    except Exception as _e:  # pragma: no cover - fall back to eager rope+norm
        print(f"fused RoPE+QK-norm unavailable ({_e}); using eager path")
        _USE_FUSED_QKNORM = False

# fused attention-output epilogue is available iff the Triton custom ops above
# defined successfully (same guard as the rope_norm op). False -> eager else-branch.
_USE_FUSED_ATTN_EPI = _USE_FUSED_QKNORM


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        # Per-head input-conditioned output gain. A full-width
        # projection of the (pre-normed) block input to one logit per head; zero-init
        # so the gain is exactly 1 at init (g_h = exp(tanh(0)) = 1, see forward). The
        # bounded log-domain gate symmetrically up/down-weights heads per token.
        self.head_route = nn.Linear(self.n_embd, self.n_head, bias=False)
        self.sdpa_backend: SDPBackend | None = None

    def _run_sdpa(self, q_t, k_t, v_t, backend):
        with _sdpa_kernel_ctx(backend):
            return F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                is_causal=True,
            )

    def _run_sdpa_with_fallback(self, q_t, k_t, v_t):
        last_error = None
        if self.sdpa_backend is not None:
            try:
                return self._run_sdpa(q_t, k_t, v_t, self.sdpa_backend)
            except RuntimeError as error:
                print(f"sdpa backend '{_SDPA_BACKEND_NAME[self.sdpa_backend]}' failed; retrying: {error}")
                self.sdpa_backend = None
                last_error = error

        for backend in _SDPA_BACKEND_ORDER:
            try:
                y = self._run_sdpa(q_t, k_t, v_t, backend)
                self.sdpa_backend = backend
                if backend != SDPBackend.CUDNN_ATTENTION:
                    print(f"sdpa selected backend: {_SDPA_BACKEND_NAME[backend]}")
                return y
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(f"All SDPA backends failed, last error: {last_error}")

    def forward(self, x, ve, cos_sin, window_size, block_mask, attn_temp):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        # Standardize each head's q/k to unit RMS BEFORE
        # injecting position, so the rotary phase is applied to a magnitude-
        # normalized feature rather than the raw projection.
        # Uses the fused RoPE+QK-RMSNorm Triton custom op. RoPE is
        # norm-preserving, so the kernel's rope-then-RMS is numerically
        # equivalent to norm-then-rope; the scaled variant folds the per-layer
        # attn_temp multiply on q. The eager else-branch is the identical math
        # (used if Triton import/compile fails).
        if _USE_FUSED_QKNORM:
            k = _fused_rope_norm(k, cos, sin)
            q = _fused_rope_norm_scaled(q, cos, sin, attn_temp)
        else:
            q, k = norm(q), norm(k)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q = q * attn_temp.to(q.dtype)

        if _USE_FA4:
            # Per-layer sliding window via FA4 window_left (fwd+bwd). The window
            # pattern (S=half context on 6/8 layers, L=full on the rest + last)
            # reduces attention FLOPs.
            y = fa4_flash_attn_func(q, k, v, causal=True, window_size=window_size)
        elif _USE_FA3:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            y = self._run_sdpa_with_fallback(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        # Per-head output regulation (two parts):
        #  (2) MAGNITUDE DECOUPLING -- divide each head's attention output by its own
        #      RMS (computed in fp32), so a head's contribution to c_proj no longer
        #      rides on the softmax/value scale; every head enters the projection at a
        #      common, well-conditioned magnitude.
        #  (1) INPUT-CONDITIONED PER-HEAD GAIN -- multiply by g_h = exp(tanh(W_route x)),
        #      a BOUNDED LOG-DOMAIN gate: range ~[1/e, e] = [0.37, 2.72], centered at 1,
        #      symmetric in log-space so a head can be SUPPRESSED toward ~1/e or BOOSTED
        #      toward ~e per token (true up/down-weighting "soft per-head MoE"). W_route
        #      zero-init so g_h == exp(tanh(0)) == 1 at init.
        #      c_proj zero-init -> block is no-op at init.
        head_gain = torch.exp(torch.tanh(self.head_route(x)))
        if _USE_FUSED_ATTN_EPI:
            # Single-pass Triton op: out = head_gain * y / sqrt(mean(y^2 over D)+eps).
            # Bit-equivalent (fp32 reduction) to the eager else-branch, ~half the HBM
            # traffic on the [B,T,H,D] attn output. head_gain stays an eager input so
            # autograd routes grad back through exp(tanh(head_route(x))) unchanged.
            y = _fused_attn_epi(y, head_gain)
        else:
            inv_rms = y.float().square().mean(dim=-1, keepdim=True).add(1e-6).rsqrt().to(y.dtype)
            y = y * inv_rms * head_gain.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, block_mask, attn_temp):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, block_mask, attn_temp)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.bigram_wte = nn.Embedding(config.ngram_vocab, config.n_embd)
        self.trigram_wte = nn.Embedding(config.trigram_vocab, config.n_embd)
        self.fourgram_wte = nn.Embedding(config.fourgram_vocab, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # OUTPUT BIGRAM LOGIT TABLE.
        # A direct vocab->vocab table: output_bigram(idx) adds a learned per-current-
        # token contribution to the NEXT-token logits, i.e. an explicit bigram channel
        # that lives in logit space rather than being routed through the residual
        # stream + lm_head. A per-token sigmoid gate (output_gate) modulates how much
        # of this lexical prior is applied. Zero-init table + gate bias -2
        # (sigmoid~0.12) -> EXACT no-op at init. The table is a SPARSE-gradient lookup
        # (only observed current-tokens get a gradient each step) -> trained with the
        # RMSProp sparse-table group, excluded from FLOPs/weight-averaging like the
        # other hashed n-gram tables. Kept bf16; the contribution is added to the
        # bf16 lm_head logits BEFORE the single float() upcast so no second full
        # [B,T,V] float tensor is materialized (activation-memory neutral).
        self.output_bigram = nn.Embedding(config.vocab_size, config.vocab_size)
        self.output_gate = nn.Linear(config.n_embd, 1, bias=True)
        # OUTPUT N-GRAM BACKOFF FOR FREE (re-uses the input n-gram embeddings).
        # output_bigram conditions the next-token logits on the current token only;
        # to extend the same lexical-prior idea one/two orders deeper we do NOT add new
        # tables. The input n-gram embeddings bigram_x=(current,prev), trigram_x=
        # (current,prev,prev2), fourgram_x=(current,prev,prev2,prev3) are ALREADY
        # gathered each forward for the input mix; they encode exactly the higher-order
        # context that should bias the NEXT token. We add them back -- per-token gated --
        # to the final normed representation just before the SHARED lm_head, giving the
        # lexical signal a direct, uncorrupted shortcut to the logits (it otherwise only
        # reaches the output through the whole transformer stack). No new gather/table:
        # one Linear(d->3) routing gate + three scalar magnitudes; zero-init -> EXACT
        # no-op at init. The same hashed tables now serve double duty (input feature +
        # next-token output prior), so they receive a direct next-token-predictive
        # gradient. They stay in the SPARSE RMSProp group via their existing names.
        self.output_ngram_gate = nn.Linear(config.n_embd, 3, bias=False)
        self.output_ngram_lambdas = nn.Parameter(torch.zeros(3))
        self.ngram_lambdas = nn.Parameter(torch.ones(3))
        # Per-token gate on the input n-gram contributions: a Linear(n_embd->3)
        # on norm(wte_x) yields a per-token 2*sigmoid(.) in (0,2) for each n-gram order,
        # zero-init -> identity at init.
        self.ngram_gate = nn.Linear(config.n_embd, 3, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Per-token adaptive gate on the x0 re-injection: a per-layer Linear(->1)
        # on norm(x) gives a per-token scalar 2*sigmoid(.) in (0,2), centered at 1
        # so zero-init is identity at init.
        self.x0_gates = nn.ModuleList([nn.Linear(config.n_embd, 1, bias=False)
                                       for _ in range(config.n_layer)])
        # Learned aggregation of intermediate-depth representations into the
        # prediction: one scalar per intermediate layer weights its (normed) block
        # output into the final representation before lm_head. Zero-init -> no-op
        # (only final layer feeds output), then learned to route mid-depth signal.
        self.output_layer_lambdas = nn.Parameter(torch.zeros(config.n_layer - 1))
        # Per-layer learnable attention temperature (applied to q after QK-norm).
        # Init 1.0 -> no-op at init. See CausalSelfAttention.forward for rationale.
        self.attn_temp = nn.Parameter(torch.ones(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_bigram_wte = nn.Embedding(config.ve_ngram_vocab, kv_dim)
        self.ve_trigram_wte = nn.Embedding(config.ve_trigram_vocab, kv_dim)
        self.ve_ngram_lambdas = nn.Parameter(torch.ones(2))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.block_masks = [None] * config.n_layer

    def prepare_attention(self, device):
        return

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.bigram_wte.weight, mean=0.0, std=0.25)
        torch.nn.init.normal_(self.trigram_wte.weight, mean=0.0, std=0.125)
        torch.nn.init.normal_(self.fourgram_wte.weight, mean=0.0, std=0.0625)
        torch.nn.init.normal_(self.ve_bigram_wte.weight, mean=0.0, std=0.125)
        torch.nn.init.normal_(self.ve_trigram_wte.weight, mean=0.0, std=0.0625)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.output_bigram.weight)   # NO-OP at init
        torch.nn.init.zeros_(self.output_gate.weight)     # context-independent at init
        torch.nn.init.constant_(self.output_gate.bias, -2.0)  # sigmoid(-2)=0.12 -> gate starts ~off
        torch.nn.init.zeros_(self.output_ngram_gate.weight)  # 2*sigmoid(0)=1 -> identity routing at init
        # output_ngram_lambdas already zeros -> the output backoff is EXACT no-op at init
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for g in self.x0_gates:
            torch.nn.init.zeros_(g.weight)  # 2*sigmoid(0)=1 -> gate is identity at init
        self.output_layer_lambdas.zero_()  # no-op at init: only final layer feeds lm_head
        self.attn_temp.fill_(1.0)  # no-op at init: identity attention temperature
        self.ngram_lambdas.fill_(1.0)
        torch.nn.init.zeros_(self.ngram_gate.weight)  # 2*sigmoid(0)=1 -> identity at init
        self.ve_ngram_lambdas[0].fill_(0.5)
        self.ve_ngram_lambdas[1].fill_(0.25)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.head_route.weight)  # g_h == exp(tanh(0)) == 1 at init
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.bigram_wte.to(dtype=torch.bfloat16)
        self.trigram_wte.to(dtype=torch.bfloat16)
        self.fourgram_wte.to(dtype=torch.bfloat16)
        self.ve_bigram_wte.to(dtype=torch.bfloat16)
        self.ve_trigram_wte.to(dtype=torch.bfloat16)
        self.output_bigram.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        tiny_window = long_window // 4  # quarter-context (T): cheapest attention, for shallow local layers
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + self.bigram_wte.weight.numel() +
                          self.trigram_wte.weight.numel() + self.fourgram_wte.weight.numel() +
                          self.ve_bigram_wte.weight.numel() + self.ve_trigram_wte.weight.numel() +
                          self.output_bigram.weight.numel() +
                          value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.output_layer_lambdas.numel() + self.attn_temp.numel() +
                          self.ngram_lambdas.numel() + self.ve_ngram_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        bigram_wte = sum(p.numel() for p in self.bigram_wte.parameters())
        trigram_wte = sum(p.numel() for p in self.trigram_wte.parameters())
        fourgram_wte = sum(p.numel() for p in self.fourgram_wte.parameters())
        ve_bigram_wte = sum(p.numel() for p in self.ve_bigram_wte.parameters())
        ve_trigram_wte = sum(p.numel() for p in self.ve_trigram_wte.parameters())
        output_bigram = sum(p.numel() for p in self.output_bigram.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.output_layer_lambdas.numel() + self.attn_temp.numel() +
                   self.ngram_lambdas.numel() + self.output_ngram_lambdas.numel() +
                   self.ve_ngram_lambdas.numel())
        total = (wte + bigram_wte + trigram_wte + fourgram_wte + ve_bigram_wte +
                 ve_trigram_wte + output_bigram + value_embeds + lm_head +
                 transformer_matrices + scalars)
        return {
            'wte': wte, 'bigram_wte': bigram_wte, 'trigram_wte': trigram_wte,
            'fourgram_wte': fourgram_wte,
            've_bigram_wte': ve_bigram_wte, 've_trigram_wte': ve_trigram_wte,
            'output_bigram': output_bigram,
            'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        # Dense embeddings (token wte + value-embeds): AdamW with warmdown.
        dense_embedding_params = (list(self.transformer.wte.parameters()) +
                                  list(self.value_embeds.parameters()))
        # SPARSE hashed n-gram tables (input bigram/trigram/fourgram + value-path
        # ve_bigram/ve_trigram): RMSProp (second moment only) + NO warmdown.
        # output_bigram joins the SPARSE hashed-table group: like the n-gram tables
        # its gradient touches only observed current-tokens each step, so RMSProp
        # (second moment only, no warmdown) is the right fit.
        sparse_ngram_params = (list(self.bigram_wte.parameters()) +
                               list(self.trigram_wte.parameters()) +
                               list(self.fourgram_wte.parameters()) +
                               list(self.ve_bigram_wte.parameters()) +
                               list(self.ve_trigram_wte.parameters()) +
                               list(self.output_bigram.parameters()))
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas, self.output_layer_lambdas, self.attn_temp]
        ngram_params = [self.ngram_lambdas, self.output_ngram_lambdas]
        ve_ngram_params = [self.ve_ngram_lambdas]
        x0_params = [self.x0_lambdas]
        x0_gate_params = (list(self.x0_gates.parameters()) + list(self.ngram_gate.parameters()) +
                          list(self.output_gate.parameters()) +
                          list(self.output_ngram_gate.parameters()))
        assert len(list(self.parameters())) == (len(matrix_params) + len(dense_embedding_params) +
            len(lm_head_params) + len(sparse_ngram_params) + len(resid_params) +
            len(ngram_params) + len(ve_ngram_params) + len(x0_params) + len(x0_gate_params))
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        sparse_betas = (adam_betas[0], 0.999)  # RMSProp beta2 only; 0.999 so rare hashed buckets keep their second-moment estimate across the long gaps between sparse hits (preserves normalization on the Zipfian tail)
        # The SPARSE RMSProp group uses a larger eps than the dense AdamW groups.
        # A rarely-hit bucket's exp_avg_sq has only ~1 real contribution
        # (~(1-beta2)*grad^2) yet the GLOBAL bias correction (1-beta2^step) treats
        # it as mature -> exp_avg_sq/bias2 is tiny -> denom~=eps -> a giant cold-start
        # step. eps=1e-8 (still >=2 orders below a well-conditioned bucket's denom
        # ~|grad|, so near-identity for frequent buckets) caps the Zipfian-tail
        # cold-start over-steps. Sparse group only; dense AdamW eps untouched.
        sparse_eps = 1e-8
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
            dict(kind='adamw', params=dense_embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=EMBED_WARMDOWN_RATIO),
            dict(kind='rmsprop', params=sparse_ngram_params, lr=embedding_lr * dmodel_lr_scale, betas=sparse_betas, eps=sparse_eps, weight_decay=0.0, warmdown=SPARSE_WARMDOWN_RATIO),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
            dict(kind='adamw', params=ngram_params, lr=scalar_lr * 0.10, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
            dict(kind='adamw', params=ve_ngram_params, lr=scalar_lr * 0.10, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
            dict(kind='adamw', params=x0_gate_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, warmdown=WARMDOWN_RATIO),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                warmdown=WARMDOWN_RATIO,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        prev = torch.roll(idx, 1, 1)
        prev[:, 0] = 0
        prev2 = torch.roll(idx, 2, 1)
        prev2[:, :2] = 0
        prev3 = torch.roll(idx, 3, 1)
        prev3[:, :3] = 0
        # Avalanche (fmix) hashing for the n-gram indices -> good bucket
        # distribution / few collisions. The
        # tables are learned from scratch so the specific hash only affects collision
        # spread; int64 wraps modulo 2^64 (the standard mix behavior) and the low bits
        # survive the &mask. bigram_h/trigram_h are shared by the input and value paths.
        def _fmix(h):
            h = h ^ (h >> 29)
            h = h * 2654435761
            h = h ^ (h >> 32)
            return h
        bigram_h = _fmix(idx * 2654435761 + prev * 2246822519)
        trigram_h = _fmix(idx * 2654435761 + prev * 2246822519 + prev2 * 3266489917)
        fourgram_h = _fmix(idx * 2654435761 + prev * 2246822519 + prev2 * 3266489917 + prev3 * 668265263)
        bigram_idx = bigram_h & (self.config.ngram_vocab - 1)
        trigram_idx = trigram_h & (self.config.trigram_vocab - 1)
        fourgram_idx = fourgram_h & (self.config.fourgram_vocab - 1)
        ve_bigram_idx = bigram_h & (self.config.ve_ngram_vocab - 1)
        ve_trigram_idx = trigram_h & (self.config.ve_trigram_vocab - 1)
        bigram_x = self.bigram_wte(bigram_idx)
        trigram_x = self.trigram_wte(trigram_idx)
        fourgram_x = self.fourgram_wte(fourgram_idx)
        # Per-token gate (identity at init) modulates each n-gram order's contribution.
        wte_x = self.transformer.wte(idx)
        ngram_gate = 2 * torch.sigmoid(self.ngram_gate(norm(wte_x)))
        ngram_x = (self.ngram_lambdas[0] * ngram_gate[..., 0:1] * bigram_x +
                   self.ngram_lambdas[1] * ngram_gate[..., 1:2] * trigram_x +
                   self.ngram_lambdas[2] * ngram_gate[..., 2:3] * fourgram_x)
        x = wte_x + ngram_x
        x = norm(x)
        x0 = x
        ve_ngram = (self.ve_ngram_lambdas[0] * self.ve_bigram_wte(ve_bigram_idx) +
                    self.ve_ngram_lambdas[1] * self.ve_trigram_wte(ve_trigram_idx))
        layer_outputs = []
        for i, block in enumerate(self.transformer.h):
            x0_gate = 2 * torch.sigmoid(self.x0_gates[i](norm(x)))
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            ve = ve_ngram
            if str(i) in self.value_embeds:
                ve = ve + self.value_embeds[str(i)](idx)
            x = block(x, ve, cos_sin, self.window_sizes[i], self.block_masks[i],
                      self.attn_temp[i])
            if i < self.config.n_layer - 1:
                layer_outputs.append(x)
        # Route intermediate-depth reps into the prediction. Each intermediate
        # block output is weighted by a learned (zero-init) scalar a_i that carries
        # both the depth WEIGHT and the depth-growing SCALE of that residual stream
        # (deeper streams are more confident). The RAW h_i is aggregated (not a
        # per-stream norm(h_i)) so a_i's gradient stays coupled to that scale.
        for i, h in enumerate(layer_outputs):
            x = x + self.output_layer_lambdas[i] * h
        x = norm(x)

        # OUTPUT N-GRAM BACKOFF (FREE). Re-inject the already-gathered input n-gram
        # embeddings -- bigram_x=(cur,prev), trigram_x=(cur,prev,prev2), fourgram_x=
        # (cur,prev,prev2,prev3) -- as a per-token gated residual on the final normed
        # rep, decoded by the SHARED lm_head below. This is the same higher-order
        # lexical-backoff signal the old rank-d tables tried to add, but it costs no new
        # gather/table (these tensors already exist from the input mix) and gives the
        # lexical prior a direct shortcut to the logits. Per-order routing gate (identity
        # at init) x per-order scalar magnitude (zero at init) -> EXACT no-op at init.
        # output_bigram's gate is read off the PRE-injection normed rep (kept faithful
        # to the scalar-head baseline), then the n-gram backoff is added to x.
        out_gate = torch.sigmoid(self.output_gate(x))
        out_ngram_gate = 2 * torch.sigmoid(self.output_ngram_gate(x))
        x = (x + self.output_ngram_lambdas[0] * out_ngram_gate[..., 0:1] * bigram_x
               + self.output_ngram_lambdas[1] * out_ngram_gate[..., 1:2] * trigram_x
               + self.output_ngram_lambdas[2] * out_ngram_gate[..., 2:3] * fourgram_x)

        if OUTPUT_EMBEDDING_CENTERING:
            lm_head_weight = self.lm_head.weight
            lm_head_weight_centered = lm_head_weight - lm_head_weight.mean(dim=0, keepdim=True)
            logits = F.linear(x, lm_head_weight_centered)
        else:
            logits = self.lm_head(x)
        # Direct bigram contribution to the next-token logits (full-vocab logit table,
        # kept in logit space). Added while still bf16 -- before the single float()
        # upcast -- so we never materialize a second [B,T,V] float tensor. The per-token
        # gate (out_gate, computed on the pre-injection normed rep above, bias -2 ->
        # ~0.12 at init) modulates how much lexical bigram prior to trust; zero-init
        # table -> no-op at init.
        logits = logits + out_gate.to(logits.dtype) * self.output_bigram(idx)
        logits = logits.float()
        logits = FINAL_LOGIT_SOFTCAP * torch.tanh(logits / FINAL_LOGIT_SOFTCAP)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    # Second-moment only (no first moment) -> ~50% less optimizer VRAM, suited to
    # the sparse hashed n-gram tables whose gradients are bursty/sparse.
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rms_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rms_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rms_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rms_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rms_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_rmsprop(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._rms_step_t.fill_(state['step'])
            self._rms_lr_t.fill_(group['lr'])
            self._rms_beta2_t.fill_(group['betas'][1])
            self._rms_eps_t.fill_(group['eps'])
            self._rms_wd_t.fill_(group['weight_decay'])
            rmsprop_step_fused(p, grad, state['exp_avg_sq'],
                               self._rms_step_t, self._rms_lr_t, self._rms_beta2_t,
                               self._rms_eps_t, self._rms_wd_t)

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'rmsprop':
                self._step_rmsprop(group)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "TTTLSSSL" # depth-graded receptive field. T=quarter (shallow local), S=half, L=full anchors

TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
TRAIN_SEQUENCE_LEN = 2048
EMBEDDING_LR = 0.20
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.020
SCALAR_LR = 0.50
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.6          # matrices (Muon) + lm_head + scalars: full cosine-tail anneal
EMBED_WARMDOWN_RATIO = 0.3    # dense token wte + value-embeds: late anneal
SPARSE_WARMDOWN_RATIO = 0.0   # hashed n-gram tables (RMSProp): NO anneal, full LR whole run
MUON_MOM_WARMDOWN_START = 0.4 # align momentum warm-down onset with the LR anneal onset (1-WARMDOWN_RATIO=0.4)
MUON_MOM_FINAL = 0.88
MUON_MOM_WARMUP_STEPS = 300 # steps to warm momentum 0.86->0.96
FINAL_LR_FRAC = 0.0

DEPTH = 8
DEVICE_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
OUTPUT_EMBEDDING_CENTERING = False
FINAL_LOGIT_SOFTCAP = 15.0
BOS_LOCAL_CURRICULUM_RATIO = 0.35
BOS_LOCAL_CURRICULUM_WINDOW = 96
BOS_LOCAL_CURRICULUM_BOOST = 0.35

# Tail-EMA weight averaging: eval on an fp32 running-mean of the params
# over the warmed-down tail of training. The optimizer state is untouched; only
# the final weights handed to the evaluator change. Excludes the sparse hashed
# n-gram tables (their full-LR-the-whole-run schedule keeps moving, averaging them
# only smears collisions).
WEIGHT_AVG_ENABLE = True       # average params over the warmed-down tail for eval
WEIGHT_AVG_START_FRAC = 0.85   # begin tail averaging at this progress (last 15%)
WEIGHT_AVG_EMA_DECAY = 0.94 # tail-EMA decay (eff ~17 tail steps): controlled A/B showed tighter beats wider (0.95->0.9061 vs 0.97->0.9072); 0.94 steps just past the proven-best 0.95 toward the most-converged near-zero-LR tail snapshots.
WEIGHT_AVG_EXCLUDE_PREFIXES = (
    "bigram_wte", "trigram_wte", "fourgram_wte", "ve_bigram_wte", "ve_trigram_wte",
    "output_bigram",
)
WALL_CLOCK_CAP = 1140.0

# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def train_model(tokenizer, make_train_dataloader, max_seq_len, time_budget, device=None):
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = device or torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    bf16_peak_flops = 2250e12 if "B200" in torch.cuda.get_device_name(0) else 989.5e12

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    train_seq_len = min(TRAIN_SEQUENCE_LEN, max_seq_len)
    config = GPTConfig(
        sequence_len=max_seq_len, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
    print(f"Model config: {asdict(config)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.prepare_attention(device)

    param_counts = model.num_scaling_params()
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Params: {num_params/1e6:.1f}M, est FLOPs/token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * train_seq_len
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    raw_model = model  # keep handle to the real module (compile shares its Parameters)
    # mode="max-autotune" enables CUDA graphs to remove per-kernel CPU launch
    # overhead; graph replay re-runs the identical captured kernels so loss is
    # numerically exact. The FA4 + fused-RoPE/QK-norm custom ops are PT2
    # custom_ops with register_fake (static shapes, no CPU sync in the captured
    # region) so they are capturable; if any op breaks capture Inductor logs
    # "skipping cudagraphs" and falls back to the same kernels (no crash, no
    # math change).
    model = torch.compile(model, mode="max-autotune", dynamic=False)

    avg_buffers = []  # list of (param, fp32 running-mean buffer)
    if WEIGHT_AVG_ENABLE:
        for name, p in raw_model.named_parameters():
            if any(name.startswith(pre) for pre in WEIGHT_AVG_EXCLUDE_PREFIXES):
                continue
            avg_buffers.append((p, torch.zeros_like(p, dtype=torch.float32)))
    avg_count = 0

    train_loader = _Prefetcher(make_train_dataloader(DEVICE_BATCH_SIZE, train_seq_len), depth=4)
    x, y, epoch = next(train_loader)  # prefetch first batch
    bos_token_id = tokenizer.get_bos_token_id()
    position_ids = torch.arange(train_seq_len, device=device, dtype=torch.long).view(1, train_seq_len)

    print(f"Time budget: {time_budget}s, train_seq_len: {train_seq_len}, grad accum steps: {grad_accum_steps}")

    def get_lr_multiplier(progress, warmdown=WARMDOWN_RATIO):
        if warmdown <= 0.0:
            return 1.0  # no anneal -> full LR the whole run (sparse n-gram tables)
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - warmdown:
            return 1.0
        else:
            cooldown = (1.0 - progress) / warmdown
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step, progress):
        frac = min(step / MUON_MOM_WARMUP_STEPS, 1)
        mom = (1 - frac) * 0.86 + frac * 0.96
        if progress > MUON_MOM_WARMDOWN_START:
            f = (progress - MUON_MOM_WARMDOWN_START) / (1.0 - MUON_MOM_WARMDOWN_START)
            mom = mom + min(f, 1.0) * (MUON_MOM_FINAL - 0.96)
        return mom

    def get_weight_decay(progress):
        decay = math.cos(0.5 * math.pi * progress)
        return WEIGHT_DECAY * decay * decay

    # HOST-OVERHEAD: the run-to-run step-time spread is host/CPU-launch contention
    # (the GPU's compute+bandwidth are constant across fast/slow runs; the slow runs
    # are the host failing to feed the GPU between launches). Each cuda.synchronize()
    # is a CPU round-trip that a contended host clears slowly. The old loop synced
    # ~3x/step: a redundant top sync (prev iter already synced), train_loss.item(),
    # and an explicit bottom sync. We collapse to ONE sync/step (the .item(), which
    # still serializes the dataloader vs the cudagraph static input buffers -> safe)
    # and time the step as the delta between successive .item() syncs.
    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    torch.cuda.synchronize()
    t_prev = time.time()

    while True:
        progress = min(total_training_time / time_budget, 1.0)
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                token_losses = model(x, y, reduction='none').view_as(y)
            raw_loss = token_losses.mean()
            train_loss = raw_loss.detach()
            if progress < BOS_LOCAL_CURRICULUM_RATIO:
                last_bos_pos = torch.where(x == bos_token_id, position_ids, torch.zeros_like(position_ids)).cummax(dim=1).values
                local_mask = (position_ids - last_bos_pos) < BOS_LOCAL_CURRICULUM_WINDOW
                local_boost = BOS_LOCAL_CURRICULUM_BOOST * (1.0 - progress / BOS_LOCAL_CURRICULUM_RATIO)
                weights = 1.0 + local_boost * local_mask.to(token_losses.dtype)
                loss = (token_losses * weights).sum() / weights.sum()
            else:
                loss = raw_loss
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)  # logged value (matrix group)
        muon_momentum = get_muon_momentum(step, progress)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr_multiplier(progress, group["warmdown"])
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if avg_buffers and progress >= WEIGHT_AVG_START_FRAC:
            avg_count += 1
            with torch.no_grad():
                if avg_count == 1:
                    for p, buf in avg_buffers:
                        buf.copy_(p.detach().float())
                else:
                    for p, buf in avg_buffers:
                        buf.lerp_(p.detach().float(), 1.0 - WEIGHT_AVG_EMA_DECAY)

        train_loss_f = train_loss.item()  # the single per-step GPU sync

        if math.isnan(train_loss_f) or train_loss_f > 100:
            raise RuntimeError(f"Training diverged at step {step}: loss={train_loss_f}")

        t1 = time.time()
        dt = t1 - t_prev
        t_prev = t1

        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

        if step % 50 == 0 or total_training_time >= time_budget:
            pct_done = 100 * progress
            tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
            mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / bf16_peak_flops
            remaining = max(0, time_budget - total_training_time)
            print(f"step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
                  f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
                  f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and total_training_time >= time_budget:
            break

        if (time.time() - t_start) > WALL_CLOCK_CAP:
            print(f"wall-clock guard hit at {time.time() - t_start:.1f}s "
                  f"(step {step}); stopping to stay under the evaluator cap", flush=True)
            break

    gc.enable()

    if avg_buffers and avg_count > 0:
        with torch.no_grad():
            for p, buf in avg_buffers:
                p.data.copy_(buf.to(p.dtype))
        print(f"applied tail EMA (decay={WEIGHT_AVG_EMA_DECAY}) over {avg_count} "
              f"tail steps ({len(avg_buffers)} tensors)", flush=True)

    total_tokens = step * TOTAL_BATCH_SIZE
    startup_time = t_start_training - t_start
    info = {
        "training_seconds": round(total_training_time, 1),
        "startup_seconds": round(startup_time, 1),
        "num_steps": step,
        "num_params_m": round(num_params / 1e6, 1),
        "total_tokens_m": round(total_tokens / 1e6, 1),
        "train_seq_len": train_seq_len,
        "depth": DEPTH,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "final_smooth_loss": round(debiased_smooth_loss, 6),
        "weight_avg_steps": avg_count,
    }
    print(f"train_model done: {info}")
    return model, info