# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _build_attn_mask(
    batch_size,
    q_len,
    k_len,
    device,
    dtype,
    q_lens=None,
    k_lens=None,
    causal=False,
):
    if q_lens is None and k_lens is None and not causal:
        return None

    valid = torch.ones((batch_size, q_len, k_len), dtype=torch.bool, device=device)
    if q_lens is not None:
        q_lens = q_lens.to(device=device, dtype=torch.long)
        q_positions = torch.arange(q_len, device=device).view(1, q_len, 1)
        valid &= q_positions < q_lens.view(-1, 1, 1)
    if k_lens is not None:
        k_lens = k_lens.to(device=device, dtype=torch.long)
        k_positions = torch.arange(k_len, device=device).view(1, 1, k_len)
        valid &= k_positions < k_lens.view(-1, 1, 1)
    if causal:
        q_positions = torch.arange(q_len, device=device).view(1, q_len, 1)
        k_positions = torch.arange(k_len, device=device).view(1, 1, k_len)
        valid &= k_positions <= q_positions

    attn_mask = torch.zeros((batch_size, 1, q_len, k_len), dtype=dtype, device=device)
    attn_mask.masked_fill_(~valid.unsqueeze(1), torch.finfo(dtype).min)
    return attn_mask


def _dense_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * (softmax_scale * math.sqrt(q.shape[-1]))

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if torch_npu is not None and q.device.type == "npu":
        attn_mask_npu = attn_mask < -1 if attn_mask is not None else None
        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            input_layout="BNSD",
            pse=None,
            atten_mask=attn_mask_npu,
            scale=1.0 / math.sqrt(q.shape[-1]),
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1.0,
            sync=False,
            inner_precise=0,
        )[0]
    else:
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal if attn_mask is None else False,
            dropout_p=dropout_p,
        )

    return out.transpose(1, 2).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    if q.device.type != 'cuda':
        attn_mask = _build_attn_mask(
            batch_size=b,
            q_len=lq,
            k_len=lk,
            device=q.device,
            dtype=q.dtype if q.dtype in half_dtypes else dtype,
            q_lens=q_lens,
            k_lens=k_lens,
            causal=causal,
        )
        return _dense_attention(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        ).type(out_dtype)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        attn_mask = _build_attn_mask(
            batch_size=q.size(0),
            q_len=q.size(1),
            k_len=k.size(1),
            device=q.device,
            dtype=q.dtype if q.dtype in (torch.float16, torch.bfloat16) else dtype,
            q_lens=q_lens,
            k_lens=k_lens,
            causal=causal,
        )
        return _dense_attention(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
