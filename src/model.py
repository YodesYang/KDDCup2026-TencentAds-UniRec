"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. FFN: Two modes (see ``ffn_mode``).
    3. Residual connection: Q_boost = Q + Q_e.

    FFN modes (T19 / ADR-007):
      - ``shared`` (default, back-compat): one ``nn.Linear`` per layer, all
        T tokens share the same FFN weights. Mirrors the legacy behavior
        of this block prior to T19.
      - ``per_token``: each of the T tokens has its own independent FFN
        weight matrix. This matches the original RankMixer paper's
        "Per-token FFN" definition: allocating independent parameters per
        feature subspace to avoid dominant features flooding long-tail
        tokens. Parameters grow ~T× vs shared mode (T=15 @ d_model=64,
        hidden_mult=4 → ~495k extra dense params per block).

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full',  # 'full' | 'ffn_only' | 'none'
        ffn_mode: str = 'shared',  # 'shared' | 'per_token'  (T19 / ADR-007)
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode
        self.ffn_mode = ffn_mode
        if ffn_mode not in ('shared', 'per_token'):
            raise ValueError(
                f"ffn_mode must be 'shared' or 'per_token', got {ffn_mode!r}"
            )

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # FFN — used by both 'full' and 'ffn_only'.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

        hidden_dim = d_model * hidden_mult
        self._hidden_dim = hidden_dim

        if ffn_mode == 'shared':
            # Legacy path: single Linear pair shared across all T tokens.
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        else:
            # Per-token FFN: independent weight matrices for every token,
            # implemented as grouped linear via torch.einsum so it stays
            # vectorized (no Python-level for-loop, torch.compile friendly).
            # Weight shapes:
            #   fc1_weight: (T, D, hidden_dim)   fc1_bias: (T, hidden_dim)
            #   fc2_weight: (T, hidden_dim, D)   fc2_bias: (T, D)
            # We do not use nn.ModuleList[Linear]*T because that forces a
            # Python for-loop on forward and defeats GPU parallelism.
            self.fc1_weight = nn.Parameter(torch.empty(n_total, d_model, hidden_dim))
            self.fc1_bias = nn.Parameter(torch.zeros(n_total, hidden_dim))
            self.fc2_weight = nn.Parameter(torch.empty(n_total, hidden_dim, d_model))
            self.fc2_bias = nn.Parameter(torch.zeros(n_total, d_model))
            # Mirror nn.Linear default initialization (Kaiming uniform on
            # weight, uniform bias in [-1/sqrt(fan_in), +1/sqrt(fan_in)]).
            # Doing it per-token keeps each token's FFN statistically
            # equivalent to a fresh nn.Linear at init.
            for t in range(n_total):
                nn.init.kaiming_uniform_(self.fc1_weight[t], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.fc2_weight[t], a=math.sqrt(5))
                fan_in_1 = d_model
                fan_in_2 = hidden_dim
                bound_1 = 1.0 / math.sqrt(fan_in_1)
                bound_2 = 1.0 / math.sqrt(fan_in_2)
                nn.init.uniform_(self.fc1_bias[t], -bound_1, bound_1)
                nn.init.uniform_(self.fc2_bias[t], -bound_2, bound_2)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def _apply_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN block. Dispatches to shared or per-token path.

        Args:
            x: (B, T, D) normalized input.

        Returns:
            (B, T, D) FFN output (pre-residual).
        """
        if self.ffn_mode == 'shared':
            h = self.fc1(x)
            h = F.gelu(h)
            h = self.dropout(h)
            return self.fc2(h)
        # per_token: each token indexes its own (D, hidden) / (hidden, D) matrix.
        # x: (B, T, D) · fc1_weight: (T, D, H) → (B, T, H)
        # einsum does not allocate intermediate (B, T, D, H) tensors.
        h = torch.einsum('btd,tdh->bth', x, self.fc1_weight) + self.fc1_bias
        h = F.gelu(h)
        h = self.dropout(h)
        # (B, T, H) · fc2_weight: (T, H, D) → (B, T, D)
        return torch.einsum('bth,thd->btd', h, self.fc2_weight) + self.fc2_bias

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # FFN block (shared or per-token)
        x = self.norm(Q_hat)
        Q_e = self._apply_ffn(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class TargetItemSeqInjector(nn.Module):
    """Inject target-item representation into each sequence token (ADR-004).

    Makes the per-sequence encoder target-aware by adding a per-sequence
    projection of the target item representation to every seq token, before
    the sequence encoder runs. This is orthogonal to the item-conditioned
    query branch in :class:`MultiSeqQueryGenerator` (which acts on the query
    side): injection here modifies the encoder's key/value input so the
    encoder itself can perform target-specific matching.

    Mode ``'additive'`` (default):
        target_repr = mean(item_ns_tokens)            # (B, D)
        target_proj_i = LinearPerSeq_i(target_repr)   # (B, D)
        seq_tokens_i' = seq_tokens_i + α_i * target_proj_i.unsqueeze(1)

    α_i is a learned per-sequence scalar initialised to ``alpha_init``
    (default 0.0). With α=0 the injector is a functional no-op, so training
    starts equivalent to the baseline and the optimiser decides the strength
    of each domain's injection. This is the safest path to enable target-
    aware conditioning without breaking a well-trained baseline.

    Padding tokens still receive the injection at the token level, but the
    downstream seq encoder masks them out via ``seq_padding_masks`` so they
    contribute neither to pooling nor to attention output.

    Mode ``'off'``: the module short-circuits (identity) and has no params.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        mode: str = 'off',
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.mode = mode
        if mode == 'off':
            return
        if mode != 'additive':
            raise ValueError(
                f"Unsupported target_item_seq_injection mode: {mode!r}. "
                f"Accepted: 'off', 'additive'.")
        self.target_proj_per_seq = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_sequences)
        ])
        # Per-sequence learnable scalar; init=0 → starts as a no-op so the
        # optimiser decides how much each domain should attend to target.
        self.alpha = nn.Parameter(
            torch.full((num_sequences,), float(alpha_init)))

    def forward(
        self,
        seq_tokens_list: list,
        item_repr: torch.Tensor,
    ) -> list:
        """Add α_i * W_i(item_repr) to each token of sequence i.

        Args:
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            item_repr: (B, D) target-item representation (mean of item NS
                tokens, same source as ICQ).

        Returns:
            New list of (B, L_i, D) tensors with target injected. Input list
            is not mutated.
        """
        if self.mode == 'off':
            return seq_tokens_list
        out = []
        for i in range(self.num_sequences):
            # (B, D) → (B, 1, D), broadcast-add across sequence length
            proj = self.target_proj_per_seq[i](item_repr).unsqueeze(1)
            out.append(seq_tokens_list[i] + self.alpha[i] * proj)
        return out


class DINInterestExtractor(nn.Module):
    """DIN-style target-aware interest pooling (T25 / G1).

    For each of the S sequence domains, computes a single interest token
    by cross-attending a target-item query against that domain's sequence
    tokens. In ``compact`` mode the S interest vectors are concatenated and
    projected back to ``d_model``; in ``per_domain`` mode the concatenated
    ``S*d_model`` vector is exposed directly to the classifier.

    This representation is emitted as a side output (B, D) and is later
    concatenated to the backbone output before the classifier — exactly
    the same pattern ``enable_dense_bypass`` already uses — so it does
    NOT require re-wiring the RankMixer divisibility constraint
    (``d_model % n_total == 0``) that makes inserting a new NS/Q token
    painful. This is the minimum-risk "bypass" integration of DIN.

    Compared to ``TargetItemSeqInjector`` (additive, already falsified
    in T16):
      * T16 additive adds a single projected target vector to EVERY
        sequence token with a scalar α. Low expressiveness; limited to
        uniformly-weighted contextual bias.
      * DIN performs an ATTENTION pooling: the target query can
        selectively weight which sequence positions are relevant. This
        matches Alibaba's DIN KDD 2018 target-aware interest mechanism.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_sequences: int,
        dropout: float = 0.0,
        merge_mode: str = 'compact',
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.merge_mode = merge_mode
        if merge_mode not in ('compact', 'per_domain'):
            raise ValueError(
                f"DINInterestExtractor merge_mode must be 'compact' or "
                f"'per_domain', got {merge_mode!r}")
        # One cross-attention head per sequence domain. We use the shared
        # CrossAttention class that already exists in this file to avoid
        # re-implementing MHA + RoPE handling.
        self.attn_per_seq = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre',
            )
            for _ in range(num_sequences)
        ])
        # Merge S interest vectors (B, S, D) → (B, D).
        if merge_mode == 'compact':
            self.merge = nn.Sequential(
                nn.Linear(d_model * num_sequences, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
        else:
            self.merge = None

    def forward(
        self,
        target_repr: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_padding_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute composite interest token via per-seq target attention.

        Args:
            target_repr: (B, D) target-item representation. Typically the
                mean of item_ns_tokens (same source as TargetItemSeqInjector).
            seq_tokens_list: List of S tensors, each (B, L_i, D).
            seq_padding_masks: List of S boolean masks (B, L_i), True =
                padding position.

        Returns:
            composite_interest: (B, D) — the merged DIN interest vector,
            ready to be concatenated to the classifier input.
        """
        if len(seq_tokens_list) != self.num_sequences:
            raise ValueError(
                f"DINInterestExtractor expects {self.num_sequences} seq "
                f"domains; got {len(seq_tokens_list)}.")
        # Target as a single-token query: (B, D) → (B, 1, D).
        query = target_repr.unsqueeze(1)
        per_seq_interests: List[torch.Tensor] = []
        for i in range(self.num_sequences):
            seq_tokens = seq_tokens_list[i]           # (B, L_i, D)
            mask = seq_padding_masks[i]               # (B, L_i)
            # Single-query cross-attention → (B, 1, D).
            attended = self.attn_per_seq[i](
                query=query,
                key_value=seq_tokens,
                key_padding_mask=mask,
            )
            per_seq_interests.append(attended.squeeze(1))  # (B, D)
        concat = torch.cat(per_seq_interests, dim=-1)  # (B, S*D)
        if self.merge_mode == 'per_domain':
            return concat
        if self.merge is None:
            raise RuntimeError("DINInterestExtractor compact merge is missing")
        return self.merge(concat)                       # (B, D)


class TINLiteInterestExtractor(nn.Module):
    """Target-aware temporal interest pooling.

    This is a conservative TIN-style upgrade over the existing DIN bypass.
    It keeps the same output shape and merge modes as ``DINInterestExtractor``
    but enriches each sequence token with a target-conditioned time-bucket
    modulation before the target-item attention pooling:

        seq'_t = seq_t + alpha_i * MLP_i([time_t, target, time_t * target])

    The final projection in each temporal MLP is zero-initialized, so the
    module starts as ordinary DIN and learns whether the explicit temporal
    modulation is useful. Sparse/id features and the main HyFormer path are
    untouched.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_sequences: int,
        num_time_buckets: int,
        dropout: float = 0.0,
        merge_mode: str = 'compact',
        time_alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.num_time_buckets = int(num_time_buckets)
        self.merge_mode = merge_mode
        if self.num_time_buckets <= 0:
            raise ValueError(
                "TINLiteInterestExtractor requires num_time_buckets > 0")
        if merge_mode not in ('compact', 'per_domain'):
            raise ValueError(
                f"TINLiteInterestExtractor merge_mode must be 'compact' or "
                f"'per_domain', got {merge_mode!r}")

        self.time_embedding = nn.Embedding(
            self.num_time_buckets, d_model, padding_idx=0)
        self.time_mod_per_seq = nn.ModuleList()
        for _ in range(num_sequences):
            mod = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            # Start from DIN-equivalent behavior; training can grow the
            # target-time modulation if it transfers.
            nn.init.zeros_(mod[3].weight)
            nn.init.zeros_(mod[3].bias)
            self.time_mod_per_seq.append(mod)
        self.time_alpha = nn.Parameter(
            torch.full((num_sequences,), float(time_alpha_init)))

        self.attn_per_seq = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre',
            )
            for _ in range(num_sequences)
        ])
        if merge_mode == 'compact':
            self.merge = nn.Sequential(
                nn.Linear(d_model * num_sequences, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
        else:
            self.merge = None

        nn.init.xavier_normal_(self.time_embedding.weight.data)
        self.time_embedding.weight.data[0, :] = 0

    def forward(
        self,
        target_repr: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_padding_masks: List[torch.Tensor],
        seq_time_buckets_list: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(seq_tokens_list) != self.num_sequences:
            raise ValueError(
                f"TINLiteInterestExtractor expects {self.num_sequences} seq "
                f"domains; got {len(seq_tokens_list)}.")
        if len(seq_time_buckets_list) != self.num_sequences:
            raise ValueError(
                f"TINLiteInterestExtractor expects {self.num_sequences} time "
                f"bucket tensors; got {len(seq_time_buckets_list)}.")

        query = target_repr.unsqueeze(1)  # (B, 1, D)
        per_seq_interests: List[torch.Tensor] = []
        for i in range(self.num_sequences):
            seq_tokens = seq_tokens_list[i]           # (B, L_i, D)
            mask = seq_padding_masks[i]               # (B, L_i)
            time_bucket_ids = seq_time_buckets_list[i]
            if time_bucket_ids.shape != seq_tokens.shape[:2]:
                raise ValueError(
                    "TINLiteInterestExtractor time bucket shape mismatch: "
                    f"domain_index={i}, time={tuple(time_bucket_ids.shape)}, "
                    f"seq={tuple(seq_tokens.shape[:2])}")
            time_repr = self.time_embedding(time_bucket_ids)  # (B, L_i, D)
            target_expand = target_repr.unsqueeze(1).expand_as(seq_tokens)
            mod_in = torch.cat(
                [time_repr, target_expand, time_repr * target_expand],
                dim=-1,
            )
            time_mod = self.time_mod_per_seq[i](mod_in)
            enriched_seq = (
                seq_tokens
                + self.time_alpha[i] * time_mod
            )
            attended = self.attn_per_seq[i](
                query=query,
                key_value=enriched_seq,
                key_padding_mask=mask,
            )
            per_seq_interests.append(attended.squeeze(1))

        concat = torch.cat(per_seq_interests, dim=-1)
        if self.merge_mode == 'per_domain':
            return concat
        if self.merge is None:
            raise RuntimeError("TINLiteInterestExtractor compact merge is missing")
        return self.merge(concat)


class DCNCrossBypass(nn.Module):
    """T34 / EXP-049 · DCN-V2 cross-feature bypass.

    Builds an explicit cross-feature signal from selected user/item scalar
    fids, then concatenates the resulting (B, D) vector to the backbone
    output BEFORE the classifier — the same bypass pattern used by
    ``_apply_dense_bypass`` (T25 dense) and ``_apply_din_interest_bypass``
    (T25 DIN). Orthogonal to all attention-based paths.

    Mechanism (DCN-V2, Wang et al. 2021):
        x0   = Linear_in(concat([emb(fid_i) for fid_i in selected]))   (B, D)
        x_l  = x_{l-1} + x0 * (W_l x_{l-1} + b_l)                       (B, D)
        out  = LayerNorm(x_L)                                           (B, D)

    The "outer-product gating" (``x0 * (W x + b)``) lets one fid's
    embedding modulate every other fid's contribution at every layer —
    DCN-V2's signature explicit cross. ``num_cross_layers`` controls
    interaction order (2 layers ≈ 4-th order interactions).

    Embedding tables are SHARED with the NS tokenizer (passed in by
    PCVRHyFormer.__init__) so the parameter overhead is purely:
      * input projection: (K_u + K_i) * emb_dim * d_model
      * cross layers:     num_cross_layers * (d_model^2 + d_model)
      * output LN:        2 * d_model
    Total ~42k dense params at K_u=2, K_i=5, d_model=64, layers=2.

    Iron-law compliance:
      * "Independent Parameters Iron Law": the cross layers are NOT
        shared with any other module; embeddings ARE shared with NS
        tokenizer (intentional — same fid representation, different
        interaction view), which is how DCN papers always do it.
      * "Stacking Seed Sensitivity": single new flag
        ``enable_dcn_cross`` default off → bit-identical when off.
    """

    def __init__(
        self,
        d_model: int,
        emb_dim: int,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_fids: List[int],
        item_fids: List[int],
        num_cross_layers: int = 2,
    ) -> None:
        """Initializes DCNCrossBypass.

        Args:
            d_model: Output dim (matches backbone output for concat).
            emb_dim: Per-fid embedding dim (matches NS tokenizer ``emb_dim``).
            user_int_feature_specs: Schema of user_int features in the
                order they appear in ``ModelInput.user_int_feats`` —
                ``[(vocab_size, offset, length), ...]``. Used to resolve
                the (offset, length) of each requested user fid.
            item_int_feature_specs: Same for item_int.
            user_fids: List of user fid INDICES into ``user_int_feature_specs``
                (NOT the raw fid number from schema.json — the caller
                must map fid number → list index before constructing).
            item_fids: Same for item.
            num_cross_layers: Number of stacked DCN-V2 layers. Default 2.

        Selected fids MUST satisfy ``length == 1`` (scalar). Multi-hot
        fids would create ambiguity (which slot to pick?) and are
        rejected with ValueError to fail loudly.
        """
        super().__init__()

        if num_cross_layers < 1:
            raise ValueError(
                f"DCNCrossBypass: num_cross_layers must be >= 1, "
                f"got {num_cross_layers}")

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.num_cross_layers = num_cross_layers

        self._validate_scalar(user_fids, user_int_feature_specs, side='user')
        self._validate_scalar(item_fids, item_int_feature_specs, side='item')

        self.user_fids = list(user_fids)
        self.item_fids = list(item_fids)

        # (offset, length) tuples — kept for forward-time slicing. We
        # also use them at __init__ to assert scalar.
        self.user_offsets: List[int] = [
            user_int_feature_specs[fi][1] for fi in self.user_fids
        ]
        self.item_offsets: List[int] = [
            item_int_feature_specs[fi][1] for fi in self.item_fids
        ]

        num_selected = len(self.user_fids) + len(self.item_fids)
        if num_selected < 2:
            raise ValueError(
                f"DCNCrossBypass: need >= 2 selected fids for cross "
                f"feature to make sense, got user={len(self.user_fids)}, "
                f"item={len(self.item_fids)}")

        # Input projection: (K_u + K_i) * emb_dim → d_model + LayerNorm
        # Mirrors the dense-bypass projection style for consistency.
        self.input_proj = nn.Sequential(
            nn.Linear(num_selected * emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # DCN-V2 cross layers — each is a single Linear(D, D) plus bias.
        # Forward pass implements x_{l+1} = x_0 * (W_l x_l + b_l) + x_l.
        # Bias zero-init keeps layer-0 forward = x_0 * (W_l x_l) + x_l ≈
        # x_l initially (small interaction perturbation), so the model
        # gracefully degrades to "x_0 bypass + LayerNorm" at init,
        # matching DCN-V2 paper's recommended starting regime.
        self.cross_W = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_cross_layers)
        ])
        for layer in self.cross_W:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Output LayerNorm — keeps the cross output magnitude in line
        # with the backbone output (which is also LayerNorm'd) so the
        # downstream classifier sees comparable scales for both.
        self.out_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _validate_scalar(
        fid_indices: List[int],
        feature_specs: List[Tuple[int, int, int]],
        side: str,
    ) -> None:
        """Ensures every selected fid is scalar (length == 1).

        Multi-hot fids would require a pooling decision (mean? first
        nonzero?); rejecting them at __init__ is safer than silently
        picking an option. If a user *wants* a multi-hot fid in the
        cross they should mean-pool it upstream and add it as a
        synthetic dense feature.
        """
        for idx in fid_indices:
            if not 0 <= idx < len(feature_specs):
                raise ValueError(
                    f"DCNCrossBypass: {side}_fids contains out-of-range "
                    f"index {idx} (valid range "
                    f"[0, {len(feature_specs)}))")
            vs, _offset, length = feature_specs[idx]
            if length != 1:
                raise ValueError(
                    f"DCNCrossBypass: {side}_fids[{idx}] has length="
                    f"{length}, but only scalar fids (length==1) are "
                    f"supported. Reject loudly to avoid silent "
                    f"mis-pooling.")
            if vs <= 0:
                raise ValueError(
                    f"DCNCrossBypass: {side}_fids[{idx}] has vocab_size="
                    f"{vs}, no embedding to draw from.")

    def forward(
        self,
        user_int_feats: torch.Tensor,
        item_int_feats: torch.Tensor,
        user_embs: nn.ModuleList,
        item_embs: nn.ModuleList,
        user_emb_index: List[int],
        item_emb_index: List[int],
    ) -> torch.Tensor:
        """Builds the cross output (B, D) from selected fids.

        Args:
            user_int_feats: (B, total_user_int_dim) raw integer features.
            item_int_feats: Same for item.
            user_embs: ``nn.ModuleList`` of NS-tokenizer user embeddings.
                ``user_embs[user_emb_index[fid_idx]]`` is the table for
                ``fid_idx``-th user fid (skipped → -1).
            item_embs: Same for item.
            user_emb_index: Maps fid_idx → real index in ``user_embs``,
                with -1 meaning "no embedding (filtered)". A selected
                fid landing on -1 raises (would have been caught by
                ``_validate_scalar`` if vocab_size>0; but if a user
                selected a fid that the tokenizer filtered via
                ``emb_skip_threshold`` we surface that here).
            item_emb_index: Same for item.

        Returns:
            (B, d_model) tensor ready to concat to backbone output.
        """
        embs_to_cat: List[torch.Tensor] = []

        for fi, offset in zip(self.user_fids, self.user_offsets):
            real = user_emb_index[fi]
            if real == -1:
                raise RuntimeError(
                    f"DCNCrossBypass: user fid index {fi} has no "
                    f"embedding (filtered by emb_skip_threshold?). "
                    f"Reduce --dcn_cross_user_fids or raise "
                    f"--emb_skip_threshold.")
            vals = user_int_feats[:, offset].long()  # (B,)
            embs_to_cat.append(user_embs[real](vals))  # (B, emb_dim)

        for fi, offset in zip(self.item_fids, self.item_offsets):
            real = item_emb_index[fi]
            if real == -1:
                raise RuntimeError(
                    f"DCNCrossBypass: item fid index {fi} has no "
                    f"embedding (filtered by emb_skip_threshold?).")
            vals = item_int_feats[:, offset].long()
            embs_to_cat.append(item_embs[real](vals))

        x_concat = torch.cat(embs_to_cat, dim=-1)  # (B, K*emb_dim)
        x0 = self.input_proj(x_concat)              # (B, d_model)

        x = x0
        for layer in self.cross_W:
            x = x0 * layer(x) + x  # DCN-V2 cross step
        return self.out_ln(x)


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]

    When ``item_conditioned=True``, a per-sequence learned gate blends the
    standard global-info query with an item-aware branch:
        item_repr   = mean(item_ns_tokens)        # (B, D)
        gate_i      = σ(W_gate_i · [global_info_i, item_repr])  # (B, D)
        q_{i,j}     = (1 - gate_i) ⊙ FFN_{i,j}(global_info_i)
                    +       gate_i  ⊙ FFN_item_{i,j}(item_repr)
    This lets each domain independently learn how much the target item
    should steer its queries (DIN-style target-item awareness).
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        item_conditioned: bool = False,
        num_item_ns: int = 0,
        # T32 / DECEM Trick 6 · Q_init=MLP(item_emb)+NS_residual.
        # When True, base queries originate from item_repr (projected via MLP)
        # plus an NS context residual, instead of pure global_info FFN.
        # Compatible with item_conditioned (ICQ): if both True, the item branch
        # still runs as a parallel mix on top of the new base path.
        # Default False = bit-identical to baseline (zero behavior change).
        q_init_item: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.item_conditioned = item_conditioned
        self.num_item_ns = num_item_ns
        self.q_init_item = q_init_item

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # T32 / Trick 6 · Build item-emb-driven base query path
        if q_init_item and num_item_ns > 0:
            # Project item_repr to a "rich" query input via MLP (d_model → d_model)
            self.q_init_item_proj = nn.Sequential(
                nn.Linear(d_model, d_model * hidden_mult),
                nn.SiLU(),
                nn.Linear(d_model * hidden_mult, d_model),
                nn.LayerNorm(d_model),
            )
            # Linear residual projection: NS global_info → d_model (preserves NS context)
            self.q_init_ns_residual = nn.Linear(global_info_dim, d_model)
            # Per-sequence per-query FFN consuming (item_repr_proj + ns_residual)
            self.q_init_ffns_per_seq = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model * hidden_mult),
                        nn.SiLU(),
                        nn.Linear(d_model * hidden_mult, d_model),
                        nn.LayerNorm(d_model),
                    )
                    for _ in range(num_queries)
                ])
                for _ in range(num_sequences)
            ])

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

        if item_conditioned and num_item_ns > 0:
            # Item-aware branch: per-sequence, per-query FFN conditioned on item_repr
            self.item_query_ffns_per_seq = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model * hidden_mult),
                        nn.SiLU(),
                        nn.Linear(d_model * hidden_mult, d_model),
                        nn.LayerNorm(d_model),
                    )
                    for _ in range(num_queries)
                ])
                for _ in range(num_sequences)
            ])
            # Per-sequence gate: [global_info, item_repr] → D-dim blend weight
            self.item_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim + d_model, d_model),
                    nn.Sigmoid(),
                )
                for _ in range(num_sequences)
            ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        # Extract item representation for item-conditioned branch.
        # item_ns tokens are the last num_item_ns tokens in ns_tokens.
        item_repr: Optional[torch.Tensor] = None
        if self.item_conditioned and self.num_item_ns > 0:
            item_repr = ns_tokens[:, -self.num_item_ns:, :].mean(dim=1)  # (B, D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Base queries: T32 (Q_init=MLP(item_emb)+NS_residual) or default global_info FFN
            if self.q_init_item and item_repr is not None:
                # Trick 6 · queries originate from item_emb + NS residual
                item_proj = self.q_init_item_proj(item_repr)        # (B, D)
                ns_res = self.q_init_ns_residual(global_info)       # (B, D)
                q_input = item_proj + ns_res                        # (B, D), residual sum
                base_queries = [ffn(q_input) for ffn in self.q_init_ffns_per_seq[i]]
            else:
                # Baseline · base queries from global info
                base_queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]

            if item_repr is not None:
                # Item-aware branch: gate blends base query with item-conditioned query
                gate = self.item_gates[i](
                    torch.cat([global_info, item_repr], dim=-1))  # (B, D)
                item_queries = [
                    ffn(item_repr) for ffn in self.item_query_ffns_per_seq[i]]
                queries = [
                    (1.0 - gate) * bq + gate * iq
                    for bq, iq in zip(base_queries, item_queries)
                ]
            else:
                queries = base_queries

            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        rank_mixer_ffn_mode: str = 'shared',
        enable_din_integrated: bool = False,
        din_integrated_alpha_init: float = 0.1,
        enable_nlir_gating: bool = False,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # T28 · DIN integrated inside each block. When enabled, a per-block
        # DINInterestExtractor computes a target-aware composite interest
        # vector from this block's encoded sequences, and it is added
        # (with learnable scalar alpha, init 0.1) to every decoded_q_i
        # BEFORE token fusion. Zero effect on RankMixer T-divisibility
        # (no new tokens). When disabled, behaviour is bit-identical to
        # the pre-T28 block.
        self.enable_din_integrated = bool(enable_din_integrated)
        if self.enable_din_integrated:
            self.din_extractor = DINInterestExtractor(
                d_model=d_model,
                num_heads=num_heads,
                num_sequences=num_sequences,
                dropout=dropout,
            )
            self.din_alpha = nn.Parameter(
                torch.full((1,), float(din_integrated_alpha_init)))

        # T33 / ADR-011 · TokenFormer NLIR gating.
        # When enabled, each cross-attention output is modulated by a
        # sigmoid-gated multiplicative transformation using the LN-
        # normalized block input as gate source, implementing
        # TokenFormer §4.4 Eq. 16-18:
        #   G = LN(X) @ W_g
        #   I = X + sigmoid(G) ⊙ A_raw
        # where X = block-input query tokens, LN(X) = the same pre-LN
        # tensor that the cross-attention's Q projection saw, and
        # A_raw = Attn(LN(X), LN(K), V) is the pure attention delta.
        # This keeps gate and attention on the same representation
        # (论文 "same X drives both"), preserves Xavier init's zero-mean
        # unit-variance assumption, and avoids sigmoid saturation risk
        # in deeper blocks where raw X's variance grows. We reuse the
        # CrossAttention's norm_q at forward time — it's the same LN.
        # A single Linear(D, D) is SHARED across all S sequence domains
        # within this block (keeps parameter increment at +D² ≈ 4k per
        # block, does not violate team-memory "per-X independent params"
        # red line). Default off → bit-identical to pre-T33 behaviour.
        self.enable_nlir_gating = bool(enable_nlir_gating)
        if self.enable_nlir_gating:
            self.nlir_gate = nn.Linear(d_model, d_model)
            # Xavier uniform init + bias=0 centers sigmoid(gate) ≈ 0.5
            # at initialization (valid under the LN-normalized gate
            # source assumption · LN outputs mean≈0 var≈1).
            nn.init.xavier_uniform_(self.nlir_gate.weight)
            nn.init.zeros_(self.nlir_gate.bias)

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
            ffn_mode=rank_mixer_ffn_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        target_repr: Optional[torch.Tensor] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            target_repr: (B, D) target-item representation. Required when
                ``enable_din_integrated`` is True; ignored otherwise.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            # T33 · NLIR gating (ADR-011). CrossAttention returns
            # X + A_raw (already residual-connected with pre-LN: A_raw
            # = Attn(LN_q(X), LN_kv(K), V)). We extract pure A_raw by
            # subtracting X, then gate it with sigmoid(LN_q(X) @ W_g)
            # — using the **same LN-normalized query** as the attention
            # input, matching TokenFormer §4.4 Eq. 16 where G is computed
            # from the normalized block input (their backbone is PreNorm
            # RMSNorm). Using raw X instead would (a) violate論文 "same
            # X drives both attn and gate" principle, (b) break Xavier
            # init's zero-mean unit-variance assumption since raw X's
            # variance grows with block depth due to residual accumulation,
            # (c) risk sigmoid saturation in block 2+. LN_q is already
            # instantiated inside CrossAttention so we reuse it for free
            # (zero new params, gradient couples attn and gate naturally).
            # When disabled, the block is bit-identical to pre-T33.
            if self.enable_nlir_gating:
                x_in = q_tokens_list[i]            # (B, Nq, D) — pre-attn input
                attn_delta = decoded_q_i - x_in    # (B, Nq, D) — pure A_raw
                # Reuse attention's pre-LN on Q. This is the EXACT same
                # tensor (LN(X)) that the attention's Q projection saw,
                # keeping gate and attention on the same representation.
                x_norm = self.cross_attns[i].norm_q(x_in)
                gate = torch.sigmoid(self.nlir_gate(x_norm))  # (B, Nq, D)
                decoded_q_i = x_in + gate * attn_delta
            decoded_qs.append(decoded_q_i)

        # 2.5. T28 · DIN integrated interest injection into decoded_q_i.
        # The composite interest vector (B, D) is computed from this
        # block's *just-encoded* next_seqs (not the pre-block seq_tokens),
        # so each block sees a progressively refined interest signal.
        if self.enable_din_integrated:
            if target_repr is None:
                raise ValueError(
                    "MultiSeqHyFormerBlock: target_repr is required when "
                    "enable_din_integrated=True.")
            interest = self.din_extractor(
                target_repr, next_seqs, next_masks)  # (B, D)
            interest_broadcast = interest.unsqueeze(1)  # (B, 1, D)
            decoded_qs = [
                q_i + self.din_alpha * interest_broadcast
                for q_i in decoded_qs
            ]

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# OneTrans: Mixed-Parameterization Unified Attention Block
# ═══════════════════════════════════════════════════════════════════════════════


class OneTransMixedAttn(nn.Module):
    """Mixed-Parameterization Multi-Head Attention for OneTrans.

    Sequence layout: [S_token_0 ... S_token_{L-1} | NS_token_0 ... NS_token_{M-1}]

    Attention mask (True = can attend):
    - S_i  → S_0..S_i (causal), S cannot see NS
    - NS_j → all S (full), NS_0..NS_j (causal within NS)

    Mixed parameterization:
    - S positions: shared W_Q^s, W_K^s, W_V^s, W_O^s
    - Each NS_j:   independent W_Q^j, W_K^j, W_V^j, W_O^j
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_ns = num_ns
        self.dropout = dropout

        # Shared projections for S-tokens
        self.W_Q_s = nn.Linear(d_model, d_model, bias=False)
        self.W_K_s = nn.Linear(d_model, d_model, bias=False)
        self.W_V_s = nn.Linear(d_model, d_model, bias=False)
        self.W_O_s = nn.Linear(d_model, d_model)

        # Per-NS-token independent projections
        self.W_Q_ns = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns)])
        self.W_K_ns = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns)])
        self.W_V_ns = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns)])
        self.W_O_ns = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_ns)])

        # Gating for S output (same pattern as RoPEMultiheadAttention)
        self.W_g_s = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_g_s.weight)
        nn.init.constant_(self.W_g_s.bias, 1.0)

    def _project_ns(
        self,
        ns: torch.Tensor,
        proj_list: nn.ModuleList,
    ) -> torch.Tensor:
        """Apply per-token linear to each NS token independently."""
        # ns: (B, M, D) → (B, M, D)
        parts = [proj_list[j](ns[:, j, :]).unsqueeze(1) for j in range(self.num_ns)]
        return torch.cat(parts, dim=1)  # (B, M, D)

    @staticmethod
    def _build_causal_mixed_mask(
        L: int,
        M: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a bool mask (L+M, L+M) where True = allowed to attend.

        Pattern:
          [ones_LxL     |  zeros_LxM  ]
          [ones_MxL     | causal_MxM  ]

        S tokens use BIDIRECTIONAL attention among themselves (all S can
        see all other S). Cross-domain causality would be arbitrary because
        S tokens from different domains are concatenated in alphabet order,
        not temporal order. NS tokens attend to all S tokens and causally
        to previous NS tokens.
        """
        total = L + M
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        # S-S block: bidirectional (every S sees every S)
        if L > 0:
            mask[:L, :L] = True
        # NS-S block: every NS sees all S
        if M > 0 and L > 0:
            mask[L:, :L] = True
        # NS-NS block: causal (NS_j sees NS_0 ... NS_j)
        if M > 0:
            mask[L:, L:] = torch.tril(
                torch.ones(M, M, dtype=torch.bool, device=device))
        return mask  # (L+M, L+M)

    def forward(
        self,
        s_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        s_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixed-param attention over the unified [S | NS] sequence.

        Args:
            s_tokens: (B, L, D)  – all domain sequence tokens, concatenated.
            ns_tokens: (B, M, D) – NS tokens.
            s_padding_mask: (B, L) bool, True = padding (masked out).

        Returns:
            s_out: (B, L, D)
            ns_out: (B, M, D)
        """
        B, L, D = s_tokens.shape
        M = self.num_ns

        # ---- QKV projections ----
        Q_s = self.W_Q_s(s_tokens)            # (B, L, D)
        K_s = self.W_K_s(s_tokens)
        V_s = self.W_V_s(s_tokens)

        Q_ns = self._project_ns(ns_tokens, self.W_Q_ns)   # (B, M, D)
        K_ns = self._project_ns(ns_tokens, self.W_K_ns)
        V_ns = self._project_ns(ns_tokens, self.W_V_ns)

        # Unified sequences
        Q = torch.cat([Q_s, Q_ns], dim=1)     # (B, L+M, D)
        K = torch.cat([K_s, K_ns], dim=1)
        V = torch.cat([V_s, V_ns], dim=1)

        # ---- Reshape for multi-head ----
        def _to_heads(x: torch.Tensor) -> torch.Tensor:
            Bx, Lx, _ = x.shape
            return x.view(Bx, Lx, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _to_heads(Q)  # (B, H, L+M, head_dim)
        K = _to_heads(K)
        V = _to_heads(V)

        # ---- Attention mask ----
        causal_mixed = self._build_causal_mixed_mask(L, M, s_tokens.device)
        # Expand to (B, 1, L+M, L+M)
        attn_mask = causal_mixed.unsqueeze(0).unsqueeze(0)

        # Incorporate S padding: padded S positions cannot be attended to.
        if s_padding_mask is not None:
            # s_padding_mask: (B, L), True = pad → should NOT be attended to
            # Convert to (B, 1, 1, L+M): False where allowed
            pad_ext = torch.zeros(
                B, 1, 1, L + M, dtype=torch.bool, device=s_tokens.device)
            pad_ext[:, :, :, :L] = s_padding_mask.unsqueeze(1).unsqueeze(2)
            # Combined: allowed = causal_mixed AND NOT padded
            attn_mask = attn_mask & (~pad_ext)

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=dp)
        out = torch.nan_to_num(out, nan=0.0)
        # (B, H, L+M, head_dim) → (B, L+M, D)
        out = out.transpose(1, 2).contiguous().view(B, L + M, D)

        # ---- Split back + mixed output projection ----
        out_s_raw = out[:, :L, :]   # (B, L, D)
        out_ns_raw = out[:, L:, :]  # (B, M, D)

        # S output: shared W_O_s with gating
        G = torch.sigmoid(self.W_g_s(s_tokens))
        s_out = self.W_O_s(out_s_raw) * G  # (B, L, D)

        # NS output: per-token W_O_ns
        ns_parts = [
            self.W_O_ns[j](out_ns_raw[:, j, :]).unsqueeze(1) for j in range(M)
        ]
        ns_out = torch.cat(ns_parts, dim=1)  # (B, M, D)

        return s_out, ns_out


class OneTransMixedFFN(nn.Module):
    """Mixed-Parameterization Feed-Forward Network.

    S-tokens: shared W1, W2.
    Each NS_j: independent W1_j, W2_j.
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = d_model * hidden_mult
        self.num_ns = num_ns

        # Shared FFN for S tokens
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        # Per-NS-token independent FFN
        self.ffn_ns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
            for _ in range(num_ns)
        ])

    def forward(
        self,
        s_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixed FFN independently to S and NS tokens.

        Args:
            s_tokens: (B, L, D)
            ns_tokens: (B, M, D)

        Returns:
            s_out: (B, L, D)
            ns_out: (B, M, D)
        """
        s_out = self.ffn_s(s_tokens)

        ns_parts = [
            self.ffn_ns[j](ns_tokens[:, j, :]).unsqueeze(1)
            for j in range(self.num_ns)
        ]
        ns_out = torch.cat(ns_parts, dim=1)  # (B, M, D)

        return s_out, ns_out


class OneTransNSConditioner(nn.Module):
    """Sequence-conditioned initialization for OneTrans NS tokens.

    Before the unified attention starts, each NS token receives an additive
    update from the mean-pooled domain sequence representations. This gives
    NS tokens initial sequence context analogous to HyFormer's
    MultiSeqQueryGenerator, preventing cold-start of NS → S attention.

    Each NS token has an independent conditioning FFN (mixed-parameterization
    consistent with OneTrans design).
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_sequences: int,
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.num_ns = num_ns
        seq_dim = num_sequences * d_model
        # Pre-LN on the concatenated sequence context
        self.seq_norm = nn.LayerNorm(seq_dim)
        # Per-NS-token independent conditioning FFN
        self.conditioners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_dim, d_model * hidden_mult),
                nn.SiLU(),
                nn.Linear(d_model * hidden_mult, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        compressed_s_list: list,
        compressed_mask_list: list,
    ) -> torch.Tensor:
        """Add sequence-conditioned delta to each NS token.

        Args:
            ns_tokens: (B, M, D)
            compressed_s_list: list of (B, K, D), one per domain.
            compressed_mask_list: list of (B, K) bool, True = padding.

        Returns:
            ns_tokens updated: (B, M, D)
        """
        # Masked mean-pool each domain's compressed S tokens → (B, D) per domain
        domain_means = []
        for s_i, mask_i in zip(compressed_s_list, compressed_mask_list):
            valid = (~mask_i).float().unsqueeze(-1)      # (B, K, 1)
            s_mean = (s_i * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            domain_means.append(s_mean)

        seq_ctx = torch.cat(domain_means, dim=-1)        # (B, num_seq * D)
        seq_ctx = self.seq_norm(seq_ctx)

        # Per-NS-token update
        updates = [self.conditioners[j](seq_ctx).unsqueeze(1)
                   for j in range(self.num_ns)]
        delta = torch.cat(updates, dim=1)                # (B, M, D)

        return ns_tokens + delta


class OneTransBlock(nn.Module):
    """Single OneTrans layer (Pre-LN).

    Layout: [compressed_S_tokens | NS_tokens]
    - Mixed-param attention (bidirectional S, NS sees all S + causal NS)
    - Mixed-param FFN (shared for S, independent per NS)
    - Residual connections + LayerNorm

    The sequence compression (top-K selection per domain) is handled
    externally by ``_gather_onetrans_s_tokens`` in ``PCVRHyFormer``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn_s = nn.LayerNorm(d_model)
        self.norm_attn_ns = nn.LayerNorm(d_model)
        self.attn = OneTransMixedAttn(d_model, num_heads, num_ns, dropout)

        self.norm_ffn_s = nn.LayerNorm(d_model)
        self.norm_ffn_ns = nn.LayerNorm(d_model)
        self.ffn = OneTransMixedFFN(d_model, num_ns, hidden_mult, dropout)

    def forward(
        self,
        s_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        s_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one OneTrans layer with Pre-LN and residuals.

        Args:
            s_tokens: (B, L, D) concatenated sequence tokens from all domains.
            ns_tokens: (B, M, D) NS tokens (user + item).
            s_padding_mask: (B, L) bool, True = padding.

        Returns:
            s_tokens: (B, L, D) updated
            ns_tokens: (B, M, D) updated
        """
        # Pre-LN attention
        s_normed = self.norm_attn_s(s_tokens)
        ns_normed = self.norm_attn_ns(ns_tokens)
        s_attn, ns_attn = self.attn(s_normed, ns_normed, s_padding_mask)

        s_tokens = s_tokens + s_attn
        ns_tokens = ns_tokens + ns_attn

        # Pre-LN FFN
        s_normed = self.norm_ffn_s(s_tokens)
        ns_normed = self.norm_ffn_ns(ns_tokens)
        s_ffn, ns_ffn = self.ffn(s_normed, ns_normed)

        s_tokens = s_tokens + s_ffn
        ns_tokens = ns_tokens + ns_ffn

        return s_tokens, ns_tokens



    """Single OneTrans layer (Pre-LN).

    Layout: [compressed_S_tokens | NS_tokens]
    - Mixed-param attention (causal S, NS sees all S + causal NS)
    - Mixed-param FFN (shared for S, independent per NS)
    - Residual connections + LayerNorm

    The sequence compression (top-K selection per domain) is handled
    externally by ``_gather_onetrans_s_tokens`` in ``PCVRHyFormer``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn_s = nn.LayerNorm(d_model)
        self.norm_attn_ns = nn.LayerNorm(d_model)
        self.attn = OneTransMixedAttn(d_model, num_heads, num_ns, dropout)

        self.norm_ffn_s = nn.LayerNorm(d_model)
        self.norm_ffn_ns = nn.LayerNorm(d_model)
        self.ffn = OneTransMixedFFN(d_model, num_ns, hidden_mult, dropout)

    def forward(
        self,
        s_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        s_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one OneTrans layer with Pre-LN and residuals.

        Args:
            s_tokens: (B, L, D) concatenated sequence tokens from all domains.
            ns_tokens: (B, M, D) NS tokens (user + item).
            s_padding_mask: (B, L) bool, True = padding.

        Returns:
            s_tokens: (B, L, D) updated
            ns_tokens: (B, M, D) updated
        """
        # Pre-LN attention
        s_normed = self.norm_attn_s(s_tokens)
        ns_normed = self.norm_attn_ns(ns_tokens)
        s_attn, ns_attn = self.attn(s_normed, ns_normed, s_padding_mask)

        s_tokens = s_tokens + s_attn
        ns_tokens = ns_tokens + ns_attn

        # Pre-LN FFN
        s_normed = self.norm_ffn_s(s_tokens)
        ns_normed = self.norm_ffn_ns(ns_tokens)
        s_ffn, ns_ffn = self.ffn(s_normed, ns_normed)

        s_tokens = s_tokens + s_ffn
        ns_tokens = ns_tokens + ns_ffn

        return s_tokens, ns_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        enable_fafe: bool = False,
        multi_emb_k: int = 1,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
            enable_fafe: T25 / H2 · Feature-aware Feature Embedding.
                When True, a per-fid ``LayerNorm -> Linear(emb_dim, emb_dim)``
                transform is applied to each raw fid embedding BEFORE
                concatenation + chunking. Each fid gets its own learnable
                transform so the downstream chunked projection sees
                feature-specific signal rather than a shared-weight view.
                Mirrors the "FeatureAware Feature Embedding" step in
                InterFormer (CIKM 2025) + Seed主楼方案2. Adds
                ``num_fids × (emb_dim² + 2·emb_dim)`` dense params (~130k
                for 46 user fids @ emb_dim=64). When False (default),
                behaviour is bit-identical to the pre-T25 tokenizer.
            multi_emb_k: T34 / RankUp Multi-Embedding K. When K=1 (default),
                bit-identical to pre-T34 single embedding table per fid.
                When K>=2, each non-skipped fid has K independent embedding
                tables; their outputs are concatenated and projected via a
                shared ``Linear(K*emb_dim, emb_dim)`` to recover emb_dim.
                Sparse reinit treats all K tables synchronously. RankUp paper
                §4.2 ablation: +0.13~0.21% AUC online (Tencent self).
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.multi_emb_k = int(multi_emb_k)

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        # T34 Multi-Emb: K>=2 produces K tables per fid (wrapped in ModuleList).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            elif self.multi_emb_k == 1:
                # Backward-compat path · single Embedding (bit-identical baseline)
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            else:
                # T34 Multi-Embedding · K independent tables per fid
                # Each table is independently Xavier-initialized via PyTorch default
                embs.append(nn.ModuleList([
                    nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                    for _ in range(self.multi_emb_k)
                ]))
        self.embs = nn.ModuleList([e for e in embs if e is not None])

        # T34 Multi-Emb projection: K*emb_dim → emb_dim
        # Shared across all fids (consistent with RankUp §3.3)
        if self.multi_emb_k >= 2:
            self.multi_emb_proj = nn.Linear(self.multi_emb_k * emb_dim, emb_dim)
        else:
            self.multi_emb_proj = None

        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        # T25 / H2 · FaFE feature-aware feature embedding.
        self.enable_fafe = bool(enable_fafe)
        if self.enable_fafe:
            # Build one transform per fid-slot in the concatenated layout.
            # The layout order is determined by iterating over groups and
            # over fid_idx within each group (see forward()), so the
            # transform list mirrors that exact order.
            fafe_transforms: List[nn.Module] = []
            for group in groups:
                for _ in group:
                    fafe_transforms.append(nn.Sequential(
                        nn.LayerNorm(emb_dim),
                        nn.Linear(emb_dim, emb_dim),
                    ))
            self.fafe_transforms = nn.ModuleList(fafe_transforms)

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        fafe_slot = 0  # tracks position in fafe_transforms (only used if enabled)
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if self.multi_emb_k == 1:
                        # Backward-compat path · single Embedding
                        if length == 1:
                            fid_emb = emb_layer(int_feats[:, offset].long())
                        else:
                            vals = int_feats[:, offset:offset + length].long()
                            emb_all = emb_layer(vals)
                            mask = (vals != 0).float().unsqueeze(-1)
                            count = mask.sum(dim=1).clamp(min=1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count
                    else:
                        # T34 Multi-Emb · K embedding tables · concat then project
                        # emb_layer is nn.ModuleList of K Embedding tables
                        sub_embs = []
                        for k_emb in emb_layer:
                            if length == 1:
                                sub_embs.append(k_emb(int_feats[:, offset].long()))
                            else:
                                vals = int_feats[:, offset:offset + length].long()
                                emb_all = k_emb(vals)
                                mask = (vals != 0).float().unsqueeze(-1)
                                count = mask.sum(dim=1).clamp(min=1)
                                sub_embs.append((emb_all * mask).sum(dim=1) / count)
                        cat_k = torch.cat(sub_embs, dim=-1)  # (B, K*emb_dim)
                        fid_emb = self.multi_emb_proj(cat_k)  # (B, emb_dim)
                # T25 / H2 · FaFE per-fid transform (before concatenation).
                if self.enable_fafe:
                    fid_emb = self.fafe_transforms[fafe_slot](fid_emb)
                fafe_slot += 1
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        # T19 / ADR-007: FFN parameterization inside RankMixerBlock.
        #   'shared'    = one FFN shared across T tokens (legacy, default).
        #   'per_token' = independent FFN weights per token (RankMixer paper
        #                 original definition). ~T× dense params per block.
        rank_mixer_ffn_mode: str = 'shared',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # OneTrans unified attention (replaces HyFormer query-decoding blocks)
        use_onetrans: bool = False,
        onetrans_top_k: int = 50,
        # T13: hash embedding for sequence features with vocab > emb_skip_threshold
        # (0 = disabled, use zero vector for skipped features as before)
        seq_hash_vocab: int = 0,
        # Item-conditioned query generation (DIN-style target-item awareness)
        item_conditioned_query: bool = False,
        # T32 / DECEM Trick 6 · Q_init=MLP(item_emb)+NS_residual.
        # Replaces base query path in MultiSeqQueryGenerator from global_info FFN
        # to item_repr FFN with NS context residual projection. Compatible with
        # item_conditioned_query (ICQ mix gate stays in outer layer).
        # Default False = no behaviour change (bit-identical baseline).
        q_init_item: bool = False,
        # T34 / RankUp Multi-Embedding K=2 · ADR-012.
        # When K>=2, each non-skipped fid in NS tokenizer has K independent
        # embedding tables (concat + projection). Compatible with all ns
        # tokenizer flags. Default 1 = bit-identical baseline.
        multi_emb_k: int = 1,
        # ADR-004 · Target-item hard interaction injected into seq tokens
        # (seq-token level target-item conditioning, orthogonal to ICQ which
        # is query-side). Default 'off' keeps behaviour identical to baseline.
        target_item_seq_injection: str = 'off',
        target_inject_alpha_init: float = 0.0,
        # Dense bypass: project dense features directly to the classifier input,
        # bypassing NS-token dilution in HyFormer blocks. When enabled the dense
        # token is removed from ns_tokens and the classifier input dimension is
        # extended by d_model (d_model + d_model = 2*d_model).
        enable_dense_bypass: bool = False,
        # T25 / G1 · Enable DIN-style target-aware interest pooling. When
        # True, a DINInterestExtractor runs after the seq encoders and
        # produces a composite interest vector (B, D) that is concatenated
        # to the backbone output before the classifier (mirrors the
        # dense_bypass integration pattern). Zero effect on RankMixer token
        # count / divisibility constraint.
        enable_din_interest: bool = False,
        din_interest_source: str = 'raw',
        din_interest_merge: str = 'compact',
        # T41 · TIN-lite target-aware temporal interest. Replaces the
        # classifier-time DIN extractor with a shape-compatible temporal
        # variant when enable_din_interest=True.
        enable_tin_interest: bool = False,
        tin_time_alpha_init: float = 1.0,
        # T28 · DIN integrated INSIDE each HyFormer block. Each block
        # computes its own composite target-aware interest vector from its
        # just-encoded sequences and injects it (with a per-block learnable
        # scalar alpha, init 0.1) into every decoded_q_i BEFORE token
        # fusion into the RankMixer. Unlike T25 bypass (which injects a
        # single interest vector at classifier-time, outside the token
        # mixing), T28 lets each block's RankMixer see target-aware
        # queries and propagate through subsequent blocks. Orthogonal to
        # T25 (can be combined) but the first ablation keeps T28 on and
        # T25 off to avoid attribution ambiguity. Adds
        # ~num_blocks * (4 * CrossAttn + merge Linear) Dense params, i.e.
        # ~208k at num_blocks=2 / num_sequences=4 / d_model=64.
        enable_din_integrated: bool = False,
        din_integrated_alpha_init: float = 0.1,
        # T33 / ADR-011 · TokenFormer NLIR gating on cross-attention output.
        # When True, a shared Linear(D, D) per HyFormer block gates the
        # pure cross-attention delta with sigmoid(X @ W_g), implementing
        # TokenFormer §4.4 Eq. 16-18. Acts as multiplicative non-linearity
        # that mitigates Sequential Collapse Propagation. Adds ~D² params
        # per block (≈ 4k per block at d_model=64, total ~8k for 2 blocks).
        # Shared across all S sequence domains within a block (does NOT
        # violate team-memory per-X independent params red line).
        # Orthogonal to T25 bypass (classifier-time injection), T34
        # Multi-Embedding (embedding-side), T30/T31 (dataset-side).
        enable_nlir_gating: bool = False,
        # T25 / H2 · Feature-aware Feature Embedding. When True, each fid's
        # raw embedding passes through a per-fid LayerNorm+Linear transform
        # (in the RankMixer NS tokenizer only, as an MVP) before chunked
        # concatenation. Shifts the representation from "shared-weight
        # chunk projection" to "feature-specific pre-transform + shared
        # projection". Adds ~130k-250k dense params. Zero effect when
        # ns_tokenizer_type != 'rankmixer' (group tokenizer path).
        enable_fafe: bool = False,
        # T34 / EXP-049 · DCN-V2 cross-feature bypass. Builds an explicit
        # outer-product cross signal from selected user_int × item_int
        # scalar fids (top correlations from EDA-correlation 2026-05-13)
        # and concatenates a (B, D) vector to the classifier input — same
        # bypass pattern as T25 dense / DIN. Orthogonal to all
        # attention paths; default off (zero param overhead). Adds
        # ~42k dense params at default config. Embeddings shared with
        # NS tokenizer (DCN-paper convention; same fid representation,
        # different interaction view).
        # 'user_fids' / 'item_fids' are SCHEMA fid numbers (the X in
        # 'user_int_feats_X'), NOT list indices. PCVRHyFormer maps
        # them to indices internally via user_int_feature_specs offset
        # ordering.
        enable_dcn_cross: bool = False,
        dcn_cross_user_fids: Optional[List[int]] = None,
        dcn_cross_item_fids: Optional[List[int]] = None,
        dcn_cross_layers: int = 2,
        # T36 / RankUp Global Token (ADR-013 · Plan B). When True, the
        # user_dense token slot is repurposed as a Global Token: instead
        # of projecting only `user_dense_feats`, the projection input
        # concatenates [user_dense_feats, mean_pool(user_ns),
        # mean_pool(item_ns), concat(mean_pool(seq_d) for d in domains)]
        # → Linear+LN+SiLU → 1 NS token. This preserves T-divisibility
        # (num_ns unchanged) while injecting an aggregated global view per
        # RankUp §3.4. Requires `user_dense_dim > 0` (the slot must exist).
        # Default False = bit-identical baseline.
        enable_global_token: bool = False,
        # T37 / Path B Step 1 · UE 分离 (ADR-014 · 5/16 修订版).
        # user_dense_feats 默认 988-dim · 含 fid_61 (256-dim 预归一化 UE
        # embedding · production user pre-trained) + count features
        # (fid_62~66) + 其他 embedding (fid_87/89/90/91). 当前
        # user_dense_proj 把全部 988-dim 一起 Linear → 1 token · UE 信号
        # 被 count features 稀释 (EXP-011 实证 scale gap 4~5 量级).
        # enable_ue_split: 把 fid_61 单独成 UE token (复用 user_dense
        # slot) · 其他 dense feats 走 dense_bypass 路径 (T25 已 work).
        # 自动启用 enable_dense_bypass=True (其他 dense 必须有 path)。
        # Plan B 复用 slot · T-divisibility 不变。
        # ue_offset / ue_dim: train.py 由 user_dense_schema 计算 fid_61
        # 在 user_dense_feats tensor 中的 (offset, length)。
        # Default False = bit-identical baseline.
        enable_ue_split: bool = False,
        ue_offset: int = 0,
        ue_dim: int = 256,
        ue_slices: Optional[List[Tuple[int, int]]] = None,
        ue_split_separate_tokens: bool = False,
        # T39 / M55 · enable_ue_int_bilinear: gate the "other" dense feats
        # by a Linear projection of the pooled user_int representation
        # before the dense_bypass projection. Implements Seed 主楼 5/5
        # 第 (1) 条后半句 "其他跟 int pair 加权". Mechanism differs from
        # DCN-V2 (M23 LB −0.011 dead): single-layer multiplicative gating
        # (not recursive cross) · OOV-friendly because it reweights
        # existing dense features by user_int signal rather than learning
        # train-internal user × item co-occurrence.
        enable_ue_int_bilinear: bool = False,
        ue_int_bilinear_alpha_init: float = 0.5,
        # T40 · Explicit UE x target-item interaction bypass. Requires
        # enable_ue_split so the UE vector is well-defined. Projects
        # [UE, item_repr, UE * item_repr] to d_model and appends it to the
        # classifier input.
        enable_ue_item_interaction: bool = False,
        ue_item_interaction_alpha_init: float = 1.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.enable_dense_bypass = enable_dense_bypass
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.rank_mixer_ffn_mode = rank_mixer_ffn_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_onetrans = use_onetrans
        self.onetrans_top_k = onetrans_top_k
        self.seq_hash_vocab = seq_hash_vocab
        self.target_item_seq_injection = target_item_seq_injection
        self.target_inject_alpha_init = target_inject_alpha_init

        # ================== NS Tokens Construction ==================

        # T25 / H2 · FaFE flag is stored ahead of tokenizer construction
        # so the rankmixer tokenizer can read it.
        self.enable_fafe = bool(enable_fafe)

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                enable_fafe=self.enable_fafe,
                multi_emb_k=multi_emb_k,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                enable_fafe=self.enable_fafe,
                multi_emb_k=multi_emb_k,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # Save for downstream modules that need to slice item NS tokens out
        # of the concatenated ns_tokens tensor (e.g. ICQ, target injector).
        self.num_item_ns = num_item_ns
        self.num_user_ns = num_user_ns

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        # T36 / ADR-013 · Global Token (Plan B): when enabled, the
        # user_dense token slot becomes an aggregated global view.
        self.enable_global_token = bool(enable_global_token)
        if self.enable_global_token and not self.has_user_dense:
            raise ValueError(
                "enable_global_token=True requires user_dense_dim > 0 "
                "(Plan B reuses the user_dense token slot)."
            )
        # T37 / ADR-014 · UE split: when enabled · user_dense slot becomes
        # UE token (only fid_61 256-dim) · 其他 dense 走 dense_bypass。
        # 与 enable_global_token 互斥 (都复用 user_dense slot)。
        self.enable_ue_split = bool(enable_ue_split)
        self.ue_split_separate_tokens = bool(ue_split_separate_tokens)
        if self.enable_ue_split and self.enable_global_token:
            raise ValueError(
                "enable_ue_split and enable_global_token both reuse the "
                "user_dense slot · cannot be combined")
        if self.enable_ue_split and not self.has_user_dense:
            raise ValueError(
                "enable_ue_split=True requires user_dense_dim > 0")
        if self.enable_ue_split:
            raw_slices = (
                list(ue_slices)
                if ue_slices is not None
                else [(int(ue_offset), int(ue_dim))]
            )
            if not raw_slices:
                raise ValueError("enable_ue_split=True requires non-empty ue_slices")
            norm_slices: List[Tuple[int, int]] = []
            for offset_raw, dim_raw in raw_slices:
                offset_i = int(offset_raw)
                dim_i = int(dim_raw)
                if dim_i <= 0 or dim_i > user_dense_dim:
                    raise ValueError(
                        f"enable_ue_split: slice dim {dim_i} must be in "
                        f"(0, user_dense_dim={user_dense_dim}]")
                if offset_i < 0 or offset_i + dim_i > user_dense_dim:
                    raise ValueError(
                        f"enable_ue_split: slice offset {offset_i} + dim "
                        f"{dim_i} > user_dense_dim {user_dense_dim}")
                norm_slices.append((offset_i, dim_i))
            sorted_slices = sorted(norm_slices, key=lambda x: x[0])
            last_end = 0
            for offset_i, dim_i in sorted_slices:
                if offset_i < last_end:
                    raise ValueError(
                        f"enable_ue_split: overlapping ue_slices {sorted_slices}")
                last_end = offset_i + dim_i
            self.ue_slices = sorted_slices
            self.ue_offset = int(sorted_slices[0][0])
            self.ue_dim = int(sum(dim for _, dim in sorted_slices))
        else:
            self.ue_slices = []
            self.ue_offset = 0
            self.ue_dim = 0
        if self.ue_split_separate_tokens and not self.enable_ue_split:
            raise ValueError(
                "ue_split_separate_tokens requires enable_ue_split=True")
        self.user_dense_ns_tokens = (
            len(self.ue_slices)
            if self.enable_ue_split and self.ue_split_separate_tokens
            else (1 if self.has_user_dense else 0)
        )

        if self.has_user_dense:
            if self.enable_global_token:
                # Concat input: [user_dense, user_int_pool, item_int_pool,
                #                seq_pool_a, ..., seq_pool_d]
                global_in_dim = (
                    user_dense_dim
                    + d_model
                    + d_model
                    + self.num_sequences * d_model
                )
                self.global_token_proj = nn.Sequential(
                    nn.Linear(global_in_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.user_dense_proj = None
                self.ue_proj = None
            elif self.enable_ue_split:
                # T37/T40 · UE token = one or more selected user_dense fids.
                if self.ue_split_separate_tokens:
                    self.ue_proj = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(dim_i, d_model),
                            nn.LayerNorm(d_model),
                        )
                        for _, dim_i in self.ue_slices
                    ])
                else:
                    self.ue_proj = nn.Sequential(
                        nn.Linear(self.ue_dim, d_model),
                        nn.LayerNorm(d_model),
                    )
                self.user_dense_proj = None
                self.global_token_proj = None
            else:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.global_token_proj = None
                self.ue_proj = None
        else:
            self.user_dense_proj = None
            self.global_token_proj = None
            self.ue_proj = None

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count: bypass mode keeps the dense NS token so that
        # the rank_mixer T-divisibility constraint (d_model % T == 0) is
        # preserved. The bypass projection adds a SECOND, undiluted path for
        # the dense features directly into the classifier — they flow through
        # both the NS-token path and the bypass path simultaneously.
        self.num_ns = (num_user_ns + self.user_dense_ns_tokens
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # Dense bypass projections: an extra projection added ALONGSIDE the
        # NS-token path when enable_dense_bypass=True. The projected vector is
        # concatenated directly to the output of output_proj before the
        # classifier, giving the dense features a full-strength signal path
        # independent of the HyFormer block dilution.
        # T37 / ADR-014 · enable_ue_split forces dense_bypass for "other"
        # dense features (everything except the UE slice). Without bypass,
        # the other dense feats would have no path into the model since
        # user_dense_proj is replaced by ue_proj (UE-only).
        self.has_dense_bypass = (
            (enable_dense_bypass or self.enable_ue_split)
            and self.has_user_dense
        )
        if self.has_dense_bypass:
            # Bypass input dim: when ue_split, only the "other" dense feats
            # (user_dense_dim - sum(ue_slice_dims)). Otherwise full
            # user_dense_dim.
            bypass_in_dim = (user_dense_dim - self.ue_dim
                             if self.enable_ue_split else user_dense_dim)
            self.user_dense_bypass_proj = nn.Sequential(
                nn.Linear(bypass_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )

        # T39 / M55 · UE × user_int Bilinear gating. Only effective when
        # has_dense_bypass=True (other dense feats route through bypass).
        # Gates the "other" dense feats by a Linear projection of pooled
        # user_int representation: gated = other_dense * (1 + alpha *
        # tanh(Linear(user_int_pool))). The (1 + ...) form ensures
        # bit-identical behavior at alpha=0 (back-compat) and the tanh
        # caps the gating in [-alpha, +alpha] · default alpha=0.5 so
        # gating is in [0.5, 1.5] (mild reweighting).
        self.enable_ue_int_bilinear = bool(enable_ue_int_bilinear)
        if self.enable_ue_int_bilinear and not self.has_dense_bypass:
            raise ValueError(
                "enable_ue_int_bilinear requires has_dense_bypass=True "
                "(set enable_dense_bypass=True or enable_ue_split=True)")
        if self.enable_ue_int_bilinear:
            bilinear_in_dim = d_model  # pooled user_ns is (B, D)
            bypass_dense_dim = (user_dense_dim - self.ue_dim
                                if self.enable_ue_split else user_dense_dim)
            self.ue_int_bilinear_proj = nn.Linear(
                bilinear_in_dim, bypass_dense_dim)
            # learnable alpha · scalar · init from CLI · clamped via
            # tanh in forward so alpha can grow without exploding.
            self.ue_int_bilinear_alpha = nn.Parameter(
                torch.tensor(float(ue_int_bilinear_alpha_init)))
            # init bilinear_proj to zeros so initial gating ≈ 1 + 0 = 1
            # (bit-identical to non-bilinear path at step 0 · learns
            # gating gradually as user_int signal becomes useful).
            nn.init.zeros_(self.ue_int_bilinear_proj.weight)
            nn.init.zeros_(self.ue_int_bilinear_proj.bias)

        # T40 · Explicit UE x target-item interaction bypass. This is kept
        # outside the token mixer so it does not disturb RankMixer's
        # d_model % T divisibility constraint.
        self.enable_ue_item_interaction = bool(enable_ue_item_interaction)
        if self.enable_ue_item_interaction and not self.enable_ue_split:
            raise ValueError(
                "enable_ue_item_interaction requires enable_ue_split=True")
        if self.enable_ue_item_interaction and self.num_item_ns <= 0:
            raise ValueError(
                "enable_ue_item_interaction requires at least one item NS token")
        if self.enable_ue_item_interaction:
            self.ue_item_interaction_proj = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.ue_item_interaction_alpha = nn.Parameter(
                torch.tensor(float(ue_item_interaction_alpha_init)))

        # T25 / G1 · DIN-style target-aware interest pooling (bypass mode).
        self.enable_din_interest = bool(enable_din_interest)
        self.enable_tin_interest = bool(enable_tin_interest)
        self.din_interest_source = str(din_interest_source)
        self.din_interest_merge = str(din_interest_merge)
        if self.din_interest_source not in ('raw', 'encoded'):
            raise ValueError(
                "din_interest_source must be 'raw' or 'encoded', "
                f"got {self.din_interest_source!r}")
        if self.din_interest_merge not in ('compact', 'per_domain'):
            raise ValueError(
                "din_interest_merge must be 'compact' or 'per_domain', "
                f"got {self.din_interest_merge!r}")
        if (self.enable_din_interest and self.use_onetrans
                and self.din_interest_source == 'encoded'):
            raise ValueError(
                "din_interest_source='encoded' is only supported on the "
                "HyFormer path; use 'raw' with --use_onetrans.")
        if self.enable_tin_interest and not self.enable_din_interest:
            raise ValueError(
                "enable_tin_interest requires enable_din_interest=True; "
                "TIN-lite is a drop-in temporal upgrade to the DIN bypass.")
        if self.enable_tin_interest and self.din_interest_source != 'raw':
            raise ValueError(
                "enable_tin_interest currently requires "
                "din_interest_source='raw' so seq_time_buckets align with "
                "the sequence tokens.")
        if self.enable_tin_interest and self.num_time_buckets <= 0:
            raise ValueError(
                "enable_tin_interest requires use_time_buckets=True")
        self.din_interest_out_dim = (
            d_model * self.num_sequences
            if self.din_interest_merge == 'per_domain' else d_model
        )
        if self.enable_din_interest:
            if self.enable_tin_interest:
                self.din_extractor = TINLiteInterestExtractor(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_sequences=self.num_sequences,
                    num_time_buckets=self.num_time_buckets,
                    dropout=dropout_rate,
                    merge_mode=self.din_interest_merge,
                    time_alpha_init=tin_time_alpha_init,
                )
            else:
                self.din_extractor = DINInterestExtractor(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_sequences=self.num_sequences,
                    dropout=dropout_rate,
                    merge_mode=self.din_interest_merge,
                )

        # T28 · DIN integrated inside each HyFormer block. The flag is
        # stored here so self.blocks construction (below) can read it;
        # the per-block DINInterestExtractor is created inside
        # MultiSeqHyFormerBlock itself. din_integrated_alpha_init is
        # forwarded as-is (each block gets an independent learnable
        # alpha initialized to this scalar).
        self.enable_din_integrated = bool(enable_din_integrated)
        self.din_integrated_alpha_init = float(din_integrated_alpha_init)

        # T33 · NLIR gating flag propagated to every MultiSeqHyFormerBlock.
        self.enable_nlir_gating = bool(enable_nlir_gating)

        # T34 / EXP-049 · DCN-V2 cross-feature bypass. The actual module
        # is constructed AFTER the NS tokenizers (which own the embedding
        # tables we share) — see further down in __init__. We only stash
        # the flags here so the classifier-dim computation below can
        # account for the (B, D) bypass vector.
        self.enable_dcn_cross = bool(enable_dcn_cross)
        self._dcn_cross_user_fids_arg = dcn_cross_user_fids
        self._dcn_cross_item_fids_arg = dcn_cross_item_fids
        self._dcn_cross_layers_arg = int(dcn_cross_layers)

        # T34 · DCN cross feature bypass module. Constructed here (after
        # NS tokenizers — they own the embedding tables we share) and
        # BEFORE classifier dim computation (so its slot can be counted).
        #
        # ``dcn_cross_user_fids`` / ``dcn_cross_item_fids`` are LIST
        # INDICES into ``user_int_feature_specs`` / ``item_int_feature_specs``
        # (NOT raw schema fid numbers — train.py / infer.py is
        # responsible for mapping ``--dcn_cross_user_fids 1 49`` (raw
        # fid numbers) → list indices via
        # ``user_int_schema.entries`` lookup before building model_args.
        # This keeps the model API typed and decoupled from schema
        # ordering, and lets train.py / infer.py validate fid
        # availability with a clear error message.
        self.dcn_cross: Optional[DCNCrossBypass] = None
        if self.enable_dcn_cross:
            user_fid_indices = list(self._dcn_cross_user_fids_arg or [])
            item_fid_indices = list(self._dcn_cross_item_fids_arg or [])
            self.dcn_cross = DCNCrossBypass(
                d_model=d_model,
                emb_dim=emb_dim,
                user_int_feature_specs=user_int_feature_specs,
                item_int_feature_specs=item_int_feature_specs,
                user_fids=user_fid_indices,
                item_fids=item_fid_indices,
                num_cross_layers=self._dcn_cross_layers_arg,
            )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0 and not use_onetrans:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list for sequence domain features.

            Features with vocab_size <= 0 → None (zero vector).
            Features with vocab_size > emb_skip_threshold → None unless
            seq_hash_vocab > 0, in which case a hash embedding of size
            seq_hash_vocab is created and values are hashed modulo
            seq_hash_vocab at forward time (T13 fix for domain_c_seq_47).
            """
            embs_raw = []
            hash_flags = []
            for vs in vocab_sizes:
                if int(vs) <= 0:
                    embs_raw.append(None)
                    hash_flags.append(False)
                elif emb_skip_threshold > 0 and int(vs) > emb_skip_threshold:
                    if seq_hash_vocab > 0:
                        # Hash embedding: recover high-vocab item IDs via modulo
                        embs_raw.append(nn.Embedding(
                            int(seq_hash_vocab) + 1, emb_dim, padding_idx=0))
                        hash_flags.append(True)
                    else:
                        embs_raw.append(None)
                        hash_flags.append(False)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                    hash_flags.append(False)
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id, hash_flags

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_hash_flags = {}   # domain -> hash_flags (True = modulo hash)
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id, hash_flags = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_hash_flags[domain] = hash_flags
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer OR OneTrans Components ==================
        # Target-item seq injector (ADR-004) — applied to seq_tokens_list
        # BEFORE the HyFormer / OneTrans path, so it benefits both. When
        # target_item_seq_injection='off' this is a zero-param identity.
        self.target_item_seq_injector = TargetItemSeqInjector(
            d_model=d_model,
            num_sequences=self.num_sequences,
            mode=target_item_seq_injection,
            alpha_init=target_inject_alpha_init,
        )

        if not use_onetrans:
            # ---- Classic HyFormer path ----
            self.query_generator = MultiSeqQueryGenerator(
                d_model=d_model,
                num_ns=self.num_ns,
                num_queries=num_queries,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
                item_conditioned=item_conditioned_query,
                num_item_ns=num_item_ns,
                q_init_item=q_init_item,
            )
            self.blocks = nn.ModuleList([
                MultiSeqHyFormerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_queries=num_queries,
                    num_ns=self.num_ns,
                    num_sequences=self.num_sequences,
                    seq_encoder_type=seq_encoder_type,
                    hidden_mult=hidden_mult,
                    dropout=dropout_rate,
                    top_k=seq_top_k,
                    causal=seq_causal,
                    rank_mixer_mode=rank_mixer_mode,
                    rank_mixer_ffn_mode=rank_mixer_ffn_mode,
                    enable_din_integrated=self.enable_din_integrated,
                    din_integrated_alpha_init=din_integrated_alpha_init,
                    enable_nlir_gating=self.enable_nlir_gating,
                )
                for _ in range(num_hyformer_blocks)
            ])
            # Output: concat Q tokens from all domains → project to d_model
            self.output_proj = nn.Sequential(
                nn.Linear(num_queries * self.num_sequences * d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            # ---- OneTrans path ----
            # Per-domain lightweight SwiGLU encoder (processes sequences before
            # top-K compression; shared across OneTrans blocks for efficiency)
            self.ot_seq_encoders = nn.ModuleDict({
                domain: SwiGLUEncoder(d_model, hidden_mult, dropout_rate)
                for domain in self.seq_domains
            })
            # Stack of OneTrans blocks
            self.onetrans_blocks = nn.ModuleList([
                OneTransBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_ns=self.num_ns,
                    hidden_mult=hidden_mult,
                    dropout=dropout_rate,
                )
                for _ in range(num_hyformer_blocks)
            ])
            # NS-token sequence conditioner: gives NS tokens initial sequence
            # context before the first unified attention (prevents cold-start)
            self.ot_ns_conditioner = OneTransNSConditioner(
                d_model=d_model,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                hidden_mult=2,
            )
            # Output: NS tokens flattened → project to d_model
            self.output_proj = nn.Sequential(
                nn.Linear(self.num_ns * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            # No query_generator or HyFormer blocks needed
            self.query_generator = None
            self.blocks = nn.ModuleList()

        # ================== RoPE (HyFormer path only) ==================
        if use_rope and not use_onetrans:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier input dimension: d_model normally; d_model*2 when dense
        # bypass is active because the bypass vector is concatenated to output;
        # plus the selected DIN interest width when active; plus d_model
        # when T34 DCN cross bypass is active (same pattern).
        clsfier_in_dim = (
            d_model
            + (d_model if self.has_dense_bypass else 0)
            + (self.din_interest_out_dim if self.enable_din_interest else 0)
            + (d_model if self.enable_dcn_cross else 0)
            + (d_model if self.enable_ue_item_interaction else 0)
        )

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(clsfier_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                if isinstance(emb, nn.ModuleList):
                    # T34 Multi-Emb · init all K tables
                    for k_emb in emb:
                        nn.init.xavier_normal_(k_emb.weight.data)
                        k_emb.weight.data[0, :] = 0
                else:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    if isinstance(emb, nn.ModuleList):
                        # T34 Multi-Emb · K embedding tables · synchronously reinit all
                        for k_emb in emb:
                            nn.init.xavier_normal_(k_emb.weight.data)
                            k_emb.weight.data[0, :] = 0
                            reinit_ptrs.add(k_emb.weight.data_ptr())
                            reinit_count += 1
                    else:
                        nn.init.xavier_normal_(emb.weight.data)
                        emb.weight.data[0, :] = 0
                        reinit_ptrs.add(emb.weight.data_ptr())
                        reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        hash_flags: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                vals = seq[:, i, :]  # (B, L)
                # Hash embedding: map high-vocab item IDs via modulo
                if hash_flags is not None and i < len(hash_flags) and hash_flags[i]:
                    # Non-zero values get hashed; zero (padding) stays 0
                    nonzero = vals != 0
                    vals = vals.clone()
                    vals[nonzero] = (vals[nonzero] % self.seq_hash_vocab).clamp(min=1)
                e = emb(vals)  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _gather_onetrans_topk(
        self,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take the last top_k valid tokens per sample (like LongerEncoder).

        Args:
            seq_tokens: (B, L, D)
            padding_mask: (B, L) True = padding

        Returns:
            topk_tokens: (B, K, D)
            topk_mask: (B, K) True = padding
        """
        B, L, D = seq_tokens.shape
        K = min(self.onetrans_top_k, L)
        valid_len = (~padding_mask).sum(dim=1)  # (B,)
        actual_k = valid_len.clamp(max=K)
        start_pos = (valid_len - actual_k).clamp(min=0)

        device = seq_tokens.device
        offsets = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
        indices = (start_pos.unsqueeze(1) + offsets).clamp(0, L - 1)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        topk_tokens = torch.gather(seq_tokens, 1, indices_exp)  # (B, K, D)

        pos_idx = torch.arange(K, device=device).unsqueeze(0)
        topk_mask = pos_idx >= actual_k.unsqueeze(1)  # (B, K) True = pad
        topk_tokens = topk_tokens * (~topk_mask).unsqueeze(-1).float()
        return topk_tokens, topk_mask

    def _run_onetrans_blocks(
        self,
        seq_tokens_list: list,
        seq_masks_list: list,
        ns_tokens: torch.Tensor,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        """OneTrans forward path.

        1. Per-domain lightweight SwiGLU encoding.
        2. Top-K compression per domain.
        3. Concatenate all domains into one S-token sequence.
        4. Stack of OneTransBlock layers.
        5. Flatten NS tokens → output_proj → d_model.
        """
        if apply_dropout:
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        # Per-domain SwiGLU encoding (within-domain local context)
        compressed_s = []
        compressed_masks = []
        for i, domain in enumerate(self.seq_domains):
            enc_out, mask_out = self.ot_seq_encoders[domain](
                seq_tokens_list[i], seq_masks_list[i])
            topk_tok, topk_mask = self._gather_onetrans_topk(enc_out, mask_out)
            compressed_s.append(topk_tok)
            compressed_masks.append(topk_mask)

        # Concatenate all domains into unified S sequence
        s_tokens = torch.cat(compressed_s, dim=1)          # (B, K*S, D)
        s_mask = torch.cat(compressed_masks, dim=1)         # (B, K*S) True=pad

        # Stack OneTrans blocks
        # Condition NS tokens on sequence context before the first block
        ns_tokens = self.ot_ns_conditioner(ns_tokens, compressed_s, compressed_masks)

        for block in self.onetrans_blocks:
            s_tokens, ns_tokens = block(s_tokens, ns_tokens, s_mask)

        # Output: NS tokens (final representations) → project to d_model
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)          # (B, M*D)
        output = self.output_proj(ns_flat)        # (B, D)
        return output

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        target_repr: Optional[torch.Tensor] = None,
        return_seq_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]]:
        """Runs the HyFormer multi-sequence block stack.

        Args:
            q_tokens_list / ns_tokens / seq_tokens_list / seq_masks_list:
                standard block inputs.
            apply_dropout: whether to apply emb_dropout (train path only).
            target_repr: (B, D) target-item representation, required when
                ``enable_din_integrated`` is True (each block uses it).
            return_seq_states: when True, also return the final encoded
                sequence tensors and masks for late classifier-time DIN.
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                target_repr=target_repr,
            )

        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)
        output = all_q.view(B, -1)
        output = self.output_proj(output)
        if return_seq_states:
            return output, curr_seqs, curr_masks
        return output

    def _build_ns_tokens(
        self,
        inputs: ModelInput,
        seq_tokens_list: Optional[List[torch.Tensor]] = None,
        seq_masks_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Build NS token tensor from user/item int and dense features.

        When ``enable_dense_bypass`` is True, user dense features are included
        here as a normal NS token (preserving rank_mixer T-divisibility) AND
        also projected via ``_apply_dense_bypass`` directly to the classifier
        input. The two paths are complementary: NS gives contextual integration
        through HyFormer, bypass gives a full-strength undiluted signal.

        T36 / ADR-013 · When ``enable_global_token`` is True, the user_dense
        slot is replaced with an aggregated Global Token built from
        ``[user_dense_feats, mean(user_ns), mean(item_ns), mean(seq_*)]``.
        Requires ``seq_tokens_list`` and ``seq_masks_list`` to be passed in.
        """
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)
        ns_parts = [user_ns]
        if self.has_user_dense:
            if self.enable_global_token:
                if seq_tokens_list is None or seq_masks_list is None:
                    raise RuntimeError(
                        "Global Token enabled but seq tokens not provided to "
                        "_build_ns_tokens. forward() must compute seq tokens "
                        "first when enable_global_token=True."
                    )
                global_token = self._build_global_token(
                    inputs.user_dense_feats, user_ns, item_ns,
                    seq_tokens_list, seq_masks_list,
                )
                ns_parts.append(global_token)
            elif self.enable_ue_split:
                # T37/T40/T42 · UE token(s) = selected dense embedding slice(s).
                # 其他 dense feats 不进 ns · 走 dense_bypass (T25 路径).
                ns_parts.append(
                    self._project_ue_dense_tokens(inputs.user_dense_feats))
            else:
                ns_parts.append(
                    F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            ns_parts.append(
                F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1))
        return torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

    def _build_global_token(
        self,
        user_dense_feats: torch.Tensor,
        user_ns: torch.Tensor,
        item_ns: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Global Token (T36 / ADR-013 Plan B).

        Returns a (B, 1, d_model) tensor that occupies the user_dense slot
        in ns_tokens. Aggregates user/item NS tokens and per-domain mean-
        pooled raw sequence embeddings via a shared Linear+LN+SiLU.
        """
        user_int_pool = user_ns.mean(dim=1)  # (B, D)
        item_int_pool = item_ns.mean(dim=1)  # (B, D)
        seq_pools: List[torch.Tensor] = []
        for seq_tok, seq_mask in zip(seq_tokens_list, seq_masks_list):
            # seq_mask: True = padding (per _make_padding_mask). Take the
            # mean over valid positions only; clamp denominator to avoid
            # division by zero on all-padding rows (treat as zero pool).
            valid = (~seq_mask).to(seq_tok.dtype).unsqueeze(-1)  # (B, L, 1)
            denom = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
            pool = (seq_tok * valid).sum(dim=1) / denom  # (B, D)
            seq_pools.append(pool)
        seq_pool_cat = torch.cat(seq_pools, dim=-1)  # (B, num_sequences * D)
        global_concat = torch.cat(
            [user_dense_feats, user_int_pool, item_int_pool, seq_pool_cat],
            dim=-1,
        )  # (B, user_dense_dim + 2D + num_sequences * D)
        return F.silu(self.global_token_proj(global_concat)).unsqueeze(1)  # (B, 1, D)

    def _slice_ue_dense(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        """Return the selected UE dense slice(s) concatenated by offset order."""
        if not self.ue_slices:
            raise RuntimeError("_slice_ue_dense called without ue_slices")
        parts = [
            user_dense_feats[:, offset:offset + dim]
            for offset, dim in self.ue_slices
        ]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def _slice_other_dense(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        """Return dense features excluding the selected UE slice(s)."""
        if not self.ue_slices:
            return user_dense_feats
        parts: List[torch.Tensor] = []
        cursor = 0
        total_dim = user_dense_feats.shape[-1]
        for offset, dim in self.ue_slices:
            if cursor < offset:
                parts.append(user_dense_feats[:, cursor:offset])
            cursor = offset + dim
        if cursor < total_dim:
            parts.append(user_dense_feats[:, cursor:total_dim])
        if not parts:
            return user_dense_feats.new_zeros((user_dense_feats.shape[0], 0))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def _project_ue_dense(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        """Project selected UE dense slice(s) to the model dimension."""
        if self.ue_split_separate_tokens:
            return self._project_ue_dense_tokens(user_dense_feats).mean(dim=1)
        ue_feats = self._slice_ue_dense(user_dense_feats)
        return F.silu(self.ue_proj(ue_feats))

    def _project_ue_dense_tokens(
        self, user_dense_feats: torch.Tensor
    ) -> torch.Tensor:
        """Project selected UE dense slice(s) as one or more NS tokens."""
        if not self.ue_split_separate_tokens:
            return self._project_ue_dense(user_dense_feats).unsqueeze(1)
        if not isinstance(self.ue_proj, nn.ModuleList):
            raise RuntimeError(
                "ue_split_separate_tokens=True but ue_proj is not ModuleList")
        tokens: List[torch.Tensor] = []
        for proj, (offset, dim) in zip(self.ue_proj, self.ue_slices):
            part = user_dense_feats[:, offset:offset + dim]
            tokens.append(F.silu(proj(part)).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def _apply_dense_bypass(
        self,
        output: torch.Tensor,
        inputs: ModelInput,
        user_ns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate bypassed dense vector to the backbone output.

        Only called when ``has_dense_bypass`` is True. Produces a
        (B, 2*D) tensor that feeds directly into the classifier.

        T37 / ADR-014 · When ``enable_ue_split`` is True, the bypass input
        is the "other" dense feats (everything except the UE slice
        user_dense_feats[ue_offset:ue_offset+ue_dim]). The UE slice goes
        to the NS token path via ue_proj.

        T39 / M55 · When ``enable_ue_int_bilinear`` is True, the "other"
        dense feats are gated by a Linear projection of pooled user_int
        representation BEFORE the bypass projection: gated = other_dense
        * (1 + alpha * tanh(Linear(mean_pool(user_ns)))). Implements
        Seed 主楼 "其他跟 int pair 加权". Caller must pass ``user_ns``
        when this flag is True.
        """
        if self.enable_ue_split:
            other_dense = self._slice_other_dense(inputs.user_dense_feats)
        else:
            other_dense = inputs.user_dense_feats

        if self.enable_ue_int_bilinear:
            if user_ns is None:
                raise RuntimeError(
                    "enable_ue_int_bilinear=True but user_ns not provided "
                    "to _apply_dense_bypass; caller must pass it explicitly.")
            user_int_pooled = user_ns.mean(dim=1)  # (B, D)
            gating_logits = self.ue_int_bilinear_proj(user_int_pooled)
            gating = 1.0 + self.ue_int_bilinear_alpha * torch.tanh(
                gating_logits)
            other_dense = other_dense * gating  # (B, bypass_dense_dim)

        bypass = self.user_dense_bypass_proj(other_dense)  # (B, D)
        return torch.cat([output, bypass], dim=-1)  # (B, 2D)

    def _apply_ue_item_interaction_bypass(
        self,
        output: torch.Tensor,
        inputs: ModelInput,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Append explicit UE x target-item interaction to classifier input."""
        ue_repr = self._project_ue_dense(inputs.user_dense_feats)  # (B, D)
        item_repr = self._compute_target_repr(ns_tokens)  # (B, D)
        interaction_in = torch.cat(
            [ue_repr, item_repr, ue_repr * item_repr],
            dim=-1,
        )
        interaction = (
            self.ue_item_interaction_alpha
            * self.ue_item_interaction_proj(interaction_in)
        )
        return torch.cat([output, interaction], dim=-1)

    def _apply_din_interest_bypass(
        self,
        output: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        ns_tokens: torch.Tensor,
        seq_time_buckets_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Concatenate DIN interest vector to the backbone output.

        Only called when ``enable_din_interest`` is True. The target
        representation is the mean of the item_ns_tokens slice, matching
        TargetItemSeqInjector's convention. Depending on
        ``din_interest_merge``, the appended interest width is either
        ``D`` (compact) or ``S*D`` (per-domain).
        """
        target_repr = self._compute_target_repr(ns_tokens)  # (B, D)
        if self.enable_tin_interest:
            if seq_time_buckets_list is None:
                raise RuntimeError(
                    "enable_tin_interest=True but seq_time_buckets_list was "
                    "not provided to _apply_din_interest_bypass")
            din_interest = self.din_extractor(
                target_repr,
                seq_tokens_list,
                seq_masks_list,
                seq_time_buckets_list,
            )
        else:
            din_interest = self.din_extractor(
                target_repr, seq_tokens_list, seq_masks_list)
        return torch.cat([output, din_interest], dim=-1)

    def _apply_dcn_cross_bypass(
        self,
        output: torch.Tensor,
        inputs: ModelInput,
    ) -> torch.Tensor:
        """T34 / EXP-049 · Concatenate DCN cross output to backbone.

        Only called when ``enable_dcn_cross`` is True. The cross module
        REUSES the NS tokenizer embedding tables (DCN-paper convention:
        same fid representation, different interaction view) so it
        does not introduce new sparse params.

        Layout-wise the cross vector is appended at the END of the
        backbone output (so its column position is stable across
        flag combinations: dense_bypass / din_interest / dcn_cross).
        Order in the final classifier input: backbone | dense_bypass |
        din_interest | dcn_cross.
        """
        cross = self.dcn_cross(
            user_int_feats=inputs.user_int_feats,
            item_int_feats=inputs.item_int_feats,
            user_embs=self.user_ns_tokenizer.embs,
            item_embs=self.item_ns_tokenizer.embs,
            user_emb_index=self.user_ns_tokenizer._emb_index,
            item_emb_index=self.item_ns_tokenizer._emb_index,
        )  # (B, D)
        return torch.cat([output, cross], dim=-1)

    def _compute_target_repr(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        """Slice the item_ns block from ns_tokens and mean-pool to (B, D).

        Used as a shared helper by T16 TargetItemSeqInjector, T25 DIN
        bypass, and T28 DIN integrated — all three mechanisms share this
        "target = mean(item_ns)" convention so their interactions stay
        predictable.

        Layout inside ns_tokens (from _build_ns_tokens):
            [user_ns (num_user_ns) | optional user_dense
             | item_ns (num_item_ns) | optional item_dense]
        Robust to presence/absence of user_dense (it comes BEFORE item_ns).
        """
        user_dense_offset = self.user_dense_ns_tokens if self.has_user_dense else 0
        item_ns_start = self.num_user_ns + user_dense_offset
        item_ns_end = item_ns_start + self.num_item_ns
        item_ns = ns_tokens[:, item_ns_start:item_ns_end, :]  # (B, num_item_ns, D)
        return item_ns.mean(dim=1)  # (B, D)

    def _build_seq_tokens(
        self, inputs: ModelInput
    ) -> Tuple[list, list]:
        """Embed all sequence domains, return (tokens_list, masks_list)."""
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                hash_flags=self._seq_hash_flags.get(domain))
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)
        return seq_tokens_list, seq_masks_list

    def _maybe_inject_target_into_seq(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
    ) -> list:
        """Inject target-item representation into each seq token (ADR-004).

        No-op (identity) when ``target_item_seq_injection='off'`` or when
        there are no item NS tokens. Reuses the same target representation
        source as ICQ: mean of the item NS tokens. Slices by an explicit
        offset (num_user_ns + has_user_dense) rather than assuming item NS
        sits at the end of ns_tokens, so the logic is robust to any future
        item_dense addition.
        """
        if self.target_item_seq_injector.mode == 'off':
            return seq_tokens_list
        if self.num_item_ns <= 0:
            return seq_tokens_list
        item_ns_start = self.num_user_ns + (
            self.user_dense_ns_tokens if self.has_user_dense else 0)
        item_ns_end = item_ns_start + self.num_item_ns
        item_repr = ns_tokens[:, item_ns_start:item_ns_end, :].mean(dim=1)
        return self.target_item_seq_injector(seq_tokens_list, item_repr)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass (HyFormer or OneTrans depending on use_onetrans)."""
        # T36 / ADR-013 · When global_token is enabled, seq tokens must be
        # built first so _build_ns_tokens can mean-pool them into the
        # aggregated Global Token slot. Otherwise we keep the original
        # (ns_first) order to minimize churn in unrelated paths.
        if self.enable_global_token:
            seq_tokens_list, seq_masks_list = self._build_seq_tokens(inputs)
            ns_tokens = self._build_ns_tokens(
                inputs, seq_tokens_list, seq_masks_list)
        else:
            ns_tokens = self._build_ns_tokens(inputs)
            seq_tokens_list, seq_masks_list = self._build_seq_tokens(inputs)
        seq_tokens_list = self._maybe_inject_target_into_seq(
            ns_tokens, seq_tokens_list)
        din_seq_tokens_list = seq_tokens_list
        din_seq_masks_list = seq_masks_list
        din_seq_time_buckets_list = [
            inputs.seq_time_buckets[domain] for domain in self.seq_domains
        ]

        # T28 · target_repr shared across all HyFormer blocks when DIN
        # integrated is enabled. Compute once to avoid re-slicing per block.
        target_repr = (
            self._compute_target_repr(ns_tokens)
            if self.enable_din_integrated else None
        )

        if self.use_onetrans:
            output = self._run_onetrans_blocks(
                seq_tokens_list, seq_masks_list, ns_tokens,
                apply_dropout=self.training)
        else:
            q_tokens_list = self.query_generator(
                ns_tokens, seq_tokens_list, seq_masks_list)
            if (self.enable_din_interest
                    and self.din_interest_source == 'encoded'):
                output, din_seq_tokens_list, din_seq_masks_list = (
                    self._run_multi_seq_blocks(
                        q_tokens_list, ns_tokens, seq_tokens_list,
                        seq_masks_list, apply_dropout=self.training,
                        target_repr=target_repr,
                        return_seq_states=True)
                )
            else:
                output = self._run_multi_seq_blocks(
                    q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                    apply_dropout=self.training,
                    target_repr=target_repr)

        if self.has_dense_bypass:
            user_ns_for_bilinear = (
                ns_tokens[:, :self.num_user_ns, :]
                if self.enable_ue_int_bilinear else None)
            output = self._apply_dense_bypass(
                output, inputs, user_ns=user_ns_for_bilinear)

        if self.enable_din_interest:
            output = self._apply_din_interest_bypass(
                output,
                din_seq_tokens_list,
                din_seq_masks_list,
                ns_tokens,
                din_seq_time_buckets_list,
            )

        if self.enable_dcn_cross:
            output = self._apply_dcn_cross_bypass(output, inputs)

        if self.enable_ue_item_interaction:
            output = self._apply_ue_item_interaction_bypass(
                output, inputs, ns_tokens)

        logits = self.clsfier(output)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning (logits, embedding)."""
        if self.enable_global_token:
            seq_tokens_list, seq_masks_list = self._build_seq_tokens(inputs)
            ns_tokens = self._build_ns_tokens(
                inputs, seq_tokens_list, seq_masks_list)
        else:
            ns_tokens = self._build_ns_tokens(inputs)
            seq_tokens_list, seq_masks_list = self._build_seq_tokens(inputs)
        seq_tokens_list = self._maybe_inject_target_into_seq(
            ns_tokens, seq_tokens_list)
        din_seq_tokens_list = seq_tokens_list
        din_seq_masks_list = seq_masks_list
        din_seq_time_buckets_list = [
            inputs.seq_time_buckets[domain] for domain in self.seq_domains
        ]

        target_repr = (
            self._compute_target_repr(ns_tokens)
            if self.enable_din_integrated else None
        )

        if self.use_onetrans:
            output = self._run_onetrans_blocks(
                seq_tokens_list, seq_masks_list, ns_tokens,
                apply_dropout=False)
        else:
            q_tokens_list = self.query_generator(
                ns_tokens, seq_tokens_list, seq_masks_list)
            if (self.enable_din_interest
                    and self.din_interest_source == 'encoded'):
                output, din_seq_tokens_list, din_seq_masks_list = (
                    self._run_multi_seq_blocks(
                        q_tokens_list, ns_tokens, seq_tokens_list,
                        seq_masks_list, apply_dropout=False,
                        target_repr=target_repr,
                        return_seq_states=True)
                )
            else:
                output = self._run_multi_seq_blocks(
                    q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                    apply_dropout=False,
                    target_repr=target_repr)

        if self.has_dense_bypass:
            user_ns_for_bilinear = (
                ns_tokens[:, :self.num_user_ns, :]
                if self.enable_ue_int_bilinear else None)
            output = self._apply_dense_bypass(
                output, inputs, user_ns=user_ns_for_bilinear)

        if self.enable_din_interest:
            output = self._apply_din_interest_bypass(
                output,
                din_seq_tokens_list,
                din_seq_masks_list,
                ns_tokens,
                din_seq_time_buckets_list,
            )

        if self.enable_dcn_cross:
            output = self._apply_dcn_cross_bypass(output, inputs)

        if self.enable_ue_item_interaction:
            output = self._apply_ue_item_interaction_bypass(
                output, inputs, ns_tokens)

        logits = self.clsfier(output)
        return logits, output
