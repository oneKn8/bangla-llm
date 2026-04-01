"""
LLaMA-style transformer parameter count calculator.

Calculates exact parameter counts for a LLaMA architecture with:
- Grouped Query Attention (GQA)
- SwiGLU FFN (3 projection matrices)
- RMSNorm (scalar per dimension, no bias)
- RoPE (no learned position embeddings)
- Optional weight tying (embedding shared with output head)
"""

from dataclasses import dataclass


@dataclass
class LLaMAConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ffn: int
    vocab_size: int
    weight_tying: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def count_params(cfg: LLaMAConfig) -> dict[str, int]:
    """Count parameters for each component of a LLaMA-style transformer."""
    d = cfg.d_model
    h = cfg.n_heads
    kv_h = cfg.n_kv_heads
    d_h = cfg.d_head
    d_ffn = cfg.d_ffn
    v = cfg.vocab_size
    L = cfg.n_layers

    # --- Per-layer breakdown ---

    # Self-attention (no bias anywhere in LLaMA)
    # Q projection: d_model -> n_heads * d_head = d_model
    attn_q = d * (h * d_h)
    # K projection: d_model -> n_kv_heads * d_head
    attn_k = d * (kv_h * d_h)
    # V projection: d_model -> n_kv_heads * d_head
    attn_v = d * (kv_h * d_h)
    # Output projection: n_heads * d_head -> d_model
    attn_o = (h * d_h) * d

    attn_total = attn_q + attn_k + attn_v + attn_o

    # SwiGLU FFN (no bias): gate_proj, up_proj, down_proj
    # gate_proj: d_model -> d_ffn
    ffn_gate = d * d_ffn
    # up_proj:   d_model -> d_ffn
    ffn_up = d * d_ffn
    # down_proj: d_ffn -> d_model
    ffn_down = d_ffn * d

    ffn_total = ffn_gate + ffn_up + ffn_down

    # RMSNorm: 2 per layer (pre-attention + pre-FFN), each has d_model learnable scales
    rmsnorm_per_layer = 2 * d

    layer_total = attn_total + ffn_total + rmsnorm_per_layer

    # --- Non-layer components ---

    # Token embedding: vocab_size * d_model
    embedding = v * d

    # Final RMSNorm (before output head)
    final_rmsnorm = d

    # Output head (lm_head): d_model -> vocab_size
    if cfg.weight_tying:
        output_head = 0  # shared with embedding
    else:
        output_head = d * v

    total = (layer_total * L) + embedding + final_rmsnorm + output_head

    return {
        "attn_q": attn_q,
        "attn_k": attn_k,
        "attn_v": attn_v,
        "attn_o": attn_o,
        "attn_total": attn_total,
        "ffn_gate": ffn_gate,
        "ffn_up": ffn_up,
        "ffn_down": ffn_down,
        "ffn_total": ffn_total,
        "rmsnorm_per_layer": rmsnorm_per_layer,
        "layer_total": layer_total,
        "all_layers": layer_total * L,
        "embedding": embedding,
        "final_rmsnorm": final_rmsnorm,
        "output_head": output_head,
        "total": total,
        "n_layers": L,
        "d_model": d,
    }


def fmt(n: int) -> str:
    """Format a number with commas and an approximate M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n:>15,}  ({n / 1e9:.2f}B)"
    if n >= 1_000_000:
        return f"{n:>15,}  ({n / 1e6:.2f}M)"
    return f"{n:>15,}"


def print_breakdown(params: dict[str, int], label: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  d_model={params['d_model']}, n_layers={params['n_layers']}")
    print(f"{'=' * 65}")

    print(f"\n  Per-layer attention:")
    print(f"    Q projection:          {fmt(params['attn_q'])}")
    print(f"    K projection:          {fmt(params['attn_k'])}")
    print(f"    V projection:          {fmt(params['attn_v'])}")
    print(f"    O projection:          {fmt(params['attn_o'])}")
    print(f"    Attention subtotal:    {fmt(params['attn_total'])}")

    print(f"\n  Per-layer FFN (SwiGLU):")
    print(f"    Gate projection:       {fmt(params['ffn_gate'])}")
    print(f"    Up projection:         {fmt(params['ffn_up'])}")
    print(f"    Down projection:       {fmt(params['ffn_down'])}")
    print(f"    FFN subtotal:          {fmt(params['ffn_total'])}")

    print(f"\n  Per-layer RMSNorm (x2):  {fmt(params['rmsnorm_per_layer'])}")
    print(f"  Per-layer TOTAL:         {fmt(params['layer_total'])}")

    print(f"\n  All {params['n_layers']} layers:            {fmt(params['all_layers'])}")
    print(f"  Token embedding:         {fmt(params['embedding'])}")
    print(f"  Final RMSNorm:           {fmt(params['final_rmsnorm'])}")
    print(f"  Output head (lm_head):   {fmt(params['output_head'])}  {'(tied)' if params['output_head'] == 0 else ''}")

    print(f"\n  TOTAL PARAMETERS:        {fmt(params['total'])}")
    print(f"{'=' * 65}")


def find_best_layers_for_target(
    target: int,
    d_model: int = 1024,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    d_ffn: int = 4096,
    vocab_size: int = 32000,
) -> LLaMAConfig:
    """Find the number of layers that gets closest to target param count."""
    best_cfg = None
    best_diff = float("inf")

    for n_layers in range(1, 100):
        cfg = LLaMAConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ffn=d_ffn,
            vocab_size=vocab_size,
        )
        total = count_params(cfg)["total"]
        diff = abs(total - target)
        if diff < best_diff:
            best_diff = diff
            best_cfg = cfg

    return best_cfg


def find_best_dmodel_for_target(
    target: int,
    n_layers: int = 24,
    vocab_size: int = 32000,
) -> LLaMAConfig:
    """Find d_model that gets closest to target param count.

    Constraints: d_model must be divisible by n_heads.
    We fix n_heads = d_model // 64 (so d_head=64, standard).
    n_kv_heads = n_heads // 4 (GQA ratio 4).
    d_ffn = 4 * d_model (standard SwiGLU hidden dim).
    """
    best_cfg = None
    best_diff = float("inf")

    # Search d_model from 256 to 4096 in steps of 64 (must be divisible by head_dim=64)
    for d_model in range(256, 4097, 64):
        n_heads = d_model // 64
        n_kv_heads = max(1, n_heads // 4)
        d_ffn = 4 * d_model  # standard multiplier

        # Also try the common 8/3 * d_model rounded to nearest 256 (more efficient SwiGLU)
        for ffn_mult_name, ffn_val in [
            ("4x", 4 * d_model),
            ("8/3x rounded", int(round(8 / 3 * d_model / 256)) * 256),
        ]:
            cfg = LLaMAConfig(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ffn=ffn_val,
                vocab_size=vocab_size,
            )
            total = count_params(cfg)["total"]
            diff = abs(total - target)
            if diff < best_diff:
                best_diff = diff
                best_cfg = cfg
                best_ffn_name = ffn_mult_name

    return best_cfg


def main() -> None:
    # --- Part 1: Exact count for the given specs ---
    cfg = LLaMAConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ffn=4096,
        vocab_size=32000,
        weight_tying=True,
    )
    params = count_params(cfg)
    print_breakdown(params, "GIVEN SPECS (weight tying ON)")

    # Also show without weight tying for reference
    cfg_no_tie = LLaMAConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ffn=4096,
        vocab_size=32000,
        weight_tying=False,
    )
    params_no_tie = count_params(cfg_no_tie)
    print_breakdown(params_no_tie, "GIVEN SPECS (weight tying OFF)")

    # --- Part 2: Best n_layers for ~300M ---
    target = 300_000_000
    print(f"\n\n{'#' * 65}")
    print(f"  SEARCH: Best n_layers for ~300M params (d_model=1024 fixed)")
    print(f"{'#' * 65}")

    best_layers_cfg = find_best_layers_for_target(target)
    best_layers_params = count_params(best_layers_cfg)
    print_breakdown(best_layers_params, f"BEST FIT: {best_layers_cfg.n_layers} layers")

    # Show neighbors
    for delta in [-1, 1]:
        neighbor_n = best_layers_cfg.n_layers + delta
        if neighbor_n < 1:
            continue
        neighbor_cfg = LLaMAConfig(
            d_model=1024,
            n_layers=neighbor_n,
            n_heads=16,
            n_kv_heads=4,
            d_ffn=4096,
            vocab_size=32000,
        )
        neighbor_total = count_params(neighbor_cfg)["total"]
        diff = neighbor_total - target
        print(f"  Neighbor: {neighbor_n} layers -> {fmt(neighbor_total)}, delta from 300M: {diff:+,}")

    # --- Part 3: Best d_model for ~300M ---
    print(f"\n\n{'#' * 65}")
    print(f"  SEARCH: Best d_model for ~300M params (24 layers fixed)")
    print(f"{'#' * 65}")

    best_dmodel_cfg = find_best_dmodel_for_target(target)
    best_dmodel_params = count_params(best_dmodel_cfg)
    print_breakdown(
        best_dmodel_params,
        f"BEST FIT: d_model={best_dmodel_cfg.d_model}, "
        f"n_heads={best_dmodel_cfg.n_heads}, "
        f"n_kv_heads={best_dmodel_cfg.n_kv_heads}, "
        f"d_ffn={best_dmodel_cfg.d_ffn}",
    )

    # Show a few neighbors
    print(f"\n  Nearby d_model options (24 layers, d_head=64, GQA ratio 4, d_ffn=4*d):")
    print(f"  {'d_model':>8} {'n_heads':>8} {'n_kv':>6} {'d_ffn':>6} {'total':>18} {'delta':>12}")
    print(f"  {'-'*60}")
    for d in range(
        max(256, best_dmodel_cfg.d_model - 256),
        best_dmodel_cfg.d_model + 320,
        64,
    ):
        nh = d // 64
        nkv = max(1, nh // 4)
        c = LLaMAConfig(d_model=d, n_layers=24, n_heads=nh, n_kv_heads=nkv, d_ffn=4 * d, vocab_size=32000)
        t = count_params(c)["total"]
        marker = " <-- best" if d == best_dmodel_cfg.d_model else ""
        print(f"  {d:>8} {nh:>8} {nkv:>6} {4*d:>6} {t:>15,} {t - target:>+12,}{marker}")


if __name__ == "__main__":
    main()
