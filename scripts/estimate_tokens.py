"""Estimate token counts for Bengali text corpora.

Bengali UTF-8 encoding uses 3 bytes per character (U+0980-U+09FF).
With a well-trained 32K BPE tokenizer, Bengali typically yields 2-4
characters per token depending on vocabulary coverage and text domain.
"""


def estimate_tokens(
    corpus_gb: float,
    bytes_per_char: int,
    chars_per_token: float,
) -> dict[str, float]:
    """Return character count, token count, and Chinchilla-optimal model size."""
    total_bytes = corpus_gb * (1024**3)
    total_chars = total_bytes / bytes_per_char
    total_tokens = total_chars / chars_per_token
    # Chinchilla scaling law: optimal model params ~ tokens / 20
    chinchilla_params = total_tokens / 20
    return {
        "corpus_gb": corpus_gb,
        "chars_per_token": chars_per_token,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chinchilla_params": chinchilla_params,
    }


def format_number(n: float) -> str:
    """Format large numbers with B/M suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,.0f}"


def corpus_gb_for_target_tokens(
    target_tokens: float,
    bytes_per_char: int,
    chars_per_token: float,
) -> float:
    """Calculate corpus size in GB needed to reach a target token count."""
    total_chars = target_tokens * chars_per_token
    total_bytes = total_chars * bytes_per_char
    return total_bytes / (1024**3)


def main() -> None:
    bytes_per_char = 3  # Bengali chars are 3 bytes in UTF-8
    corpus_sizes = [5.0, 6.0]
    chars_per_token_ratios = [2.0, 3.0, 4.0]

    # Header
    print("=" * 90)
    print("Bengali Corpus Token Estimation")
    print(f"Assumptions: {bytes_per_char} bytes/char (UTF-8 Bengali), 32K BPE vocab")
    print("=" * 90)

    # Character counts
    print("\n--- Raw Character Counts ---")
    for gb in corpus_sizes:
        total_bytes = gb * (1024**3)
        total_chars = total_bytes / bytes_per_char
        print(f"  {gb:.0f} GB  ->  {format_number(total_chars)} characters  ({total_chars:.4e})")

    # Token estimation table
    print("\n--- Token Estimates & Chinchilla-Optimal Model Size ---")
    header = f"{'Corpus (GB)':>12} | {'Chars/Token':>12} | {'Est. Tokens':>14} | {'Chinchilla Params (T/20)':>25}"
    print(header)
    print("-" * len(header))

    for gb in corpus_sizes:
        for cpt in chars_per_token_ratios:
            result = estimate_tokens(gb, bytes_per_char, cpt)
            tokens_str = format_number(result["total_tokens"])
            params_str = format_number(result["chinchilla_params"])
            print(
                f"{result['corpus_gb']:>12.0f} | "
                f"{result['chars_per_token']:>12.1f} | "
                f"{tokens_str:>14} | "
                f"{params_str:>25}"
            )
        print("-" * len(header))

    # Reverse calculation: how much data for 5B tokens?
    print("\n--- Corpus Size Needed for 5B Tokens ---")
    target = 5e9
    for cpt in chars_per_token_ratios:
        needed_gb = corpus_gb_for_target_tokens(target, bytes_per_char, cpt)
        print(f"  At {cpt:.0f} chars/token:  {needed_gb:.2f} GB of clean Bengali UTF-8 text")

    # Context for our project
    print("\n--- Project Context ---")
    print("  300M model (Chinchilla-optimal): needs ~6B tokens")
    print("  1B model   (Chinchilla-optimal): needs ~20B tokens")
    for model_tokens, label in [(6e9, "300M model (6B tokens)"), (20e9, "1B model (20B tokens)")]:
        gb_needed = corpus_gb_for_target_tokens(model_tokens, bytes_per_char, 3.0)
        print(f"  {label} at 3 chars/token -> {gb_needed:.1f} GB clean text needed")

    print()


if __name__ == "__main__":
    main()
