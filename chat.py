#!/usr/bin/env python3
"""Interactive chat with Kotha-1.

Usage:
    python chat.py --model checkpoints/kotha-1-sft

    # With the base model (text completion, no chat)
    python chat.py --model training/checkpoints/step-4000 --raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent / "finetune"))
from bangla_tokenizer import load_bangla_tokenizer

CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"


def load_model_and_tokenizer(model_dir: str):
    model_path = Path(model_dir)

    # Find tokenizer
    sp_path = model_path / "bangla_bpe_32k.model"
    if not sp_path.exists():
        sp_path = Path("tokenizer/output/bangla_bpe_32k.model")
    if not sp_path.exists():
        print(f"ERROR: Cannot find bangla_bpe_32k.model in {model_path} or tokenizer/output/")
        sys.exit(1)

    tokenizer = load_bangla_tokenizer(str(model_path if model_path.is_dir() else sp_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    print(f"Loaded on {device} ({dtype})\n")

    return model, tokenizer, device


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, device: str,
             max_new_tokens: int = 256, temperature: float = 0.7,
             top_p: float = 0.9, top_k: int = 50,
             repetition_penalty: float = 1.1) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Stop on </s> and, when present, the ChatML turn terminator <|im_end|>.
    # generation_config only declares </s>=2, but SFT ends turns with <|im_end|>,
    # so without this the chat path never halts and runs to max_new_tokens.
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids(CHATML_END)
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": eos_ids,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    output = model.generate(input_ids, **gen_kwargs)
    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated.tolist(), skip_special_tokens=False)


def chat_loop(model, tokenizer, device, raw_mode: bool = False,
              max_tokens: int = 256, temperature: float = 0.7):
    print("=" * 50)
    if raw_mode:
        print("  Kotha-1 Base (text completion mode)")
        print("  Type Bengali text and it will continue it.")
    else:
        print("  Kotha-1 Chat")
        print("  Type in Bengali or English. Type 'quit' to exit.")
    print("=" * 50)
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if raw_mode:
            # Raw completion mode for base model
            response = generate(model, tokenizer, user_input, device,
                                max_new_tokens=max_tokens, temperature=temperature)
            # Clean up
            if "</s>" in response:
                response = response[:response.index("</s>")]
            print(f"Kotha: {response.strip()}\n")
        else:
            # ChatML format for SFT model
            history.append({"role": "user", "content": user_input})

            # Build prompt
            prompt_parts = []
            for msg in history:
                prompt_parts.append(
                    f"{CHATML_START}{msg['role']}\n{msg['content']}{CHATML_END}"
                )
            prompt_parts.append(f"{CHATML_START}assistant\n")
            prompt = "\n".join(prompt_parts)

            response = generate(model, tokenizer, prompt, device,
                                max_new_tokens=max_tokens, temperature=temperature)

            # Extract response (stop at im_end)
            if CHATML_END in response:
                response = response[:response.index(CHATML_END)]
            response = response.strip()

            history.append({"role": "assistant", "content": response})
            print(f"Kotha: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Kotha-1")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--raw", action="store_true", help="Raw completion mode (for base model)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model)
    chat_loop(model, tokenizer, device, raw_mode=args.raw,
              max_tokens=args.max_tokens, temperature=args.temperature)


if __name__ == "__main__":
    main()
