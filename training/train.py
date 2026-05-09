"""Pre-training script for Bangla-LLM 300M.

Uses HuggingFace Transformers LlamaForCausalLM with Accelerate for distributed
training and mixed precision. Designed to run on a single Brev A100 instance.

Usage:
    accelerate launch training/train.py \\
        --tokens data/tokens.bin \\
        --output-dir checkpoints/pretrain-v1 \\
        --epochs 3 --batch-size 8 --grad-accum 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, get_cosine_schedule_with_warmup

from config import BanglaLLMConfig
from dataset import TokenDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train Bangla-LLM 300M on tokenized corpus.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to binary uint16 token file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device micro batch size (default: 8).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=32,
        help="Gradient accumulation steps (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear warmup steps. Overrides --warmup-ratio when set.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup as a fraction of total optimizer steps (default: 0.05).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N optimizer steps (default: 1000).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.01,
        help="Reserve the tail of the token stream as a held-out eval slice (default: 0.01). Set 0 to disable.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="Run held-out eval every N optimizer steps (default: 250).",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=32,
        help="Maximum eval batches per evaluation pass (default: 32).",
    )
    return parser.parse_args()


def build_model(cfg: BanglaLLMConfig) -> LlamaForCausalLM:
    """Initialize a fresh LlamaForCausalLM from BanglaLLMConfig."""
    hf_config = cfg.to_hf_config()
    model = LlamaForCausalLM(hf_config)
    return model


@torch.no_grad()
def run_eval(
    model: LlamaForCausalLM,
    dataloader: DataLoader | None,
    accelerator: Accelerator,
    max_batches: int,
) -> dict[str, float] | None:
    """Evaluate on a held-out token slice without blowing out run time."""
    if dataloader is None:
        return None

    model.eval()
    losses: list[float] = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        outputs = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = outputs.loss.detach().float()
        reduced_loss = accelerator.gather_for_metrics(loss.unsqueeze(0)).mean().item()
        losses.append(reduced_loss)

    model.train()

    if not losses:
        return None

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 20.0))
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "batches": float(len(losses)),
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Accelerator ---
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
    )

    # --- Model ---
    cfg = BanglaLLMConfig(max_seq_len=args.seq_len)

    if args.resume:
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        model = LlamaForCausalLM.from_pretrained(args.resume)
    else:
        accelerator.print("Initializing fresh model from BanglaLLMConfig")
        model = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model parameters: {total_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # --- Dataset info for scheduling ---
    # Peek at token count to compute steps
    token_file = Path(args.tokens)
    if not token_file.exists():
        accelerator.print(f"ERROR: Token file not found: {token_file}")
        sys.exit(1)

    file_size = token_file.stat().st_size
    total_tokens = file_size // 2  # uint16 = 2 bytes per token
    raw_eval_tokens = int(total_tokens * max(args.eval_ratio, 0.0))
    min_eval_tokens = args.seq_len + 1
    eval_tokens = raw_eval_tokens if raw_eval_tokens >= min_eval_tokens else 0
    train_tokens = total_tokens - eval_tokens
    if train_tokens <= args.seq_len:
        accelerator.print("ERROR: Not enough tokens left for training after eval split.")
        sys.exit(1)

    seqs_per_epoch = max(1, (train_tokens - 1) // args.seq_len)
    steps_per_epoch = math.ceil(
        seqs_per_epoch / (args.batch_size * args.grad_accum * accelerator.num_processes)
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, int(total_steps * args.warmup_ratio))
    )
    effective_batch_tokens = (
        args.batch_size * args.grad_accum * accelerator.num_processes * args.seq_len
    )

    accelerator.print(f"Token file: {token_file}")
    accelerator.print(f"Total tokens: {total_tokens:,}")
    accelerator.print(f"Train tokens: {train_tokens:,}")
    accelerator.print(f"Eval tokens: {eval_tokens:,}")
    accelerator.print(f"Sequences per epoch: {seqs_per_epoch:,}")
    accelerator.print(f"Steps per epoch: {steps_per_epoch:,}")
    accelerator.print(f"Total steps: {total_steps:,}")
    accelerator.print(f"Effective batch size: {args.batch_size * args.grad_accum * accelerator.num_processes} sequences")
    accelerator.print(f"Effective batch tokens: {effective_batch_tokens:,}")
    accelerator.print(f"Epochs: {args.epochs}")
    accelerator.print(f"Peak LR: {args.lr}")
    accelerator.print(f"Warmup steps: {warmup_steps}")

    # --- Scheduler ---
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Prepare with Accelerate (model, optimizer, scheduler) ---
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def build_loader(*, epoch: int, start_token: int, end_token: int, shuffle: bool) -> DataLoader:
        dataset = TokenDataset(
            token_file=args.tokens,
            seq_len=args.seq_len,
            epoch=epoch,
            start_token=start_token,
            end_token=end_token,
            shuffle=shuffle,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        return accelerator.prepare(dataloader)

    eval_dataloader = None
    if eval_tokens:
        eval_dataloader = build_loader(
            epoch=0,
            start_token=train_tokens,
            end_token=total_tokens,
            shuffle=False,
        )

    best_eval_loss: float | None = None

    def save_metadata(
        save_dir: Path,
        *,
        global_step: int,
        epoch: int,
        avg_loss: float | None = None,
        eval_metrics: dict[str, float] | None = None,
    ) -> None:
        if not accelerator.is_main_process:
            return

        payload = {
            "global_step": global_step,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "avg_train_loss": avg_loss,
            "eval": eval_metrics,
        }
        (save_dir / "training_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    initial_eval = run_eval(model, eval_dataloader, accelerator, args.eval_max_batches)
    if initial_eval is not None:
        accelerator.print(
            "Initial eval: "
            f"loss={initial_eval['loss']:.4f} "
            f"ppl={initial_eval['perplexity']:.2f} "
            f"batches={int(initial_eval['batches'])}"
        )
        best_eval_loss = initial_eval["loss"]

    # --- Training loop ---
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        accelerator.print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        dataloader = build_loader(
            epoch=epoch,
            start_token=0,
            end_token=train_tokens,
            shuffle=True,
        )

        model.train()
        epoch_loss_sum = 0.0
        epoch_micro_steps = 0

        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss_sum += loss.detach().float().item()
            epoch_micro_steps += 1

            # Logging at optimizer step boundaries
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    avg_loss = epoch_loss_sum / max(epoch_micro_steps, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    tokens_seen = global_step * effective_batch_tokens
                    accelerator.print(
                        f"  epoch={epoch + 1} step={global_step} "
                        f"loss={avg_loss:.4f} lr={current_lr:.2e} "
                        f"tokens={tokens_seen:,} elapsed={elapsed:.0f}s"
                    )

                if eval_dataloader is not None and global_step % args.eval_every == 0:
                    eval_metrics = run_eval(
                        model,
                        eval_dataloader,
                        accelerator,
                        args.eval_max_batches,
                    )
                    if eval_metrics is not None:
                        accelerator.print(
                            "  eval "
                            f"step={global_step} "
                            f"loss={eval_metrics['loss']:.4f} "
                            f"ppl={eval_metrics['perplexity']:.2f}"
                        )

                # Save periodic checkpoint
                if global_step % args.save_every == 0:
                    ckpt_dir = output_dir / f"step-{global_step}"
                    accelerator.print(f"  Saving checkpoint: {ckpt_dir}")
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(
                        ckpt_dir,
                        save_function=accelerator.save,
                    )
                    save_metadata(
                        ckpt_dir,
                        global_step=global_step,
                        epoch=epoch + 1,
                        avg_loss=epoch_loss_sum / max(epoch_micro_steps, 1),
                    )

        # --- End of epoch ---
        avg_epoch_loss = epoch_loss_sum / max(epoch_micro_steps, 1)
        accelerator.print(
            f"  Epoch {epoch + 1} complete. "
            f"Avg loss: {avg_epoch_loss:.4f}, "
            f"Global step: {global_step}"
        )

        epoch_eval = run_eval(model, eval_dataloader, accelerator, args.eval_max_batches)
        if epoch_eval is not None:
            accelerator.print(
                f"  Eval after epoch {epoch + 1}: "
                f"loss={epoch_eval['loss']:.4f} "
                f"ppl={epoch_eval['perplexity']:.2f}"
            )

        epoch_ckpt_dir = output_dir / f"epoch-{epoch + 1}"
        accelerator.print(f"  Saving epoch checkpoint: {epoch_ckpt_dir}")
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            epoch_ckpt_dir,
            save_function=accelerator.save,
        )
        save_metadata(
            epoch_ckpt_dir,
            global_step=global_step,
            epoch=epoch + 1,
            avg_loss=avg_epoch_loss,
            eval_metrics=epoch_eval,
        )

        if (
            epoch_eval is not None
            and (best_eval_loss is None or epoch_eval["loss"] < best_eval_loss)
        ):
            best_eval_loss = epoch_eval["loss"]
            best_dir = output_dir / "best"
            accelerator.print(f"  New best checkpoint: {best_dir}")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                best_dir,
                save_function=accelerator.save,
            )
            save_metadata(
                best_dir,
                global_step=global_step,
                epoch=epoch + 1,
                avg_loss=avg_epoch_loss,
                eval_metrics=epoch_eval,
            )

    # --- Save final model ---
    final_dir = output_dir / "final"
    accelerator.print(f"\nSaving final model: {final_dir}")
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        final_dir,
        save_function=accelerator.save,
    )
    final_eval = run_eval(model, eval_dataloader, accelerator, args.eval_max_batches)
    save_metadata(
        final_dir,
        global_step=global_step,
        epoch=args.epochs,
        avg_loss=avg_epoch_loss,
        eval_metrics=final_eval,
    )

    total_time = time.time() - start_time
    accelerator.print(f"\nTraining complete. Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
    accelerator.print(f"Final step: {global_step}, Checkpoints in: {output_dir}")


if __name__ == "__main__":
    main()
