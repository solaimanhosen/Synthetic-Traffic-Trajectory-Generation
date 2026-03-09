"""
Usage:
    python finetune.py \
        --train_file ngsim_rowbyrow_train.jsonl \
        --eval_file  ngsim_rowbyrow_eval.jsonl \
        --output_dir ./results \
        --model_save_dir ./ngsim_model \
        --num_train_epochs 1 \
        --resume_from_checkpoint
"""

import os
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

ALPACA_PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B on NGSIM row-by-row data."
    )
    # Data
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training JSONL file.")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Path to evaluation JSONL file.")
    # Model I/O
    parser.add_argument("--base_model", type=str, default=BASE_MODEL,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Checkpoint / training artifact directory.")
    parser.add_argument("--model_save_dir", type=str, default="./ngsim_model",
                        help="Where to save the final model.")
    parser.add_argument("--tokenizer_save_dir", type=str, default=None,
                        help="Where to save the tokenizer (defaults to model_save_dir).")
    # Training hyper-parameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=500,
                    help="Cap eval set size for faster evaluation.")
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from the latest checkpoint.")

    return parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(train_file: str, eval_file: str, max_eval_samples: int, tokenizer):
    """Load JSONL files and apply Alpaca-style formatting."""
    eos_token = tokenizer.eos_token

    def formatting_fn(batch):
        texts = []
        for instruction, input_text, output in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            text = ALPACA_PROMPT.format(instruction, input_text, output) + eos_token
            texts.append(text)
        return {"text": texts}

    train_ds = load_dataset("json", data_files=train_file, split="train")
    eval_ds = load_dataset("json", data_files=eval_file, split="train")
    if max_eval_samples and len(eval_ds) > max_eval_samples:
        eval_ds = eval_ds.select(range(max_eval_samples))

    num_proc = min(os.cpu_count(), 8)
    train_ds = train_ds.map(formatting_fn, batched=True, num_proc=num_proc)
    eval_ds  = eval_ds.map(formatting_fn, batched=True, num_proc=num_proc)

    print(f"Train examples: {len(train_ds):,}")
    print(f"Eval  examples: {len(eval_ds):,}")
    return train_ds, eval_ds


# ══════════════════════════════════════════════════════════════════════════════
# Model + tokenizer loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str):
    """Load the base model in bfloat16 with gradient checkpointing."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    print(f"Loaded model: {model_name}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Trainer setup
# ══════════════════════════════════════════════════════════════════════════════

def build_trainer(model, tokenizer, train_ds, eval_ds, args: argparse.Namespace):
    """Configure and return an SFTTrainer."""
    sft_config = SFTConfig(
        output_dir                  = args.output_dir,
        max_length                  = args.max_length,
        packing                     = True,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        learning_rate               = args.learning_rate,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size  = args.eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_train_epochs            = args.num_train_epochs,
        weight_decay                = args.weight_decay,
        warmup_steps                = args.warmup_steps,
        logging_steps               = args.logging_steps,
        dataset_text_field          = "text",
        save_total_limit            = 2,
        report_to                   = "none",
        bf16                        = True,
        optim                       = "paged_adamw_32bit",
        lr_scheduler_type           = "cosine",
        seed                        = args.seed,
    )

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        args             = sft_config,
    )
    return trainer

# ══════════════════════════════════════════════════════════════════════════════
# Save model
# ══════════════════════════════════════════════════════════════════════════════

def save_model(trainer, tokenizer, model_dir: str, tokenizer_dir: str):
    """Save the fine-tuned model and tokenizer to disk."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Model     saved → {model_dir}")
    print(f"Tokenizer saved → {tokenizer_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 1. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model)

    # 2. Load and format datasets
    train_ds, eval_ds = load_datasets(args.train_file, args.eval_file, args.max_eval_samples, tokenizer)

    # 3. Build trainer
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, args)

    # 4. Patch trainer state if resuming with different schedule args
    if args.resume_from_checkpoint:
        import glob, json as _json
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
        if ckpts:
            state_path = os.path.join(ckpts[-1], "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = _json.load(f)
                state["save_steps"]    = args.save_steps
                state["eval_steps"]    = args.eval_steps
                state["logging_steps"] = args.logging_steps
                with open(state_path, "w") as f:
                    _json.dump(state, f, indent=2)
                print(f"Patched trainer_state.json in {ckpts[-1]}")

    # 5. Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None)

    # 5. Save
    tokenizer_dir = args.tokenizer_save_dir or args.model_save_dir
    save_model(trainer, tokenizer, args.model_save_dir, tokenizer_dir)

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
