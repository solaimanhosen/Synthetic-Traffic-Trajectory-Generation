"""
Usage:
    # Run on a JSONL eval file (defaults to first 10 examples):
    python inference.py \
        --model_dir ./ngsim_model \
        --eval_file data/ngsim_rowbyrow_eval.jsonl \
        --num_examples 100 \
        --output_file predictions.jsonl

    # Run on a single JSON string:
    python inference.py \
        --model_dir ./ngsim_model \
        --single_input '{"instruction":"...","input":"..."}'
"""

import os
import json
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 512

ALPACA_PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
"""

COMPARE_KEYS = ["Local_X", "Local_Y", "v_Vel", "v_Acc", "Space_Headway"]


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned NGSIM model."
    )
    # Model
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL,
                        help="Path to saved model (or HF model ID).")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Path to saved tokenizer (defaults to model_dir).")
    # Input source (choose one)
    parser.add_argument("--eval_file", type=str, default=None,
                        help="Path to evaluation JSONL file.")
    parser.add_argument("--single_input", type=str, default=None,
                        help='Single example as a JSON string with '
                             '"instruction" and "input" keys.')
    # Inference settings
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to evaluate from eval_file.")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Truncation length for the prompt tokens.")
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional JSONL path to save predictions.")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model + tokenizer loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_dir: str, tokenizer_dir: Optional[str] = None):
    """Load a fine-tuned (or base) model for inference."""
    tokenizer_dir = tokenizer_dir or model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loaded model    : {model_dir}")
    print(f"Loaded tokenizer: {tokenizer_dir}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Single-example inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    example: dict,
    model,
    tokenizer,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_input_length: int = 256,
) -> str:
    """Generate a prediction for one example dict with 'instruction' and 'input' keys."""
    prompt = ALPACA_PROMPT.format(example["instruction"], example["input"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_deltas(expected_json: str, predicted_str: str) -> Optional[dict]:
    """Parse both JSON strings and return per-key absolute errors, or None."""
    try:
        exp = json.loads(expected_json)
        pred = json.loads(predicted_str)
    except (json.JSONDecodeError, TypeError):
        return None

    deltas = {}
    for key in COMPARE_KEYS:
        if key in exp and key in pred:
            try:
                deltas[key] = abs(float(pred[key]) - float(exp[key]))
            except (ValueError, TypeError):
                deltas[key] = None
    return deltas


def print_example_result(index: int, example: dict, predicted: str, deltas: Optional[dict]):
    """Pretty-print one evaluation result."""
    print(f"\n{'─' * 60}")
    print(f"Example {index + 1}")
    print(f"  Input (last 200 chars): ...{example['input'][-200:]}")
    print(f"  Expected : {example['output']}")
    print(f"  Predicted: {predicted}")

    if deltas:
        for key, value in deltas.items():
            label = f"  Δ {key:<15s}"
            print(f"{label}: {value:.3f}" if value is not None else f"{label}: parse error")
    else:
        print("  (could not parse predicted JSON)")


# ══════════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_file(
    eval_file: str,
    model,
    tokenizer,
    num_examples: int,
    max_new_tokens: int,
    max_input_length: int,
    output_file: Optional[str] = None,
):
    """Run inference on the first `num_examples` from a JSONL eval set."""
    ds = load_dataset("json", data_files=eval_file, split="train")
    num_examples = min(num_examples, len(ds))
    print(f"\nRunning inference on {num_examples} examples from {eval_file}")
    print("=" * 60)

    results = []
    all_deltas: dict[str, list[float]] = {k: [] for k in COMPARE_KEYS}

    for i in range(num_examples):
        example   = ds[i]
        predicted = run_inference(example, model, tokenizer, max_new_tokens, max_input_length)
        deltas    = compute_deltas(example["output"], predicted)

        print_example_result(i, example, predicted, deltas)

        record = {
            "index": i,
            "expected": example["output"],
            "predicted": predicted,
            "deltas": deltas,
        }
        results.append(record)

        if deltas:
            for key, value in deltas.items():
                if value is not None:
                    all_deltas[key].append(value)

    # Summary statistics
    print(f"\n{'═' * 60}")
    print("Aggregate MAE (Mean Absolute Error)")
    for key in COMPARE_KEYS:
        values = all_deltas[key]
        if values:
            mae = sum(values) / len(values)
            print(f"  {key:<15s}: {mae:.4f}  (n={len(values)})")
        else:
            print(f"  {key:<15s}: no valid predictions")

    parse_failures = sum(1 for r in results if r["deltas"] is None)
    print(f"\nParse failures: {parse_failures}/{num_examples}")

    # Save predictions
    if output_file:
        save_predictions(results, output_file)

    return results


def save_predictions(results: list[dict], path: str):
    """Write prediction records to a JSONL file."""
    with open(path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"Predictions saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Validate input source
    if not args.eval_file and not args.single_input:
        raise ValueError("Provide either --eval_file or --single_input.")

    # 1. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.tokenizer_dir)

    # 2a. Single-example mode
    if args.single_input:
        example = json.loads(args.single_input)
        if "instruction" not in example or "input" not in example:
            raise KeyError("--single_input JSON must have 'instruction' and 'input' keys.")

        predicted = run_inference(example, model, tokenizer,
                                 args.max_new_tokens, args.max_input_length)
        print(f"\nPrediction:\n{predicted}")
        return

    # 2b. File evaluation mode
    if not os.path.isfile(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    evaluate_file(
        eval_file=args.eval_file,
        model=model,
        tokenizer=tokenizer,
        num_examples=args.num_examples,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
