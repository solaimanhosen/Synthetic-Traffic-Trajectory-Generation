import os
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_STEPS = 3
ADD_NEIGHBORS = False
OUTPUT_FILE = "data/ngsim_rowbyrow_wo_nei.jsonl"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_SIZE = 100_000
EVAL_SIZE = 20_000

INSTRUCTION = (
    "You are a traffic simulation model. "
    "Given a vehicle's recent trajectory and its nearest neighbor ahead in the same lane, "
    "predict its state in the next frame. "
    "Return only a single JSON object with keys: "
    "Frame_ID, Vehicle_ID, Lane_ID, Local_X, Local_Y, v_Vel, v_Acc, Space_Headway."
)

ALPACA_PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _round(v):
    """Round floats to 3 decimal places; leave other types untouched."""
    return round(v, 3) if isinstance(v, float) else v


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 – Load and prepare NGSIM data
# ══════════════════════════════════════════════════════════════════════════════

def load_ngsim_data(csv_path: str) -> pd.DataFrame:
    """Read the raw NGSIM CSV, keep relevant columns, and sort."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    cols_to_keep = [
        'Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y',
        'v_Vel', 'v_Acc', 'Lane_ID', 'Space_Headway',
    ]
    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    df = df[cols_to_keep]
    df = df.sort_values(by=['Frame_ID', 'Vehicle_ID']).reset_index(drop=True)

    print(f'Rows: {len(df):,}  |  Frames: {df["Frame_ID"].nunique():,}')
    print(f'Vehicles: {df["Vehicle_ID"].nunique():,}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – Generate row-by-row dataset
# ══════════════════════════════════════════════════════════════════════════════

def get_nearest_neighbor(vehicle_row, frame_df):
    """Return the closest vehicle ahead (higher Local_Y) in the same lane."""
    same_lane = frame_df[
        (frame_df["Lane_ID"]    == vehicle_row["Lane_ID"]) &
        (frame_df["Vehicle_ID"] != vehicle_row["Vehicle_ID"]) &
        (frame_df["Local_Y"]    >  vehicle_row["Local_Y"])
    ]
    if same_lane.empty:
        return None
    nearest = same_lane.loc[(same_lane["Local_Y"] - vehicle_row["Local_Y"]).idxmin()]
    return {k: _round(v) for k, v in nearest.items() if k != "Frame_ID"}


def _build_lookups(df: pd.DataFrame):
    """Create frame-level and vehicle-history lookup structures."""
    frame_lookup = {
        fid: grp.reset_index(drop=True)
        for fid, grp in df.groupby("Frame_ID")
    }

    vehicle_history: dict[int, list] = {}
    for _, row in df.iterrows():
        vid = int(row["Vehicle_ID"])
        vehicle_history.setdefault(vid, []).append(
            (int(row["Frame_ID"]),
             {k: _round(v) for k, v in row.items() if k != "Frame_ID"})
        )
    for vid in vehicle_history:
        vehicle_history[vid].sort(key=lambda x: x[0])

    return frame_lookup, vehicle_history


def create_rowbyrow_dataset(
    df: pd.DataFrame,
    output_file: str = OUTPUT_FILE,
    history: int = HISTORY_STEPS,
    add_neighbors: bool = ADD_NEIGHBORS,
) -> list[dict]:
    """Walk every vehicle's timeline and produce instruction/input/output triples."""
    frame_lookup, vehicle_history = _build_lookups(df)
    examples = []

    for vid, frames in vehicle_history.items():
        for i in range(history, len(frames)):
            window       = frames[i - history : i + 1]   # H past + current
            cur_frame_id = window[-1][0]
            nxt_frame_id = cur_frame_id + 1

            if nxt_frame_id not in frame_lookup:
                continue
            nxt_df   = frame_lookup[nxt_frame_id]
            nxt_rows = nxt_df[nxt_df["Vehicle_ID"] == vid]
            if nxt_rows.empty:
                continue

            # Build trajectory list
            trajectory = [{"frame": fid, **rd} for fid, rd in window]
            input_data: dict = {"trajectory": trajectory}

            # Nearest neighbor from current frame
            if add_neighbors:
                cur_row_dict = window[-1][1]
                cur_series   = pd.Series({**cur_row_dict, "Frame_ID": cur_frame_id})
                neighbor     = get_nearest_neighbor(cur_series, frame_lookup[cur_frame_id])
                if neighbor:
                    input_data["nearest_neighbor_ahead"] = neighbor

            # Output: vehicle's next state
            nxt_state = {"Frame_ID": nxt_frame_id}
            nxt_state.update({
                k: _round(v) for k, v in nxt_rows.iloc[0].items() if k != "Frame_ID"
            })

            examples.append({
                "instruction": INSTRUCTION,
                "input":       json.dumps(input_data),
                "output":      json.dumps(nxt_state),
            })

    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples):,} examples → {output_file}")
    return examples


def print_sample(examples: list[dict], index: int = 0):
    """Print a single example for quick sanity checking."""
    if index >= len(examples):
        print(f"WARNING: requested sample index {index} but only "
              f"{len(examples)} examples exist. Showing last example instead.")
        index = len(examples) - 1

    print("\n── Sample input ────────────────────────────────────────")
    print(examples[index]["input"])
    print("\n── Sample output ───────────────────────────────────────")
    print(examples[index]["output"])


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 – Verify token lengths
# ══════════════════════════════════════════════════════════════════════════════

def verify_token_lengths(data_file: str, tokenizer):
    """Load the JSONL and report token-length statistics."""
    ds = load_dataset("json", data_files=data_file, split="train")

    def _text(ex):
        return ALPACA_PROMPT.format(ex["instruction"], ex["input"], ex["output"])

    lengths = [len(tokenizer.encode(_text(ex))) for ex in ds]

    print(f"\nToken length stats for {data_file}:")
    print(f"  Max tokens   : {max(lengths)}")
    print(f"  Mean tokens  : {int(sum(lengths) / len(lengths))}")
    print(f"  Min tokens   : {min(lengths)}")
    print(f"  ≤ 512 tokens : {sum(1 for l in lengths if l <= 512):,} / {len(lengths):,}")
    print(f"  ≤ 1024 tokens: {sum(1 for l in lengths if l <= 1024):,} / {len(lengths):,}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 – Format and split dataset
# ══════════════════════════════════════════════════════════════════════════════

def save_jsonl(ds, path: str):
    """Write a HuggingFace dataset (or list of dicts) to a JSONL file."""
    with open(path, 'w') as f:
        for ex in ds:
            f.write(json.dumps(ex) + '\n')
    print(f'Saved {len(ds):,} → {path}')


def format_and_split(
    data_file: str,
    tokenizer,
    train_size: int = TRAIN_SIZE,
    eval_size: int = EVAL_SIZE,
):
    """Apply Alpaca formatting, split into train / eval, and save JSONL."""
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(batch):
        texts = []
        for instruction, input_text, output in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            texts.append(
                ALPACA_PROMPT.format(instruction, input_text, output) + eos_token
            )
        return {"text": texts}

    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"\nTotal examples: {len(dataset):,}")
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

    total = len(dataset)
    if train_size + eval_size > total:
        raise ValueError(
            f"train_size ({train_size:,}) + eval_size ({eval_size:,}) = "
            f"{train_size + eval_size:,} exceeds dataset length ({total:,}). "
            f"Reduce the split sizes or generate more data."
        )

    train_dataset = dataset.select(range(train_size))
    eval_dataset  = dataset.select(range(train_size, train_size + eval_size))
    print(f"Train: {len(train_dataset):,}  |  Val: {len(eval_dataset):,}")

    save_jsonl(train_dataset, "data/ngsim_rowbyrow_train.jsonl")
    save_jsonl(eval_dataset,  "data/ngsim_rowbyrow_eval.jsonl")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Load data
    df = load_ngsim_data("data/trajectories-0515-0530.csv")

    # 2. Generate row-by-row examples
    examples = create_rowbyrow_dataset(df)
    print("Prompt Generation Done!")
    print_sample(examples, index=100_000)

    # 3. Load tokenizer (used by both verification and formatting)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Verify token lengths
    verify_token_lengths(OUTPUT_FILE, tokenizer)

    # 5. Format and split
    format_and_split(OUTPUT_FILE, tokenizer)

if __name__ == "__main__":
    main()
