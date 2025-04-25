import os
import gzip
import json
import random
import re
from pathlib import Path
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH

# Check if a conversation is single-turn
def is_single_turn_string(convo: str) -> bool:
    num_human = len(re.findall(r"\bHuman:", convo))
    num_assistant = len(re.findall(r"\bAssistant:", convo))
    return num_human == 1 and num_assistant == 1

# Extract the first turn pair
def extract_first_turns(text: str):
    turns = re.findall(r"(Human|Assistant):\s*(.*?)\s*(?=\n(?:Human|Assistant):|$)", text, re.DOTALL)
    human_msg, assistant_msg = None, None

    for role, content in turns:
        if role == "Human" and human_msg is None:
            human_msg = content.strip()
        elif role == "Assistant" and assistant_msg is None:
            assistant_msg = content.strip()
        if human_msg and assistant_msg:
            break

    return human_msg, assistant_msg

# Load and filter HH dataset
def load_hh_dataset_local_single_turn_only(base_dir):
    subsets = [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled"
    ]
    combined_data = []

    for subset in subsets:
        jsonl_path = os.path.join(base_dir, subset, "train.jsonl.gz")
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found. Skipping.")
            continue

        print(f"Processing {subset}/train => {jsonl_path}")
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")

                # Filter for strictly single-turn dialogs only
                if not (is_single_turn_string(chosen) and is_single_turn_string(rejected)):
                    continue
                # Extract only the first Human-Assistant pair from each response
                user_chosen, assistant_chosen = extract_first_turns(chosen)
                user_rejected, assistant_rejected = extract_first_turns(rejected)

                # Must contain all components and same user instruction
                if not all([user_chosen, assistant_chosen, user_rejected, assistant_rejected]):
                    continue
                if user_chosen != user_rejected:
                    continue

                entry = {
                    "instruction": user_chosen,
                    "chosen_response": assistant_chosen,
                    "rejected_response": assistant_rejected,
                    "source_file": f"{subset}"
                }
                combined_data.append(entry)

    return combined_data

def main():
    base_dir = str(KOA_PATH / "hh")
    combined_data = load_hh_dataset_local_single_turn_only(base_dir)
    print(f"Total single-turn HH examples loaded: {len(combined_data)}")

    if combined_data:
        random_example = random.choice(combined_data)
        print("Sample entry:\n", json.dumps(random_example, indent=2))
    else:
        print("No valid examples found.")

    # Save to JSONL
    output_path = os.path.join(base_dir, "hh_rlhf_single_turn1.jsonl")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in combined_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(combined_data)} examples to {output_path}")

if __name__ == "__main__":
    main()
