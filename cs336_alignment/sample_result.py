import json
import random
import os
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, SAMPLED_RESULT_PATH, DPO_RESULT_PATH
# Separate lists for input and output paths
input_paths = [
    FINETUNED_RESULT_PATH /"alpacaeval_results_finetuned.json", #0 Finetuned AlpacaEval
    BEFORE_FINETUNED_RESULT_PATH /"alpacaeval_results.json", #1 Before Finetuned AlpacaEval
    FINETUNED_RESULT_PATH /"gsm8k_results_finetuned.json", #2 Finetuned GSM8K
    BEFORE_FINETUNED_RESULT_PATH /"gsm8k_results.json", #3 Before Finetuned GSM8K
    FINETUNED_RESULT_PATH /"mmlu_results_finetuned.json", #4 Finetuned MMLU
    BEFORE_FINETUNED_RESULT_PATH /"mmlu_results.json", #5 Before Finetuned MMLU
    FINETUNED_RESULT_PATH /"sst_baseline_predictions_finetuned.jsonl", #6 Finetuned SST
    BEFORE_FINETUNED_RESULT_PATH /"sst_baseline_predictions.jsonl", #7 Before Finetuned SST
    BEFORE_FINETUNED_RESULT_PATH /"qwen2.5-0.5b_not_finetuned_safety_eval.jsonl", #8 Before Finetuned SST eval
    FINETUNED_RESULT_PATH /"qwen2.5-0.5b_finetuned_safety_eval.jsonl", #9 Finetuned SST eval
    BEFORE_FINETUNED_RESULT_PATH / "before_finetuned_annotations.json", #10 Before Finetuned Alpaca Annotation 
    FINETUNED_RESULT_PATH / "finetuned_annotations.json", #11 Finetuned Alpaca Annotation 
    DPO_RESULT_PATH / "sst_predictions_DPO.json", #12 DPO SST
    DPO_RESULT_PATH / "qwen2.5-0.5b_DPO_safety_eval.jsonl", #13 DPO SST eval
    DPO_RESULT_PATH / "alpacaeval_results_DPO.json", #14 DPO AlpacaEval
    DPO_RESULT_PATH / "gsm8k_results_DPO.json", #15 DPO GSM8K

    
]

output_paths = [
    SAMPLED_RESULT_PATH /"alpacaeval_sampled_finetuned.txt", #0 Finetuned AlpacaEval
    SAMPLED_RESULT_PATH /"alpacaeval_sampled_before_tuned.txt", #1 Before Finetuned AlpacaEval
    SAMPLED_RESULT_PATH /"gsm8k_sampled_finetuned.txt", #2 Finetuned GSM8K
    SAMPLED_RESULT_PATH /"gsm8k_sampled_before_tuned.txt", #3 Before Finetuned GSM8K
    SAMPLED_RESULT_PATH /"mmlu_sampled_finetuned.txt", #4 Finetuned MMLU
    SAMPLED_RESULT_PATH /"mmlu_sampled_before_tuned.txt", #5 Before Finetuned MMLU
    SAMPLED_RESULT_PATH /"sst_baseline_predictions_sampled_finetuned.txt", #6 Finetuned SST
    SAMPLED_RESULT_PATH /"sst_baseline_predictions_sampled_before_tuned.txt", #7 Before Finetuned SST
    SAMPLED_RESULT_PATH /"sst_baseline_predictions_sampled_unsafe.txt", #8 Before Finetuned SST eval
    SAMPLED_RESULT_PATH /"sst_finetuned_predictions_sampled_unsafe.txt", #9 Finetuned SST eval
    SAMPLED_RESULT_PATH /"alpaca_baseline_dispreferred.txt", #10 Before Finetuned SST eval
    SAMPLED_RESULT_PATH /"alpaca_finetuned_dispreferred.txt", #11 Finetuned SST eval
    SAMPLED_RESULT_PATH /"sst_DPO_predictions_sampled.txt", #12 DPO SST eval
    SAMPLED_RESULT_PATH /"alpaca_DPO_dispreferred.txt", #13 DPO AlpacaEval
    SAMPLED_RESULT_PATH /"gsm8k_DPO_predictions_sampled.txt", #14 DPO GSM8K
    SAMPLED_RESULT_PATH /"mmlu_sampled_DPO.txt", #15 DPO MMLU
]

def grab_random_safe_zero_txt(input_json_path, output_txt_path, num_samples=10):

    # 1) Load data from JSON lines (one JSON object per line)
    data = []
    with open(input_json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # 2) Filter to those with safe == 0.0
    safe_zero_entries = [
        entry for entry in data
        if entry.get("metrics", {}).get("safe") == 0.0
    ]

    # 3) Randomly choose up to num_samples from the filtered list
    n = min(num_samples, len(safe_zero_entries))
    random_entries = random.sample(safe_zero_entries, n)

    # 4) Write them to a text file, each entry labeled
    with open(output_txt_path, 'w') as out_f:
        for idx, entry in enumerate(random_entries, start=1):
            out_f.write(f"Sample {idx}:\n")
            # Convert the JSON object to a nicely formatted string
            out_f.write(json.dumps(entry, indent=2))
            out_f.write("\n\n")  # blank line between samples

    print(f"Saved {n} randomly selected entries (safe=0.0) to {output_txt_path} as plain text.")


def sample_incorrect_mmlu_predictions(input_json, output_txt, num_samples=10):

    # Step 1: Load the JSON array
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Step 2: Filter entries where correct == false
    incorrect_entries = [entry for entry in data if entry.get("correct") == False]

    # Step 3: Randomly sample up to num_samples
    n = min(num_samples, len(incorrect_entries))
    sampled_incorrect = random.sample(incorrect_entries, n)

    # Step 4: Write them to a plain text file
    with open(output_txt, 'w') as out_f:
        for idx, entry in enumerate(sampled_incorrect, start=1):
            # You can customize which fields to include. Below is an example:
            out_f.write(f"Sample {idx}\n")
            out_f.write(f"Prompt: {entry.get('prompt', '')}\n")
            out_f.write(f"Model Output: {entry.get('model_output', '')}\n")
            out_f.write(f"Parsed Answer: {entry.get('parsed_answer', '')}\n")
            out_f.write(f"Gold Answer: {entry.get('gold_answer', '')}\n")
            out_f.write(f"Correct: {entry.get('correct', '')}\n")
            out_f.write("\n")  # blank line between samples

    print(f"Saved {n} random incorrect predictions to '{output_txt}'.")


def sample_incorrect_gsm8k_predictions(input_json, output_txt, num_samples=10):

    # Load the JSON array from file
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Filter out only incorrectly predicted examples
    incorrect_entries = [item for item in data if item.get('correct') == False]

    # Randomly sample up to `num_samples` from the incorrect subset
    n = min(num_samples, len(incorrect_entries))
    sampled_incorrect = random.sample(incorrect_entries, n)

    # Write them to a plain text file
    with open(output_txt, 'w') as out_f:
        for idx, entry in enumerate(sampled_incorrect, start=1):
            # You can decide which fields to include. Below is one example:
            out_f.write(f"Sample {idx}:\n")
            out_f.write(f"Question: {entry.get('question', '')}\n")
            out_f.write(f"Model Output: {entry.get('model_output', '')}\n")
            out_f.write(f"Gold Answer: {entry.get('gold_answer', '')}\n")
            out_f.write(f"Parsed Answer: {entry.get('parsed_answer', '')}\n")
            out_f.write("--------------\n\n")

    print(f"Saved {n} randomly selected incorrect examples to '{output_txt}'.")


def hh_data_sample(input_path, output_path, num_samples=3):
  
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Separate into helpful and harmless
    helpful_examples = [ex for ex in data if ex["source_file"].startswith("helpful")]
    harmless_examples = [ex for ex in data if ex["source_file"].startswith("harmless")]

    # Randomly sample examples
    random_helpful = random.sample(helpful_examples, num_samples)
    random_harmless = random.sample(harmless_examples, num_samples)

    lines = []
    for i, ex in enumerate(random_helpful, 1):
        lines.append(f"\nSample {i}:")
        lines.append(f"Instruction: {ex['instruction']}\n")
        lines.append(f"Chosen: {ex['chosen_response']}\n")
        lines.append(f"Rejected: {ex['rejected_response']}\n")
        lines.append(f"data_source: {ex['source_file']}\n")
        lines.append(f"="*50)
    
    lines.append("\n🔹 Harmless examples:")
    for i, ex in enumerate(random_harmless, 1):
        lines.append(f"\nSample {i}:")
        lines.append(f"Instruction: {ex['instruction']}\n")
        lines.append(f"Chosen: {ex['chosen_response']}\n")
        lines.append(f"Rejected: {ex['rejected_response']}\n")
        lines.append(f"data_source: {ex['source_file']}\n")
        lines.append(f"="*50)

    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))

    print(f"Saved sampled examples to: {output_path}")

def extract_dispreferred_samples(json_path, output_path, num_samples=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter where GPT-4 (generator_1) is preferred over baseline (generator_2)
    dispreferred = [
        ex for ex in data
        if ex.get("preference") == 2.0 and "output_1" in ex and "output_2" in ex
    ]

    sampled = random.sample(dispreferred, min(num_samples, len(dispreferred)))

    with open(output_path, 'w') as f:
        for i, ex in enumerate(sampled, 1):
            f.write(f"Sample {i}:\n")
            f.write(f"Instruction: {ex['instruction']}\n\n")
            f.write(f"GPT-4 Turbo Response:\n{ex['output_1']}\n\n")
            f.write(f"Baseline (Qwen) Response:\n{ex['output_2']}\n\n")
            f.write(f"Annotator Preference: GPT-4 Turbo (Rank 1)\n")
            f.write("-" * 80 + "\n")

    return output_path

def sample_unsafe_responses(jsonl_path, output_txt_path, num_samples=10):
    unsafe_entries = []

    # Read and collect unsafe entries
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("model_response") == "True" or obj.get("metrics", {}).get("safe") == 0.0:
                    unsafe_entries.append(obj)  # keep parsed object
            except json.JSONDecodeError:
                continue

    # Sample and format
    sampled = random.sample(unsafe_entries, min(num_samples, len(unsafe_entries)))
    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        for i, entry in enumerate(sampled, 1):
            out_f.write(f"Sample #{i}\n")
            out_f.write(json.dumps(entry, indent=2))  # pretty print JSON
            out_f.write("\n\n" + "-" * 80 + "\n\n")

    print(f"Wrote {len(sampled)} unsafe examples to {output_txt_path}")



# sample_unsafe_responses(str(input_paths[8]), str(output_paths[8]), num_samples = 10)
# sample_unsafe_responses(str(input_paths[9]), str(output_paths[9]), num_samples = 10)


# Run the function
extract_dispreferred_samples(str(input_paths[10]), str(output_paths[10]), num_samples = 10)
extract_dispreferred_samples(str(input_paths[11]), str(output_paths[11]), num_samples = 10)

# Usage
# hh_data_sample(input_path= str(KOA_PATH / "hh/hh_rlhf_single_turn.json"), output_path= SAMPLED_RESULT_PATH /"hh_sample_analysis.txt")


# Example usage:
# sample_incorrect_gsm8k_predictions(str(input_paths[2]), str(output_paths[2]), num_samples=10)
# sample_incorrect_gsm8k_predictions(str(input_paths[3]), str(output_paths[3]), num_samples=10)

# Example usage:
# sample_incorrect_mmlu_predictions(str(input_paths[4]), str(output_paths[4]), num_samples=10)
# sample_incorrect_mmlu_predictions(str(input_paths[5]), str(output_paths[5]), num_samples=10)

# grab_random_safe_zero_txt(input_paths[8], output_paths[8], num_samples=10)
# grab_random_safe_zero_txt(input_paths[9], output_paths[9], num_samples=10)
