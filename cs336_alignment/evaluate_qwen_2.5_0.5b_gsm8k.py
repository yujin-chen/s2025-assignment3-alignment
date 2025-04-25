import json
import time
import re
from typing import Any
from vllm import LLM, SamplingParams
from pathlib import Path
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH
from gsm8k_parse_response import parse_gsm8k_response

# Prompt format for GSM8K
def format_prompt(example: dict[str, Any]) -> str:
    return f"{example['question']}\nAnswer:"

#Function to parse the expectd answer (gold answer) from the example
def extract_final_numeric_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def evaluate_gsm8k(model_path: str, output_file: str):
    # Load local GSM8K test data
    with open(str(DATA_PATH / "gsm8k/test.jsonl")) as f:
        dataset = [json.loads(line) for line in f]

    prompts = [format_prompt(ex) for ex in dataset]

    # Set up language model and decoding parameters
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    results = []
    start = time.time()
    batch_size = 8

     # Loop over prompts in batches and collect predictions
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            model_output = output.outputs[0].text
            example = dataset[i + j]
            parsed_answer = parse_gsm8k_response(model_output)
            gold_answer = extract_final_numeric_answer(example["answer"])
            correct = parsed_answer is not None and parsed_answer == gold_answer

            correct = str(parsed_answer) == str(gold_answer)

            results.append({
                "question": example["question"],
                "model_output": model_output.strip(),
                "gold_answer": gold_answer,
                "parsed_answer": parsed_answer,
                "correct": correct,
            })

    end = time.time()

    # Compute metrics
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    failed_parse_count = sum(1 for r in results if r["parsed_answer"] is None)
    accuracy = correct_count / total if total else 0
    throughput = total / (end - start)

    # Print metrics to terminal
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Failed to parse: {failed_parse_count}")
    print(f"Throughput: {throughput:.2f} examples/second")

    # Save only results to disk
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results



# model_path = str(KOA_PATH / "Qwen/Qwen2.5-0.5B")
# output_path = str(BEFORE_FINETUNED_RESULT_PATH / "gsm8k_results.json")


# finetuned_model_path = str(KOA_PATH / "Qwen2.5-finetuned")
# finetuned_output_path = str(FINETUNED_RESULT_PATH / "gsm8k_results_finetuned.json")

DPO_model_path = str(KOA_PATH / "qwen2.5_dpo")
DPO_output_path = str(DPO_RESULT_PATH / "gsm8k_results_DPO.json")

# # Run evaluation
# evaluate_gsm8k(model_path, output_path)
# evaluate_gsm8k(finetuned_model_path, finetuned_output_path)
evaluate_gsm8k(DPO_model_path, DPO_output_path)

'''
Baseline Model:
Accuracy: 0.3237
Failed to parse: 0
Throughput: 2.66 examples/second

Finetuned:
Accuracy: 0.2153
Failed to parse: 1
Throughput: 6.59 examples/second

DPO model:
Accuracy: 0.2146
Failed to parse: 1
Throughput: 5.56 examples/second
'''