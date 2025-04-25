import json
import time
import os
import traceback
import csv
from typing import Any
from datasets import Dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH
from mmlu_parse_response import parse_mmlu_response

# List of all available MMLU subjects from local files
MMLU_TEST_DIR = str(DATA_PATH / "mmlu/test")
ALL_MMLU_SUBJECTS = [f.replace("_test.csv", "") for f in os.listdir(MMLU_TEST_DIR) if f.endswith("_test.csv")]

SYSTEM_PROMPT ="""\
Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {option_0}
B. {option_1}
C. {option_2}
D. {option_3}
Answer:"""

#  Loads all MMLU test sets across subjects, performing basic validation and formatting.
def load_all_mmlu_tests():
    all_datasets = []
    for subject in ALL_MMLU_SUBJECTS:
        try:
            print(f"Loading subject: {subject}")
            csv_path = os.path.join(MMLU_TEST_DIR, f"{subject}_test.csv")
            
            # Verify file exists and is readable
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"File not found: {csv_path}")
            if not os.access(csv_path, os.R_OK):
                raise PermissionError(f"No read permissions for: {csv_path}")
            
            examples = []
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:  # question + 4 options + answer
                        raise ValueError(f"Invalid row format in {csv_path}, line: {row}")
                    
                    # Join question parts in case they contain commas
                    question = ','.join(row[:-5]).strip()
                    options = [opt.strip() for opt in row[-5:-1]]
                    answer = row[-1].strip()
                    
                    if answer not in ['A','B','C','D']:
                        raise ValueError(f"Invalid answer '{answer}' in {csv_path}, expected A-D")
                    
                    if len(options) != 4:
                        raise ValueError(f"Expected 4 options in {csv_path}, got {len(options)}")
                    
                    examples.append({
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "answer": answer
                    })
            
            # Convert to Hugging Face dataset format
            ds = Dataset.from_list(examples)
            all_datasets.append(ds)
        except Exception as e:
            print(f"Failed to load subject '{subject}': {str(e)}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            continue
            
    if not all_datasets:
        raise RuntimeError("Failed to load any MMLU test subjects")
    return concatenate_datasets(all_datasets)



# Format the prompt for the model
def format_prompt(example: dict[str, Any]) -> str:
    subject = example["subject"]
    question = example["question"]
    options = example["options"]
    return SYSTEM_PROMPT.format(
    subject=subject,
    question=question,
    option_0=options[0],
    option_1=options[1],
    option_2=options[2],
    option_3=options[3],
    )

# MMLU evaluation function
def evaluate_mmlu(model_path: str, output_file: str):
    # Load dataset and generate prompts
    dataset = load_all_mmlu_tests()
    prompts = [format_prompt(ex) for ex in dataset]

    # Initialize model and inference parameters
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    results = []
    start = time.time()
    batch_size = 8

    # Batched generation loop
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            model_output = output.outputs[0].text
            example = dataset[i + j]
            parsed_letter = parse_mmlu_response(example, model_output)
            gold_answer = example["answer"]  # Already converted to A-D


            results.append({
                "prompt": output.prompt,
                "model_output": model_output,
                "parsed_answer": parsed_letter,
                "gold_answer": gold_answer,
                "correct": parsed_letter == gold_answer,
            })

    end = time.time()

    # Evaluation Summary
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    failed_parse = sum(1 for r in results if r["parsed_answer"] is None)
    accuracy = correct / total if total else 0
    throughput = total / (end - start)

    # Print metrics to terminal
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Failed parse: {failed_parse}/{total}")
    print(f"Throughput: {throughput:.2f} examples/sec")

    # Save only results to disk
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # model_path = str(KOA_PATH /"Qwen/Qwen2.5-0.5B")
    # output_path = str(BEFORE_FINETUNED_RESULT_PATH / "mmlu_results.json")

    # finetuned_model_path = str(KOA_PATH / "Qwen2.5-finetuned")
    # finetuned_output_path = str(FINETUNED_RESULT_PATH / "mmlu_results_finetuned.json")

    DPO_model_path = str(KOA_PATH / "qwen2.5_dpo")
    DPO_output_path = str(DPO_RESULT_PATH / "mmlu_results_DPO.json")

    # evaluate_mmlu(model_path, output_path)
    # evaluate_mmlu(finetuned_model_path, finetuned_output_path)
    evaluate_mmlu(DPO_model_path, DPO_output_path)
'''
Baseline Model
Accuracy: 0.4644
Failed parse: 0/14042
Throughput: 37.98 examples/sec

finetuned:
Accuracy: 0.4541
Failed parse: 0/14042
Throughput: 36.77 examples/sec

DPO model:
Accuracy: 0.4541
Failed parse: 0/14042
Throughput: 35.40 examples/sec
'''