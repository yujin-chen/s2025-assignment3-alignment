import json
import time
from vllm import LLM, SamplingParams
from pathlib import Path
import os
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH
os.environ["LD_PRELOAD"] = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libstdc++.so.6")


def evaluate_alpacaeval(model_path: str, output_path: str, model_name: str):
    # Load AlpacaEval data
    with open(str(DATA_PATH/"alpaca_eval/alpaca_eval.jsonl")) as f:
        dataset = [json.loads(line) for line in f]

    prompts = [ex["instruction"] for ex in dataset]
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    results = []
    start = time.time()

    batch_size = 8
    # Loop over `prompts` in increments of `batch_size` so we can process multiple
    # examples in parallel, which is faster than single-example processing.
    for i in range(0, len(prompts), batch_size):
        # Extract a "batch" of prompts 
        batch = prompts[i:i + batch_size]
        
        # Generate outputs for the current batch using the vllm LLM
        outputs = llm.generate(batch, sampling_params)
        
        # For each returned output in this batch, retrieve and store the result
        for j, output in enumerate(outputs):
            # Extract the actual text of the first output and strip leading/trailing whitespace
            model_output = output.outputs[0].text.strip()
            
            # Match the generated output back to the corresponding example in the dataset
            example = dataset[i + j]
            
            # Append the result to list
            results.append({
                "instruction": example["instruction"],
                "output": model_output,
                "generator": model_name,
                "dataset": "alpaca_eval"
            })


    end = time.time()
    total_time = end - start
    throughput = len(prompts) / total_time
    avg_time_per_example = total_time / len(prompts)

    print(f"Generated {len(prompts)} examples in {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Avg. time per example: {avg_time_per_example:.4f} seconds")

    # Save results to disk
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2) 
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")



# # Paths
# model_path = str(KOA_PATH / "Qwen/Qwen2.5-0.5B")
# output_path = str(BEFORE_FINETUNED_RESULT_PATH / "alpacaeval_results.json")
# model_name = "qwen-2.5-0.5b"

# # Use the fine-tuned model path (not base model path)
# finetuned_model_path = str(KOA_PATH / "Qwen2.5-finetuned")
# finetuned_output_path = str(FINETUNED_RESULT_PATH / "alpacaeval_results_finetuned.json")
# finetuned_model_name = "qwen-2.5-0.5b-finetuned"

DPO_model_path = str(KOA_PATH / "qwen2.5_dpo")
DPO_output_path = str(DPO_RESULT_PATH / "alpacaeval_results_DPO.json")
DPO_model_name = "qwen-2.5-0.5b-DPO"


# Run
# evaluate_alpacaeval(model_path, output_path, model_name)
# evaluate_alpacaeval(finetuned_model_path, finetuned_output_path, finetuned_model_name)
evaluate_alpacaeval(DPO_model_path, DPO_output_path, DPO_model_name)

'''
part (b) result :
Baseline model:
Generated 805 examples in 281.84s
Throughput: 2.86 examples/second
Avg. time per example: 0.3501 seconds
Results saved to ./cs336_alignment/result/before_finetuned_result/alpacaeval_results.json


fintuned model:
Generated 805 examples in 157.20s
Throughput: 5.12 examples/second
Avg. time per example: 0.1953 seconds
Results saved to ./cs336_alignment/result/finetuned_result/alpacaeval_results_finetuned.json

DPO model:
Generated 805 examples in 174.76s
Throughput: 4.61 examples/second
Avg. time per example: 0.2171 seconds

alpaca_eval \
  --model_outputs ./cs336_alignment/result/before_finetuned_result/alpacaeval_results.json \
  --annotators_config scripts/alpaca_eval_vllm_llama3_70b_fn \
  --base-dir .

alpaca_eval \
  --model_outputs ./cs336_alignment/result/finetuned_result/alpacaeval_results_finetuned.json \
  --annotators_config scripts/alpaca_eval_vllm_llama3_70b_fn \
  --base-dir .


  alpaca_eval \
  --model_outputs ./cs336_alignment/result/DPO_result/alpacaeval_results_DPO.json \
  --annotators_config scripts/alpaca_eval_vllm_llama3_70b_fn \
  --base-dir .
'''
