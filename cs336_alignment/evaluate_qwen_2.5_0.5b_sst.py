import json
import time
import pandas as pd
from vllm import LLM, SamplingParams
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH
from pathlib import Path
import os

# Optional: preload system libraries if needed
os.environ["LD_PRELOAD"] = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libstdc++.so.6")

# Generate model responses to the Simple Safety Tests (SST) prompts using the given model
def generate_sst_outputs(model_path: str, output_path: str, model_name: str):
    
    # Load from local CSV file
    df = pd.read_csv( str(DATA_PATH / "simple_safety_tests/simple_safety_tests.csv"))
    prompts = df["prompts_final"].tolist()

    # Initialize Model and Sampler
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    start = time.time()
    batch_size = 8

    # Run Inference in Batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for j, output in enumerate(outputs):
            result = {
                "prompts_final": batch[j],
                "output": output.outputs[0].text.strip()
            }
            results.append(result)

    end = time.time()
    total_time = end - start
    throughput = len(prompts) / total_time
    avg_time = total_time / len(prompts)

    # Print metrics
    print(f"Generated {len(prompts)} examples in {total_time:.2f} seconds.")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Average time per example: {avg_time:.4f} sec")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved results to: {output_path}")

# Paths and execution
# model_path = str(KOA_PATH / "Qwen/Qwen2.5-0.5B")
# output_path = str(BEFORE_FINETUNED_RESULT_PATH / "sst_baseline_predictions.jsonl")
# model_name = "qwen-2.5-0.5b"

# # Use the fine-tuned model path (not base model path)
# finetuned_model_path = str(KOA_PATH / "Qwen2.5-finetuned")
# finetuned_output_path = str(FINETUNED_RESULT_PATH / "sst_baseline_predictions_finetuned.jsonl")
# finetuned_model_name = "qwen-2.5-0.5b-finetuned"

DPO_model_path = str(KOA_PATH / "qwen2.5_dpo")
DPO_output_path = str(DPO_RESULT_PATH / "sst_predictions_DPO.jsonl")
DPO_model_name = "qwen-2.5-0.5b-DPO"

# generate_sst_outputs(model_path, output_path, model_name)
# generate_sst_outputs(finetuned_model_path, finetuned_output_path, finetuned_model_name)
generate_sst_outputs(DPO_model_path, DPO_output_path, DPO_model_name)

'''
Baseline model:
Generated 100 examples in 36.12 seconds.
Throughput: 2.77 examples/sec
Average time per example: 0.3612 sec
Saved results to: ./cs336_alignment/result/sst_baseline_predictions.jsonl
safe model output proportion: 20/100


Finetuned model:
Generated 100 examples in 20.37 seconds.
Throughput: 4.91 examples/sec
Average time per example: 0.2037 sec
Saved results to: ./cs336_alignment/result/sst_baseline_predictions_finetuned.jsonl
safe model output proportion: 44/100

DPO model:
Generated 100 examples in 22.33 seconds.
Throughput: 4.48 examples/sec
Average time per example: 0.2233 sec
Saved results to: /home/yujin31/s2025-assignment3-alignment/cs336_alignment/result/DPO_result/sst_baseline_predictions_DPO.jsonl
safe model output proportion: 100/100
'''

'''
python scripts/evaluate_safety.py \
--input-path ./cs336_alignment/result/before_finetuned_result/sst_baseline_predictions.jsonl \
--model-name-or-path ../koa_scratch/ece491b-assignment3/Qwen/Qwen2.5-3B-Instruct \
--num-gpus 2 \
--output-path ./cs336_alignment/result/before_finetuned_result/qwen2.5-0.5b_finetuned_safety_eval.jsonl


python scripts/evaluate_safety.py \
--input-path ./cs336_alignment/result/finetuned_result/sst_baseline_predictions.jsonl \
--model-name-or-path ../koa_scratch/ece491b-assignment3/Qwen/Qwen2.5-3B-Instruct \
--num-gpus 2 \
--output-path ./cs336_alignment/result/finetuned_result/qwen2.5-0.5b_finetuned_safety_eval.jsonl


python scripts/evaluate_safety.py \
--input-path ./cs336_alignment/result/DPO_result/sst_predictions_DPO.jsonl \
--model-name-or-path ../koa_scratch/ece491b-assignment3/qwen2.5_dpo \
--num-gpus 2 \
--output-path ./cs336_alignment/result/DPO_result/qwen2.5-0.5b_DPO_safety_eval.jsonl
'''
