import gzip
import json
import random
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, SAMPLED_RESULT_PATH
# Load and sample 10 random examples
path =  str(KOA_PATH / "ultrachat/train.jsonl.gz")
output_path = str(SAMPLED_RESULT_PATH / "look_at_sft_sample.txt")

with gzip.open(path, "rt", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

samples = random.sample(lines, 10)
print(samples)

with open(output_path, "w", encoding="utf-8") as out_file:
    for i, ex in enumerate(samples):
        out_file.write(f"Example {i+1}:\n")
        out_file.write(f"Prompt: {ex['prompt']}\n")
        out_file.write(f"Response: {ex['response']}\n")
        out_file.write("=" * 80 + "\n")

print(f"10 random samples written to {output_path}")
