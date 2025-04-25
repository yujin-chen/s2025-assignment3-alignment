import re
from typing import Any

def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    # First try to extract a letter directly
    match = re.search(r"\b([ABCD])\b", model_output.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # If no letter found, try to match answer text
    options = mmlu_example.get("options", [])
    for idx, choice in enumerate(options):
        # Use word boundaries or punctuation to match exact text
        pattern = r"\b{}\b".format(re.escape(choice.lower()))
        if re.search(pattern, model_output.lower()):
            return "ABCD"[idx]

    return None  # If no match found
