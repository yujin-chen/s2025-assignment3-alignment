import re

def parse_gsm8k_response(model_output: str) -> str | None:

    # Find all numbers (including decimals) in the string
    numbers = re.findall(r"\d+(?:\.\d+)?", model_output)
    
    if numbers:
        return numbers[-1]  # return the last number as string
    else:
        return None
