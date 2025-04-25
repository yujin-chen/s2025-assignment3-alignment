import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

def format_input(prompt: str, response: str, tokenizer):

    alpaca_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:\n{response}{eos}"
    )
    input_text = alpaca_template.format(
        prompt=prompt.strip(),
        response=response.strip(),
        eos=tokenizer.eos_token
    )
    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        add_special_tokens=True
    )
    return tokens

# Computes log-probabilities of the given labels under the logits.
def compute_logprobs(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    # Select the log-probs for the token indices provided in labels.
    selected_logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Sum over token positions.
    return selected_logprobs.sum(dim=-1)

def per_instance_dpo_loss(lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:

    lm.eval()
    lm_ref.eval()

    with torch.no_grad():
        # Format inputs
        chosen_input = format_input(prompt, response_chosen, tokenizer)
        rejected_input = format_input(prompt, response_rejected, tokenizer)

    device = next(lm.parameters()).device
    chosen_input = {k: v.to(device) for k, v in chosen_input.items()}
    rejected_input = {k: v.to(device) for k, v in rejected_input.items()}

    # Compute number of tokens for the prompt 
    # This ensures that we slice exactly at the start of the response.
    prefix = f"{prompt.strip()} "
    prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    n_prefix = len(prefix_tokens)

    # Generate lm outputs for chosen and rejected examples.
    chosen_out = lm(**chosen_input)
    rejected_out = lm(**rejected_input)

    logprob_chosen = compute_logprobs(
        chosen_out.logits[:, n_prefix-1:-1],
        chosen_input["input_ids"][:, n_prefix:]
    )
    logprob_rejected = compute_logprobs(
        rejected_out.logits[:, n_prefix-1:-1],
        rejected_input["input_ids"][:, n_prefix:]
    )

    # Compute log-probs under the reference lm.
    ref_device = next(lm_ref.parameters()).device
    with torch.no_grad():
        chosen_input_ref = {k: v.to(ref_device) for k, v in chosen_input.items()}
        rejected_input_ref = {k: v.to(ref_device) for k, v in rejected_input.items()}
        chosen_out_ref = lm_ref(**chosen_input_ref)
        rejected_out_ref = lm_ref(**rejected_input_ref)
        logprob_chosen_ref = compute_logprobs(
            chosen_out_ref.logits[:, n_prefix-1:-1],
            chosen_input_ref["input_ids"][:, n_prefix:]
        )
        logprob_rejected_ref = compute_logprobs(
            rejected_out_ref.logits[:, n_prefix-1:-1],
            rejected_input_ref["input_ids"][:, n_prefix:]
        )

    # Compute the (log) ratios for the lm and reference.
    # The difference between them is then passed through the negative log-sigmoid.
    pi_logratios = beta * (logprob_chosen - logprob_rejected)
    ref_logratios = beta * (logprob_chosen_ref - logprob_rejected_ref)
    ref_logratios = ref_logratios.to(pi_logratios.device)
    loss = -F.logsigmoid(pi_logratios - ref_logratios)

    # Return the mean over the batch 
    return loss.mean()
