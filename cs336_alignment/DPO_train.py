import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from DPO_loss import per_instance_dpo_loss
from previous_implemented.checkpoint import save_checkpoint, load_checkpoint
from previous_implemented.lr_schedule import cosine_learning_rate_schedule
from previous_implemented.gradient_clipping import gradient_clipping
from previous_implemented.AdamW import AdamW
from common import KOA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH

# Dataset
class HHDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["instruction"], item["chosen_response"], item["rejected_response"]

def format_input(prompt, response, tokenizer):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    return tokenizer(text, return_tensors="pt", padding=True)

# Fixed logprob function with masking
def compute_logprobs(logits, labels, pad_token_id):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = torch.gather(logprobs, 2, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != pad_token_id).float()
    loss = -(selected * mask).sum() / mask.sum()
    return loss

# Configuration
BETA = 0.1
LR = 1e-6
GRAD_ACCUM_STEPS = 1
VAL_INTERVAL = 20
LOG_INTERVAL = 10
VAL_SIZE = 200
OUTDIR = str(KOA_PATH / "qwen2.5_dpo")
os.makedirs(OUTDIR, exist_ok=True)

# Devices
device_train = torch.device("cuda:0")
device_ref = torch.device("cuda:1")

# Load Models
model_path = str(KOA_PATH / "Qwen2.5-finetuned")
tokenizer = AutoTokenizer.from_pretrained(model_path)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).to(device_train)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).to(device_ref)
ref_model.eval()

# Data
dataset = HHDataset(str( KOA_PATH / "hh/hh_rlhf_single_turn.jsonl"))
val_dataset = Subset(dataset, range(0, VAL_SIZE))
train_dataset = Subset(dataset, range(VAL_SIZE, len(dataset)))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LR)

# Accuracy and Validation Loss
def compute_validation_metrics():
    model.eval()
    correct, total = 0, 0
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            prompt, chosen, rejected = batch
            for i in range(len(prompt)):
                chosen_input = format_input(prompt[i], chosen[i], tokenizer)
                rejected_input = format_input(prompt[i], rejected[i], tokenizer)
                chosen_input = {k: v.to(device_train) for k, v in chosen_input.items()}
                rejected_input = {k: v.to(device_train) for k, v in rejected_input.items()}

                out_chosen = model(**chosen_input)
                out_rejected = model(**rejected_input)

                loss_chosen = compute_logprobs(out_chosen.logits[:, :-1], chosen_input["input_ids"][:, 1:], pad_token_id)
                loss_rejected = compute_logprobs(out_rejected.logits[:, :-1], rejected_input["input_ids"][:, 1:], pad_token_id)

                correct += (loss_chosen.item() < loss_rejected.item())
                total += 1
                val_losses.append((loss_chosen.item() + loss_rejected.item()) / 2.0)
    model.train()
    return correct / total, sum(val_losses) / len(val_losses)

# Train
global_step = 0
accum_count = 0
step_loss = 0.0
losses = []
steps_logged = []
val_acc_log = []
val_loss_log = []
best_val_acc = 0.0

pbar = tqdm(train_loader, desc="DPO Training")
model.train()

for batch in pbar:
    prompt, chosen, rejected = batch
    batch_loss = 0.0
    for i in range(len(prompt)):
        loss = per_instance_dpo_loss(
            model, ref_model, tokenizer, BETA,
            prompt[i], chosen[i], rejected[i]
        ).to(device_train)
        batch_loss += loss.item()
        loss.backward()
    
    avg_loss = batch_loss / len(prompt)
    if global_step % LOG_INTERVAL == 0:
        losses.append(avg_loss)
        steps_logged.append(global_step)

    accum_count += 1
    if accum_count % GRAD_ACCUM_STEPS == 0:
    
        gradient_clipping(model.parameters(), max_norm=1.0)
        
        current_lr = cosine_learning_rate_schedule(
            t=global_step,
            alpha_max=LR,
            alpha_min=LR*0.1,
            T_w=1000,  # warmup steps
            T_c=len(train_loader)*GRAD_ACCUM_STEPS  # total steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        accum_count = 0

        if global_step % LOG_INTERVAL == 0:
            print(f"[Step {global_step}] Latest Step Loss: {avg_loss:.4f}")

        if global_step % VAL_INTERVAL == 0:
            acc, val_loss = compute_validation_metrics()
            val_acc_log.append((global_step, acc))
            val_loss_log.append((global_step, val_loss))
            print(f"[Step {global_step}] Val Acc: {acc:.4f} | Val Loss: {val_loss:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                acc_str = f"{acc:.4f}".replace(".", "_")
                checkpoint_dir = os.path.join(OUTDIR, f"checkpoint-{global_step}_acc{acc_str}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_dir)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=global_step,
                    out=os.path.join(checkpoint_dir, "training_state.pt")
                )

                print(f"Saved new best checkpoint (acc={acc:.4f}) to {checkpoint_dir}")

# Final Save
model.eval()
model.save_pretrained(OUTDIR, safe_serialization=True)
tokenizer.save_pretrained(OUTDIR)
torch.save(model.state_dict(), os.path.join(OUTDIR, "model.pt"))

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump({
        "training_loss": losses,
        "steps": steps_logged,
        "val_accuracy": val_acc_log,
        "val_loss": val_loss_log,
        "final_accuracy": val_acc_log[-1][1] if val_acc_log else None,
        "final_validation_loss": val_loss_log[-1][1] if val_loss_log else None
    }, f, indent=2)
