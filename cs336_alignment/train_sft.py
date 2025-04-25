#!/usr/bin/env python
import os
import sys
import torch
import json
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from previous_implemented.AdamW import AdamW
from previous_implemented.checkpoint import save_checkpoint
from previous_implemented.cross_entropy import cross_entropy
from previous_implemented.lr_schedule import cosine_learning_rate_schedule
from previous_implemented.gradient_clipping import gradient_clipping
from cs336_alignment.data_loading import PackedSFTDataset, get_batch_iterator

from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH

def train():
    # Training configuration
    config = {
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "num_epochs": 1,
        "max_seq_length": 512,
        "output_dir": str(KOA_PATH / "Qwen2.5-finetuned"),
        "logging_steps": 250,
        "val_steps": 500,
        "save_steps": 3000,
        "max_steps": 6700
    }

    # Load model and tokenizer
    model_path =  str(KOA_PATH / "Qwen/Qwen2.5-0.5B")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda")

    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float32  # safer default
    # ).to("cuda")

    # Load dataset
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path= str(KOA_PATH / "ultrachat/train.jsonl"),
        seq_length=config["max_seq_length"],
        shuffle=True
    )
    val_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path= str(KOA_PATH  / "ultrachat/test.jsonl"),
        seq_length=config["max_seq_length"],
        shuffle=False
    )

    
    train_loader = get_batch_iterator(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = get_batch_iterator(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    num_training_steps = config["num_epochs"] * len(train_loader) // config["gradient_accumulation_steps"]


    alpha_max = config["learning_rate"]
    alpha_min = alpha_max * 0.1
    T_w = 0.03 * num_training_steps
    T_c = num_training_steps

    def get_lr(t):
        return cosine_learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c)

    # Initialize training variables
    model.train()
    global_step = 0
    step_loss = 0.0
    train_losses = []
    val_losses = []
    steps = []
    lr_schedule = []

    # Training loop
    for epoch in range(config["num_epochs"]):
        for step, batch in enumerate(tqdm(train_loader)):
            if global_step >= config["max_steps"]:
                print(f"Reached max_steps ({config['max_steps']}), stopping.")
                break

            # Move batch to GPU
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")

            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits

            # Compute loss
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            ) / config["gradient_accumulation_steps"]

            loss.backward()
            step_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                gradient_clipping(model.parameters(), max_norm=2.0)
                optimizer.step()
                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                optimizer.zero_grad()

                global_step += 1
                avg_loss = step_loss
                step_loss = 0.0

                # Logging
                if global_step % config["logging_steps"] == 0:
                    print(f"Step {global_step}: Loss {avg_loss:.4f} | LR: {current_lr:.2e}")
                    train_losses.append(avg_loss)
                    steps.append(global_step)
                    lr_schedule.append(current_lr)

                # Validation
                if global_step % config["val_steps"] == 0:
                    model.eval()
                    val_loss = 0
                    for val_batch in val_loader:
                        input_ids = val_batch["input_ids"].to("cuda")
                        labels = val_batch["labels"].to("cuda")
                        with torch.no_grad():
                            outputs = model(input_ids)
                            logits = outputs.logits
                            val_loss += cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1)
                            )
                    val_loss /= len(val_loader)
                    val_losses.append(val_loss.item())
                    print(f"Step {global_step}: Val Loss: {val_loss:.4f}")
                    if len(val_losses) >= 2:
                        delta = val_losses[-1] - val_losses[-2]
                        print(f"Step {global_step}: Δ Val Loss: {delta:+.4f}")
                    model.train()

                # Save checkpoint
                if global_step % config["save_steps"] == 0:
                    checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        iteration=global_step,
                        out=os.path.join(checkpoint_dir, 'checkpoint.pt')
                    )

        if global_step >= config["max_steps"]:
            break

    # Final evaluation
    model.eval()
    for val_batch in val_loader:
        input_ids = val_batch["input_ids"].to("cuda")
        labels = val_batch["labels"].to("cuda")
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            val_loss += cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
    val_loss /= len(val_loader)
    val_losses.append(val_loss.item())
    print(f"Final Val Loss: {val_loss:.4f}") 

    # Save final model and tokenizer   
    os.makedirs(config["output_dir"], exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=global_step,
        out=os.path.join(config["output_dir"], 'model.pt')
    )
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # Save training metrics
    metrics = {
        "training_losses": train_losses,
        "validation_losses": val_losses,
        "steps": steps,
        "lr_schedule": lr_schedule,
        "learning_rate": config["learning_rate"],
        "final_step": global_step
    }
    if val_losses:
        metrics["final_validation_loss"] = val_losses[-1]
    if train_losses:
        metrics["final_train_loss"] = train_losses[-1]
    with open(os.path.join(config["output_dir"], 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train()
