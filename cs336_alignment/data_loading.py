import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from typing import Union, List

class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: Union[str, os.PathLike], seq_length: int, shuffle: bool):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.tokenized: List[int] = []

        # Load and process the dataset
        with open(dataset_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        # Format using Alpaca template
        examples = []
        for ex in raw_data:
            text = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{ex['prompt']}\n\n"
                "### Response:\n"
                f"{ex['response']}"
            )
            examples.append(text)

        # Shuffle
        if shuffle:
            import random
            random.shuffle(examples)

        # Tokenize all examples and concatenate with special tokens
        self.tokenized = []
        for example in examples:
            tokens = self.tokenizer.encode(example, add_special_tokens=False)
            # Add special tokens: 128000 at start, EOS at end
            tokens = [128000] + tokens + [self.tokenizer.eos_token_id]
            self.tokenized.extend(tokens)

        # Calculate number of sequences (each starts with 128000)
        if len(self.tokenized) == 0:
            self.num_sequences = 0
        else:
            # Subtract 1 to account for first token not being part of a sequence
            self.num_sequences = max(0, (len(self.tokenized) - 1) // seq_length)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.num_sequences}")

        start = idx * self.seq_length
        end = start + self.seq_length
        
        # Get input_ids for this sequence
        input_ids = self.tokenized[start:end]
        
        # Labels are simply input_ids shifted by 1
        labels = self.tokenized[start+1:end+1] if end+1 <= len(self.tokenized) else []
        
        # Pad if needed (only possible for last sequence)
        if len(input_ids) < self.seq_length:
            pad_length = self.seq_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_length
            labels += [self.tokenizer.pad_token_id] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def get_batch_iterator(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
