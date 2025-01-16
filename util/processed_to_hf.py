from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
import sys


def csv_to_hf(splits, num_shots, task) -> DatasetDict:
    """
    Takes a task and a particular number of shots, finds the few-shot csv associated with it
    and turns it into an HF dataset. Creates and returns an HF DatasetDict to have the splits
    in one structure.
    """
    path = f"~/research_projects/FewSoftPrompting/data/{task}"
    dsd = DatasetDict()
    for elem in splits:
        dataset = pd.read_csv(f"{path}/{elem}/{num_shots}shot.csv")
        dsd[elem] = Dataset.from_pandas(dataset)
    return dsd


def init_dataset(shots, task, tokenizer):
    for shot in shots:
        if shot == 0:
            splits = ["train_eval", "train_train", "validation", "test"]
        else:
            splits = ["train_train", "validation"]
        dataset = csv_to_hf(splits, shot, task)
        tokenized_dataset = {split: subset.map(
                                        lambda examples: tokenizer(examples["prompt"], padding=True, return_tensors='pt', truncation=True, max_length=1024),
                                        batched=True,
                                        load_from_cache_file=False,
                                        desc=f"Running Tokenizer on {split} Dataset")
                            for split, subset in dataset.items()}
        tokenized_dataset = DatasetDict(tokenized_dataset)
        tokenized_dataset.save_to_disk(f'datasets/FewSoftPrompting/hf/{task}/{shot}shot')

    


def main():
    args = sys.argv[1:]

    task = args[0]

    print("Initializing dataset")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token="YOURHFTOKEN", padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mixtral-8x7B-v0.1",
        device_map='auto',
        cache_dir = f"./mistral8x7b",
        token="YOURHFTOKEN"
    )

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    shots = [0, 1, 3]
    init_dataset(shots, task, tokenizer)

if __name__ == "__main__":
    main()