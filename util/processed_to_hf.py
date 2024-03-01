from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
import sys


def csv_to_hf(num_shots, task) -> DatasetDict:
    path = f"~/research_projects/FewSoftPrompting/data/processed/{task}/{num_shots}shot"
    if task != "siqa":
        splits = ["train", "test", "valid"]
    else:
        splits = ["train", "valid"]
    dsd = DatasetDict()
    for elem in splits:
        dataset = pd.read_csv(f"{path}/{elem}.csv")
        dsd[elem] = Dataset.from_pandas(dataset)
    return dsd


def init_dataset(num_shots, task, tokenizer):
    # self.target_max_length = max([len(self.tokenizer(class_label)) for class_label in LABELS_DICT[self.task]])
    dataset = csv_to_hf(num_shots, task)
    tokenized_dataset = {split: subset.map(
                                    lambda examples: tokenizer(examples["prompt"], padding=True, return_tensors='pt'),
                                    batched=True,
                                    remove_columns=["label"],
                                    load_from_cache_file=False,
                                    desc=f"Running Tokenizer on {split} Dataset")
                        for split, subset in dataset.items()}
    tokenized_dataset = DatasetDict(tokenized_dataset)
    tokenized_dataset.save_to_disk(f'datasets/FewSoftPrompting/{task}/{num_shots}shot')


def main():
    args = sys.argv[1:]

    task = args[0]
    assert task == 'piqa' or task == 'siqa' or task == 'swag', "Please ensure task is one of \{piqa, siqa, swag\}"
    num_shots = int(args[1])

    print("Initializing dataset")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        device_map='auto',
        cache_dir = f"./llama7b",
        token="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"
    )

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    init_dataset(num_shots, task, tokenizer)

if __name__ == "__main__":
    main()