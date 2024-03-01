from fewShotModel import FewSoftModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
from transformers import AutoTokenizer
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
                                    lambda examples: tokenizer(examples["prompt"]),
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
    exists = args[1].lower()
    assert exists == 'true' or exists == 'false', "save should be either true or false"
    save = True if save == 'true' else False
    num_shots = int(args[2])

    print("Initializing dataset")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
    if not exists:
        init_dataset(num_shots, task, tokenizer)

if __name__ == "__main__":
    main()