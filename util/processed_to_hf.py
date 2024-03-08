from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
import sys


def csv_to_hf(num_shots, task) -> DatasetDict:
    """
    Takes a task and a particular number of shots, finds the few-shot csv associated with it
    and turns it into an HF dataset. Creates and returns an HF DatasetDict to have the splits
    in one structure.

    Question: This function seems to grab, for n=3, the 3-shot train, 3-shot validate, and 0-shot
    test. Is that the intended behavior and should it be? Isn't the n-shot validate is only meant to be
    used to get LLaMa's few-shot baseline? Aside from that I thought everywhere else we only use the 0-shot
    validate to see how our model performs (as in how the trained n-shot soft-prompt performs in a 0-shot context).

    In simple words: why do we need n-shot validate at any time other than evaluating on one of our particular baselines?
    """
    path = f"~/research_projects/FewSoftPrompting/data/{task}"
    if task != "siqa":
        splits = ["train", "validation"]
    else:
        splits = ["train", "valid"]
    dsd = DatasetDict()
    for elem in splits:
        dataset = pd.read_csv(f"{path}/{elem}/{num_shots}shot.csv")
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
    tokenized_dataset.save_to_disk(f'datasets/FewSoftPrompting/hf/{task}/{num_shots}shot')


def main():
    args = sys.argv[1:]

    task = args[0]
    assert task == 'piqa' or task == 'siqa' or task == 'arc', "Please ensure task is one of \{piqa, siqa, swag\}"
    num_shots = int(args[1])

    print("Initializing dataset")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token="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")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        device_map='auto',
        cache_dir = f"./mistral7b",
        token="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"
    )

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    init_dataset(num_shots, task, tokenizer)

if __name__ == "__main__":
    main()