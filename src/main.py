from fewShotModel import FewSoftModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd

def csv_to_hf(num_shots, task) -> DatasetDict:
    path = f"~/research_projects/FewSoftPrompting/data/processed/{num_shots}shot/{task}"
    if task != "siqa":
        splits = ["train", "test", "valid"]
    else:
        splits = ["train", "valid"]
    dsd = DatasetDict()
    for elem in splits:
        dataset = pd.read_csv(f"{path}/{elem}.csv")
        dsd[elem] = Dataset.from_pandas(dataset)
    return dsd

def preprocess_function(dataset, tokenizer, text="text", label="label"):
    batch_size = len(dataset[text])

    inputs = [example for example in dataset[text]]
    model_inputs = tokenizer(inputs)

    max_length = max([len(input_ids) for input_ids in model_inputs["input_ids"]])
    max_model_length = tokenizer.model_max_length
    max_length = min(max_length, max_model_length)

    targets = [str(y) for y in dataset[label]]
    labels = tokenizer(targets)
    
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        # print(model_inputs["input_ids"][i][:max_length])
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def init_dataset(num_shots, task, tokenizer):
    # self.target_max_length = max([len(self.tokenizer(class_label)) for class_label in LABELS_DICT[self.task]])
    dataset = csv_to_hf(num_shots, task)
    tokenized_dataset = {split: subset.map(
                                    lambda examples: preprocess_function(examples, tokenizer=tokenizer, text="text", label="label"),
                                    batched=True,
                                    num_proc=1,
                                    remove_columns=["label"],
                                    load_from_cache_file=False,
                                    desc=f"Running Tokenizer on {split} Dataset")
                        for split, subset in dataset.items()}
    tokenized_dataset = DatasetDict(tokenized_dataset)
    tokenized_dataset.save_to_disk(f'datasets/FewSoftPrompting/{task}/{num_shots}shot')

def main():
    # dist.init_process_group(backend='nccl')
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # torch.cuda.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
    # device = torch.device("cuda", local_rank)
    model = FewSoftModel(num_shots=3, tokenizer_path="meta-llama/Llama-2-13b-chat-hf", model_path="meta-llama/Llama-2-13b-chat-hf", task="piqa")
    print("Initializing model and tokenizer")
    model.init_LLM_n_tokenizer()
    print("Initializing dataset")

    task = "piqa"
    num_shots = 3
    dataset = False
    if not dataset:
        tokenizer = model.tokenizer
        init_dataset(num_shots, task, tokenizer)

    model.tokenized_dataset = load_from_disk(f'datasets/FewSoftPrompting/{task}/{num_shots}shot')

    print(model.tokenized_dataset)

    print("Initializing PEFT model")
    model.init_PEFT()
    print("Initializing dataloader and optimizer")
    model.init_DataLoader_n_Optimizer()
    print("Training Model")
    model.train()

if __name__ == "__main__":
    main()