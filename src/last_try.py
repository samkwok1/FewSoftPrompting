from transformers import AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import DatasetDict, load_dataset, Dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
import pandas as pd
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
from fewShotModel import FewSoftModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import os
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print("Init model and tokenizer")
path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(path, token="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")
tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

# login()

LLM_model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=path,
    device_map='auto',
    cache_dir = "./llama7b",
    token="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"
)

print("Init dataset")
task = "piqa"
num_shots = 3
dataset = load_from_disk(f'datasets/FewSoftPrompting/{task}/{num_shots}shot')
dataset

print("Init PEFT")
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    # prompt_tuning_init_text=f"{INNIT_DICT_FEW_SHOT[self.task]}",
    num_virtual_tokens= 8,
    # self.num_virtual_tokens,
    tokenizer_name_or_path=path
)
PEFT_model = get_peft_model(model=LLM_model, peft_config=peft_config)
PEFT_model.print_trainable_parameters()


print("Training")
training_args = TrainingArguments(
    output_dir="outputs",
    auto_find_batch_size=True,
    learning_rate=0.0035,
    num_train_epochs=8
)
trainer = Trainer(
    model=PEFT_model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
trainer.model.save_pretrained("outputs")