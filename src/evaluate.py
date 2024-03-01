from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from vllm import LLM
import torch
import tqdm
import csv
import os

HUGGING_FACE = False

def get_outputs(model, inputs, max_new_tokens=2):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
    )
    return outputs

def compute_stats(model_path, model_nickname, tokenizer_path, dataset, num_eval_shots, save_path):

    if HUGGING_FACE:
        print("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Initializing model")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', cache_dir=f"./{model_nickname}")
        model.eval()
        print("Tokenizing inputs")
        
        dataset = dataset.remove_columns(['prompt'])
        dataset.set_format(type='torch', device='cuda')
        dataloader = DataLoader(dataset, batch_size=3084, collate_fn=default_data_collator)
        # A lesson in batching and having a consistent batch size here...
        output = model.generate(**dataset[0])
        tokens = tokenizer.batch_decode(output, skip_special_tokens=True)

        print("Generating eval outputs")
        generated_tokens = []
        length = len(dataloader)
        count = 0
        for batch in dataloader:
            count += 1
            if count == 3:
                break
            print(f"Item: {count}/{length}")

            with torch.no_grad():
                outputs = model.generate(**dataset, eos_token_id=tokenizer.eos_token_id)
                tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(tokens)

    else:
        print("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Initializing model")
        model = LLM(model_path, download_dir='vLLM/llama7b')
        with open('data/valid.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            prompts = [elem["prompt"] for elem in csv_reader]
            outputs = model.generate(prompts)
            completions = [output.outputs[0].text for output in outputs]
        print(completions)

