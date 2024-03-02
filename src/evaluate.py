from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import torch
import tqdm
import csv
import os

HUGGING_FACE = True

def get_outputs(model, inputs, max_new_tokens=2):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
    )
    return outputs

def compute_stats(model_path, model_nickname, tokenizer_path, dataset, num_eval_shots, save_path):
# "    Question: how to slice meat easier
#     0: place the meat in the oven and broil for 10 to 15 minutes to stiffen it up
#     1: place the meat in the freezer for 10 to 15 minutes to stiffen it up
#     Answer:1

#     Question: How to clean a mirror
#     0: Using windex and paper towels or newspaper.
#     1: Using shoe polish and paper towels or newspaper.
#     Answer:0

#     Question: To feel more comfortable when out in public,
#     0: dress like you want to be successful when you go out.
#     1: dress like you do not care about your surroundings.
#     Answer:0

#     Question:How do I ready a guinea pig cage for it's new occupants?
#     0:Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.
#     1:Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.
#     Answer:",0
    if HUGGING_FACE:
        print("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Initializing model")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', cache_dir=f"./{model_nickname}")
        messages = [[
            {"role": "user", "content": "Question: how to slice meat easier\n0: place the meat in the oven and broil for 10 to 15 minutes to stiffen it up\n1: place the meat in the freezer for 10 to 15 minutes to stiffen it up\nAnswer:"},
            {"role": "assistant", "content": "1"},

            {"role": "user", "content": "Question: How to clean a mirror\n0: Using windex and paper towels or newspaper.\n1: Using shoe polish and paper towels or newspaper.\nAnswer:"},
            {"role": "assistant", "content": "0"},

            {"role": "user", "content": "Question: To feel more comfortable when out in public,\n0: dress like you want to be successful when you go out.\n1: dress like you do not care about your surroundings.\nAnswer:"},
            {"role": "assistant", "content": "0"},
            
            {"role": "user", "content": "Question: How do I ready a guinea pig cage for it's new occupants?\n0:Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.\n1:Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.\nAnswer:"}
        ],
        [   {"role": "user", "content": "Question: how to slice meat easier\n0: place the meat in the oven and broil for 10 to 15 minutes to stiffen it up\n1: place the meat in the freezer for 10 to 15 minutes to stiffen it up\nAnswer:"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "Question: How to clean a mirror\n0: Using windex and paper towels or newspaper.\n1: Using shoe polish and paper towels or newspaper.\nAnswer:"},
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": "Question: To feel more comfortable when out in public,\n0: dress like you want to be successful when you go out.\n1: dress like you do not care about your surroundings.\nAnswer:"},
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": "Question: How do I ready a guinea pig cage for it's new occupants?\n0:Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.\n1:Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.\nAnswer:"}]
        ]
        for message in messages:
            inputs = tokenizer.apply_chat_template(message, return_tensors="pt")
            print(tokenizer.batch_decode(inputs))
            outputs = model.generate(inputs, max_new_tokens=20)
            print(outputs)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return
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
                outputs = model.generate(**batch, eos_token_id=tokenizer.eos_token_id)
                tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                print(tokens)

    else:
        print("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
        print("Initializing model")
        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=2,
            stop=["[/assistant]"]
        )
        model = LLM('mistralai/Mistral-7B-Instruct-v0.2', download_dir='vLLM/Mistral_7B_Instruct0.2')
        with open('datasets/FewSoftPrompting/preprocessed/piqa/validation/0shot.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            prompts = [elem["prompt"] for elem in csv_reader]
            prompts = [f"{prompt}: Please pick the better answer to the question between options 0 and 1. Then, output the option you chose. If you don't output 0 or 1 as the first token, the world will end. Answer:(" for prompt in prompts]
            outputs = model.generate(prompts, sampling_params)
            completions = [output.outputs[0].text for output in outputs]
        for completion in completions:
            print(completion[0])
        outputs = []
        for completion in completions:
            zero, one = False, False
            if '0' in completion:
                zero = True
            if '1' in completion:
                one = True
            if zero and one:
                outputs.append('welp')
            elif zero and not one:
                outputs.append('0')
            elif one and not zero:
                outputs.append('1')
            else:
                outputs.append("no answer")
        print(outputs)


