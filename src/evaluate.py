from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import csv
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HUGGING_FACE = True

def find_output_labels(output_path):
    file = open('outputs.txt', 'r')
    lines = file.readlines()
    flag = False
    counter = 0
    outputs = []
    for line in lines:
        if flag == True:
            if counter == 8:
                flag = False
                label = 0 if '0' in line else 1
                outputs.append(label)

        if line.startswith('[INST'):
            counter = 0
            flag = True
        counter += 1
    return outputs

def turn_csv_to_messages_dict(path):
    # only for validation set
    #messages_dict will have this structure:
    """
    messages_dict = {
        '0': List[List[dict]],
        '1': List[List[dict]],
        '3': List[List[dict]],
        '5': List[List[dict]]
    }
    """
    messages_dict = {}
    labels_dict = {}
    # for each shot value
    for shot in ['0', '1', '3', '5']:
        messages_dict[shot] = []
        labels_dict[shot] = []
        with open(f"{path}/{shot}shot.csv", mode='r') as file:
            csv_reader = csv.DictReader(file)
            # for each row in csv...has format {'prompt': 'text', 'label': true_label}
            for row in csv_reader:
                # extract prompt and label for eachh row
                prompt = row["prompt"]
                labels_dict[shot].append(int(row["label"]))

                # split the prompt by answer: ['Question:...\n0:...\n1:...\n', 'int(label)', 'Question:...\n0:...\n1:...\n']
                sections = prompt.split('Answer:')
                messages = []
                for i, section in enumerate(sections):
                    # this happens on our incomplete example
                    if section == "":
                        break
                            
                    # if you're on the first section, start it from 0
                    if i == 0:
                        messages.append({"role": "user", "content": section + "Answer:"})
                    else:
                        messages.append({"role": "assistant", "content": section[0]})
                        messages.append({"role": "user", "content": section[2:] + "Answer:"})
                
                messages_dict[shot].append(messages)

    return messages_dict, labels_dict

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
        
        csv_path = f"datasets/FewSoftPrompting/preprocessed/{dataset}/validation"
        messages_dict, labels_dict = turn_csv_to_messages_dict(csv_path)
        eval_list = messages_dict[str(num_eval_shots)]
        for elem in eval_list:
            print(elem)

        predictions = []
        for i in tqdm(range(len(eval_list))):
            message = eval_list[i]
            inputs = tokenizer.apply_chat_template(message, return_tensors="pt")
            outputs = model.generate(inputs, max_new_tokens=20, temperature=0.1)
            predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

        for prediction in predictions:
            predictions1 = prediction.split("\n")
            print()
            for pred in predictions1:
                print(pred)
            print()

        preds = []
        for prediction in predictions:
            prediction = prediction.split("Answer:")
            prediction = prediction[-1]
            if '0' in prediction:
                result = 0
            elif '1' in prediction:
                result = 1
            elif '2' in prediction:
                result = 2
            elif '3' in prediction:
                result = 3
            preds.append(result)
        predictions = preds
        print(predictions)
        labels = labels_dict[str(num_eval_shots)]
        assert len(eval_list) == len(labels)
        assert len(predictions) == len(labels)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        return

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


