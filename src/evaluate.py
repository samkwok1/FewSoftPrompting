from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, AutoModelForSequenceClassification, PretrainedConfig, pipeline, set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers.pipelines.pt_utils import KeyDataset
import torch
import pandas as pd
from tqdm import tqdm
import csv
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from peft import PeftModel, PeftConfig

def save_to_csv(y_hat, y_hat_processed, y, dataset_name, num_shots, num_eval_shots, type):
    save_dir = os.path.expanduser(f"~/research_projects/FewSoftPrompting/results/{dataset_name}")

    headers = ["output", "correct_label"]
    os.makedirs(save_dir, exist_ok=True)
    csv_data = []
    assert len(y_hat) == len(y)
    for i in range(len(y_hat_processed)):
        csv_data.append([y_hat[i], y[i]])

    with open(f"{save_dir}/{num_shots}train_{num_eval_shots}eval_{type}.csv", "w", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

def process_predictions(dataset_name, num_shots, y_hat, find_idx):
    if 'arc' in dataset_name:
        ys = ['0', '1', '2', '3']
    else:
        ys = ['0', '1']

    preds = []
    for prediction in y_hat:
        prediction = prediction.split("Answer:")
        result = -1 
        if len(prediction[num_shots + find_idx]) == 0:
            preds.append(result)
            continue
        prediction = prediction[num_shots + find_idx][:2]
        for y in ys:
            if y in prediction:
                result = int(y)
        preds.append(result)
    return preds

def eval_predictions(old_y_hat, y_hat, y, dataset_name, old_length, num_shots):
    if 'arc' in dataset_name:
        ys = [0, 1, 2, 3]
    else:
        ys = [0, 1]

    accuracy = accuracy_score(y, y_hat)
    print(f"Accuracy: {accuracy}")
    print_dict = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
    for elem in y_hat:
        print_dict[elem] += 1
    print(print_dict)
    print(f"Old_length: {len(y_hat)}")
    new_yhat = []
    new_y = []
    for i, pred in enumerate(y_hat):
        if pred in ys:
            new_yhat.append(pred)
            new_y.append(y[i])
    print(f"New length: {len(new_yhat)}")
    new_length = len(new_y)
    print(f"Incorrect Generations: {(old_length - new_length) / old_length}")

    num_same = {y: 0 for y in ys}
    num_total = {y: 0 for y in ys}
    if num_shots > 0:
        shot_label = process_predictions(dataset_name, num_shots, old_y_hat, 0)
        for i in range(len(shot_label)):
            if shot_label[i] == y_hat[i]:
                num_same[shot_label[i]] += 1
            num_total[shot_label[i]] += 1
    
        for key, value in num_same.items():
            print(f"Percentage of similar values when previous label is {key}: {value / num_total[key]}")

    average = "macro"
    precision = precision_score(new_y, new_yhat, average=average)
    recall = recall_score(new_y, new_yhat, average=average)
    f1 = f1_score(new_y, new_yhat, average=average)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    
def get_labels(dataset_name, num_eval_shots):
    path = f'~/research_projects/FewSoftPrompting/data/{dataset_name}/validation/{num_eval_shots}shot.csv'
    df = pd.read_csv(path)
    labels = df["label"]
    return labels


def eval_pipeline(dataset, model_path, tokenizer_path, task='text-generation'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="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", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    pipe = pipeline(model=model_path, tokenizer=tokenizer, task=task, device_map='auto')
    set_seed(42)
    outputs = []
    for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, "prompt"), batch_size=32, truncation=True, max_new_tokens=2, do_sample=False))):
        outputs.append(out[0])
    return outputs

def evaluate(model_path, model_nickname, tokenizer_path, num_shots, dataset, dataset_name, num_eval_shots, save_path, type):
    if type != "base":
        device = "cuda"
        print("Init PEFT model")
        account="skwoks"        
        peft_model_id=f"{account}/{model_nickname}-{dataset_name}-{num_shots}shot"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="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", padding_side='left')
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

        model.to(device)
        model.eval()
        batch_size = 100

        total_batches = len(dataset["input_ids"]) // batch_size + (0 if len(dataset["input_ids"]) % batch_size == 0 else 1)
        y_hat = []
        with torch.no_grad():
            for batch_idx in tqdm(range(total_batches)):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_ids = dataset["input_ids"][start_idx:end_idx]
                batch_masks = dataset["attention_mask"][start_idx:end_idx]
                
                input_ids = torch.tensor(batch_ids, dtype=torch.int64).to(device)
                attention_mask = torch.tensor(batch_masks, dtype=torch.int64).to(device)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    repetition_penalty=0.5,
                    temperature=0.5
                )
                y_hat.extend(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
                del input_ids
                del attention_mask
                torch.cuda.empty_cache()

    else:
        print(model_path)
        y_hat = eval_pipeline(dataset=dataset, model_path=model_path, tokenizer_path=tokenizer_path)
        y_hat = [elem["generated_text"] for elem in y_hat]
    
    y = get_labels(dataset_name, num_eval_shots).tolist()
    old_length = len(y_hat)
    y_hat_new = process_predictions(dataset_name=dataset_name, num_shots=num_eval_shots, y_hat=y_hat, find_idx=1)
    eval_predictions(y_hat, y_hat_new, y, dataset_name, old_length, num_shots=num_eval_shots)
    save_to_csv(y_hat, y_hat_new, y, dataset_name, num_shots, num_eval_shots, type)



# def find_output_labels(output_path):
#     file = open('outputs.txt', 'r')
#     lines = file.readlines()
#     flag = False
#     counter = 0
#     outputs = []
#     for line in lines:
#         if flag == True:
#             if counter == 8:
#                 flag = False
#                 label = 0 if '0' in line else 1
#                 outputs.append(label)

#         if line.startswith('[INST'):
#             counter = 0
#             flag = True
#         counter += 1
#     return outputs

# def turn_csv_to_messages_dict(path):
#     # only for validation set
#     #messages_dict will have this structure:
#     """
#     messages_dict = {
#         '0': List[List[dict]],
#         '1': List[List[dict]],
#         '3': List[List[dict]],
#         '5': List[List[dict]]
#     }
#     """
#     messages_dict = {}
#     labels_dict = {}
#     # for each shot value
#     for shot in ['0', '1', '3', '5']:
#         messages_dict[shot] = []
#         labels_dict[shot] = []
#         with open(f"{path}/{shot}shot.csv", mode='r') as file:
#             csv_reader = csv.DictReader(file)
#             # for each row in csv...has format {'prompt': 'text', 'label': true_label}
#             for row in csv_reader:
#                 # extract prompt and label for eachh row
#                 prompt = row["prompt"]
#                 labels_dict[shot].append(int(row["label"]))

#                 # split the prompt by answer: ['Question:...\n0:...\n1:...\n', 'int(label)', 'Question:...\n0:...\n1:...\n']
#                 sections = prompt.split('Answer:')
#                 messages = []
#                 for i, section in enumerate(sections):
#                     # this happens on our incomplete example
#                     if section == "":
#                         break
                            
#                     # if you're on the first section, start it from 0
#                     if i == 0:
#                         messages.append({"role": "user", "content": section + "Answer:"})
#                     else:
#                         messages.append({"role": "assistant", "content": section[0]})
#                         messages.append({"role": "user", "content": section[2:] + "Answer:"})
                
#                 messages_dict[shot].append(messages)

#     return messages_dict, labels_dict

# def get_outputs(model, inputs, max_new_tokens=2):
#     outputs = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=max_new_tokens,
#     )
#     return outputs

# def compute_stats(model_path, model_nickname, tokenizer_path, dataset, num_eval_shots, save_path):
#     if HUGGING_FACE:
#         print("Initializing tokenizer")
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         print("Initializing model")
#         model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', cache_dir=f"./{model_nickname}")
        
#         csv_path = f"datasets/FewSoftPrompting/preprocessed/{dataset}/validation"
#         messages_dict, labels_dict = turn_csv_to_messages_dict(csv_path)
#         eval_list = messages_dict[str(num_eval_shots)]
#         for elem in eval_list:
#             print(elem)

#         predictions = []
#         for i in tqdm(range(len(eval_list))):
#             message = eval_list[i]
#             inputs = tokenizer.apply_chat_template(message, return_tensors="pt")
#             outputs = model.generate(inputs, max_new_tokens=20, temperature=0.1)
#             predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

#         for prediction in predictions:
#             predictions1 = prediction.split("\n")
#             print()
#             for pred in predictions1:
#                 print(pred)
#             print()

#         preds = []
#         for prediction in predictions:
#             prediction = prediction.split("Answer:")
#             prediction = prediction[-1]
#             if '0' in prediction:
#                 result = 0
#             elif '1' in prediction:
#                 result = 1
#             elif '2' in prediction:
#                 result = 2
#             elif '3' in prediction:
#                 result = 3
#             preds.append(result)
#         predictions = preds
#         print(predictions)
#         labels = labels_dict[str(num_eval_shots)]
#         assert len(eval_list) == len(labels)
#         assert len(predictions) == len(labels)

#         accuracy = accuracy_score(labels, predictions)
#         precision = precision_score(labels, predictions, average='macro')
#         recall = recall_score(labels, predictions, average='macro')
#         f1 = f1_score(labels, predictions, average='macro')
#         print(f"Accuracy: {accuracy}")
#         print(f"Precision: {precision}")
#         print(f"Recall: {recall}")
#         print(f"F1 Score: {f1}")
        
#         return

#     else:
#         print("Initializing tokenizer")
#         tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
#         print("Initializing model")
#         sampling_params = SamplingParams(
#             temperature=0,
#             top_p=0.95,
#             max_tokens=2,
#             stop=["[/assistant]"]
#         )
#         model = LLM('mistralai/Mistral-7B-Instruct-v0.2', download_dir='vLLM/Mistral_7B_Instruct0.2')
#         with open('datasets/FewSoftPrompting/preprocessed/piqa/validation/0shot.csv', 'r') as file:
#             csv_reader = csv.DictReader(file)
#             prompts = [elem["prompt"] for elem in csv_reader]
#             prompts = [f"{prompt}: Please pick the better answer to the question between options 0 and 1. Then, output the option you chose. If you don't output 0 or 1 as the first token, the world will end. Answer:(" for prompt in prompts]
#             outputs = model.generate(prompts, sampling_params)
#             completions = [output.outputs[0].text for output in outputs]
#         for completion in completions:
#             print(completion[0])
#         outputs = []
#         for completion in completions:
#             zero, one = False, False
#             if '0' in completion:
#                 zero = True
#             if '1' in completion:
#                 one = True
#             if zero and one:
#                 outputs.append('welp')
#             elif zero and not one:
#                 outputs.append('0')
#             elif one and not zero:
#                 outputs.append('1')
#             else:
#                 outputs.append("no answer")
#         print(outputs)


