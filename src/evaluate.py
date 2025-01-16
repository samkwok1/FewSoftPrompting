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
    path = f'YOUR_PATH_NAME/FewSoftPrompting/data/{dataset_name}/validation/{num_eval_shots}shot.csv'
    df = pd.read_csv(path)
    labels = df["label"]
    return labels


def eval_pipeline(dataset, model_path, tokenizer_path, task='text-generation'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="YOURHFTOKEN", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    pipe = pipeline(model=model_path, tokenizer=tokenizer, task=task, device_map='auto', token="YOURHFTOKEN")
    set_seed(42)
    outputs = []
    for out in tqdm(pipe(KeyDataset(dataset, "prompt"), batch_size=32, truncation=True, max_new_tokens=2, do_sample=True, temperature=0.5)):
        outputs.append(out[0])
    return outputs

def evaluate(model_path, model_nickname, tokenizer_path, num_shots, dataset, dataset_name, num_eval_shots, save_path, type):
    if type != "base":
        device = "cuda"
        print("Init PEFT model")
        account="YOURACCOUNTNAME"        
        peft_model_id=f"{account}/{model_nickname}-{dataset_name}-{num_shots}shot"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="YOURHFTOKEN", padding_side='right')
        tokenizer.padding_side = 'right'
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
        device = "cuda"
        print("Init base model")
        model = AutoModelForCausalLM.from_pretrained(model_path, token="YOURHFTOKEN")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="YOURHFTOKEN", padding_side='right')
        tokenizer.padding_side = 'right'
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
    
    y = get_labels(dataset_name, num_eval_shots).tolist()
    old_length = len(y_hat)
    y_hat_new = process_predictions(dataset_name=dataset_name, num_shots=num_eval_shots, y_hat=y_hat, find_idx=1)
    eval_predictions(y_hat, y_hat_new, y, dataset_name, old_length, num_shots=num_eval_shots)
    save_to_csv(y_hat, y_hat_new, y, dataset_name, num_shots, num_eval_shots, type)