import csv
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def process_predictions(dataset_name, num_shots, y_hat, find_idx):
    if 'arc' in dataset_name:
        ys = ['0', '1', '2', '3']
    else:
        ys = ['0', '1']

    preds = []
    for prediction in y_hat:
        prediction = prediction.split("Answer:")
        result = -1 
        if len(prediction[find_idx]) == 0:
            preds.append(result)
            continue
        prediction = prediction[find_idx][:2]
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

    average = "macro"
    precision = precision_score(new_y, new_yhat, average=average)
    recall = recall_score(new_y, new_yhat, average=average)
    f1 = f1_score(new_y, new_yhat, average=average)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
def get_labels(dataset_name, num_eval_shots):
    path = f'data/{dataset_name}/validation/{num_eval_shots}shot.csv'
    df = pd.read_csv(path)
    labels = df["label"]
    return labels


def main():
    dataset_name = "arc_e"
    prediction_path = f"results/10_tokens/{dataset_name}"
    shots = [0, 1, 3]
    for elem in shots:
        print(f"Number of shots: {elem}")
        labels = get_labels(dataset_name, elem)
        pred_df = pd.read_csv(f"{prediction_path}/{elem}train_0eval_soft_prompted.csv")
        y_hat = pred_df["output"].to_list()
        old_length = len(y_hat)
        new_y_hat = process_predictions(dataset_name, elem, y_hat, 1)
        new_y_hat[-1] = 1
        eval_predictions(y_hat, new_y_hat, labels, dataset_name, old_length, elem)
        print()



if __name__ == "__main__":
    main()