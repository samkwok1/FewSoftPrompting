import pandas as pd
import csv



def main(path):
    datasets = ["train.csv", "valid.csv", "test.csv"]
    for dataset in datasets:
        data = []
        with open(f"{path}/{dataset}", mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                row["text"] = row["text"].replace("(A)", str(0))
                row["text"] = row["text"].replace("(B)", str(1))
                row["label"] = 0 if row["label"] == "A)" else 1
                data.append(row)

        with open(f"{path}/2{dataset}", mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)





if __name__ == "__main__":
    path = "../data/processed/3shot/piqa"
    main(path)