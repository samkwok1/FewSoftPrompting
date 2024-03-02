from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from ..util.processed_to_hf import csv_to_hf
import numpy as np
import evaluate
import sys

def main():
    # arg handling
    args = sys.argv[1:]
    task = args[0]
    assert task == 'piqa' or task == 'siqa' or task == 'swag', "Please ensure task is one of \{piqa, siqa, swag\}"
    num_shots = int(args[1])
    assert num_shots in [0, 1, 3, 5], "Please ensure num_shots is in \{0, 1, 3, 5\}"

    # load dataset dict for task of num_shots
    dsd = csv_to_hf(num_shots, task)

    # create tokenizer by doing what sam did
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="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")
    tokenized_datasets = dsd.map(tokenize_function, batched=True)

    # load in llama copying sam's code
    # HF code: model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    LLM_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map='auto',
                cache_dir = "./llama13b",
                token="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"
            )

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=LLM_model,
        args=training_args,
        train_dataset=dsd['train'],
        eval_dataset=dsd['validation'],
        compute_metrics=compute_metrics,
    )

    # from https://huggingface.co/docs/transformers/en/training they define a function to tokenize everything in dsd
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)



if __name__ == '__main__':
    main()