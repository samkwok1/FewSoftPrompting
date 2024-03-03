from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch

def train(model_path, model_nickname, tokenizer_path, dataset, training_params, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    print("Init model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="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")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    #  model.resize_token_embeddings(len(tokenizer))

    LLM_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map='auto',
        cache_dir = f"./{model_nickname}",
        token="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"
    )
    print(LLM_model.config)

    print("Init dataset")

    num_virtual_tokens = training_params.num_virtual_tokens
    print("Init PEFT")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=tokenizer_path
    )
    PEFT_model = get_peft_model(model=LLM_model, peft_config=peft_config)
    PEFT_model.print_trainable_parameters()


    print("Training")

    learning_rate = training_params.learning_rate
    num_epochs = training_params.num_epochs
    training_args = TrainingArguments(
        output_dir="outputs",
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs
    )
    trainer = Trainer(
        model=PEFT_model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    trainer.model.save_pretrained(save_path)