from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig
from huggingface_hub import notebook_login
import torch


PROMPT_DICT = {
    "piqa": "Goal: How to wash your face?\n0: Wet your face with the sink. Put some gentle soap onto your hands and massage it into your face evenly. Use the water from the sink to rinse the soap off.\n1: Wet your face with the sink. Put some dish soap onto your hands and massage it into your face evenly. Use the water from the sink to rinse the soap off.\nAnswer:0\nGoal: To keep taco fillings from spilling out.\n0: Line your taco shells with a napkin. Then when your shell breaks, the napkin will catch the rest of the fillings.\n1: Line your taco shells with a lettuce leaf. Then when your shell breaks, the lettuce leaf will catch the rest of the fillings.\nAnswer:1",
    "wino": "You are a helpful AI assistant that completes sentences by filling in blanks. You must choose the option that best replaces the _ character, meaning you must choose either 0 or 1. If you are unsure, or if the question and/or options are ambiguous, you must guess between the two options. Importantly, your response must begin with and be limited to ONLY your numeric answer (one single character), or else the world will end. There is exactly one correct answer.",
    "arc": "You are a helpful AI science assistant. A grade-school student is asking for your help. Choose the option that most correctly answers the question. Your answer must be either 0, 1, 2, or 3. If you are unsure, you must guess between the four options. Importantly, your response must begin with and be limited to ONLY your numeric answer (one single character), or else the world will end. There is exactly one correct answer."
}

def train_soft_prompt(model_path, model_nickname, tokenizer_path, train_dataset, train_eval_dataset, dataset_name, num_shots, training_params, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Init model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token="YOURHFTOKEN")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    LLM_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map='auto',
        cache_dir = f"./{model_nickname}",
        token="YOURHFTOKEN"
    )
    print(LLM_model.config)

    LLM_model.resize_token_embeddings(len(tokenizer))
    
    print("Init dataset")
    train_dataset, eval_dataset = train_dataset.rename_column("label", "labels"), train_eval_dataset.rename_column("label", "labels")

    # dataset_dict = train_dataset.train_test_split(test_size=0.3)
    # train_dataset, eval_dataset = dataset_dict["train"].remove_columns(["label"]), dataset_dict["test"].remove_columns(["label"])

    print("Init PEFT")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=5,
        tokenizer_name_or_path=tokenizer_path
    )
    PEFT_model = get_peft_model(model=LLM_model, peft_config=peft_config)
    PEFT_model.config.pad_token_id = tokenizer.eos_token_id
    PEFT_model.config.pad_token = tokenizer.eos_token
    PEFT_model.resize_token_embeddings(len(tokenizer))
    PEFT_model.print_trainable_parameters()


    print("Training")

    learning_rate = training_params.learning_rate
    num_epochs = training_params.num_epochs
    training_args = TrainingArguments(
        output_dir=f"outputs/{dataset_name}-{num_shots}",
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs
    )
    trainer = Trainer(
        model=PEFT_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    account = "skwoks"
    peft_model_id = f"{account}/{model_nickname}-{dataset_name}-{num_shots}shot"
    PEFT_model.push_to_hub(peft_model_id)