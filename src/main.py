from fewShotModel import FewSoftModel

def main():
    model = FewSoftModel(num_shots=3, tokenizer_path="meta-llama/Llama-2-13b-chat-hf", model_path="meta-llama/Llama-2-13b-chat-hf", task="piqa")
    print("Initializing model and tokenizer")
    model.init_LLM_n_tokenizer()
    print("Initializing dataset")
    model.init_dataset()
    print("Initializing PEFT model")
    model.init_PEFT()
    print("Initializing dataloader and optimizer")
    model.init_DataLoader_n_Optimizer()
    print("Training Model")
    model.train()

if __name__ == "__main__":
    main()