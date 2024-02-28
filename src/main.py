from fewShotModel import FewSoftModel

def main():
    model = FewSoftModel(num_shots=3, tokenizer_path="meta-llama/Llama-2-13b-chat-hf", model_path="meta-llama/Llama-2-13b-chat-hf", task="siqa")
    model.init_LLM_n_tokenizer()
    model.init_PEFT()
    model.init_DataLoader_n_Optimizer()
    model.train()

if __name__ == "__main__":
    main()