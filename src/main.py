import hydra
from omegaconf import DictConfig
import os
from train import train, train_lora
from evaluate import compute_stats
from datasets import load_from_disk

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    sim = args.sim
    env = args.env
    ouput_save_path = f"{sim.repo_path}/outputs/{env.model_description.type}/{env.dataset}/{env.num_shots}/{sim.sim_id}"
    model_save_path = f"models/{env.model_description.type}/{env.num_shots}"

    print("Initializing dataset")
    dataset = load_from_disk(f'datasets/FewSoftPrompting/hf/{env.dataset}/{env.num_shots}shot')
    hf_train_dataset = dataset["train"]
    # hf_eval_dataset = dataset["valid"] if env.dataset != "siqa" else dataset["test"]
    # hf_test_dataset = dataset["test"] if env.dataset != "siqa" else dataset["valid"]
    

    if env.train:
        train_lora(
            model_path=env.model_description.model_path,
            model_nickname=env.model_description.model_nickname,
            tokenizer_path=env.model_description.tokenizer_path,
            dataset=hf_train_dataset,
            training_params=env.training_params,
            save_path=model_save_path
        )
    if env.fine_tune:
        pass
    if env.eval:
        compute_stats(
            model_path=env.model_description.model_path,
            model_nickname=env.model_description.model_nickname,
            tokenizer_path=env.model_description.tokenizer_path,
            dataset=env.dataset,
            num_eval_shots=env.num_eval_shots,
            save_path=model_save_path
        )
    if env.test:
        pass

if __name__ == '__main__':
    main()
    