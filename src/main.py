import hydra
from omegaconf import DictConfig
import os
from train import train_soft_prompt, train_lora
from evaluate import evaluate
from datasets import load_from_disk

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    sim = args.sim
    env = args.env
    ouput_save_path = f"{sim.repo_path}/outputs/{env.model_description.type}/{env.dataset}/{env.num_shots}/{sim.sim_id}"
    model_save_path = f"models/{env.model_description.type}/{env.num_shots}"

    print("Initializing dataset")
    if env.num_shots != env.num_eval_shots:
        hf_train_dataset = load_from_disk(f'datasets/FewSoftPrompting/hf/{env.dataset}/{env.num_shots}shot')["train"]
        hf_eval_dataset = load_from_disk(f'datasets/FewSoftPrompting/hf/{env.dataset}/{env.num_eval_shots}shot')["validation"]
    else:
        dataset = load_from_disk(f'datasets/FewSoftPrompting/hf/{env.dataset}/{env.num_shots}shot')
        hf_train_dataset = dataset["train"]
        hf_eval_dataset = dataset["validation"]

    if env.num_shots == 0:
        hf_test_dataset = dataset["test"]
    else:
        print("More than 0 shots were specified, no test dataset initialized. Continuing...")
    

    if env.train:
        train_soft_prompt(
            model_path=env.model_description.model_path,
            model_nickname=env.model_description.model_nickname,
            tokenizer_path=env.model_description.tokenizer_path,
            dataset=hf_train_dataset,
            dataset_name=env.dataset,
            num_shots=env.num_shots,
            training_params=env.training_params,
            save_path=model_save_path
        )
    if env.fine_tune:
        pass
    if env.eval:
        evaluate(
            model_path=env.model_description.model_path,
            model_nickname=env.model_description.model_nickname,
            tokenizer_path=env.model_description.tokenizer_path,
            dataset=hf_eval_dataset,
            dataset_name=env.dataset,
            num_shots=env.num_shots,
            num_eval_shots=env.num_eval_shots,
            save_path=model_save_path,
            type=env.model_description.type
        )
    if env.test:
        pass

if __name__ == '__main__':
    main()
    