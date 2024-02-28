from fewShotModel import FewSoftModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import os

def main():
    # dist.init_process_group(backend='nccl')
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # torch.cuda.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
    # device = torch.device("cuda", local_rank)
    print("sdfsdfsdfsdfsdfsdfsdfsdf")
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