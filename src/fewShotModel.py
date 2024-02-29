from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import DatasetDict, load_dataset, Dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
import pandas as pd
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
LABELS_DICT = {
    "siqa": ['1', '2', '3'],
    "piqa": ['0', '1'],
    "swag": ['0', '1', '2', '3']
}

LABELS_MAP = {
    "piqa": {'A)': 0, 'B)': 1}
}

INNIT_DICT_FEW_SHOT = {
    "siqa": """
    Please find the best answer to the question posed. Three example completions are provided next.
    Context: Sydney got Sasha's picture taken secretly after silently stalking them throughout the town.
    Question: What does Sydney need to do before this?
    0: needed to find Sasha
    1: print it
    2: needed to be invisible
    Answer:(A)

    Context: Quinn, while playing with his new toy, accidentally broke their mother\'s favorite vase and quickly tried to buy a new one.
    Question: How would Quinn feel afterwards?
    (A): thought their mom could use a new vase
    (B): thought their toy was more important
    (C): ashamed to break things
    Answer:(C)

    Context: Robin came home from a good vacation and was relaxed and rested.
    Question: How would Robin feel afterwards?
    (A): as smart
    (B): as normal
    (C): relieved afterwards
    Answer:(B)

    """,
    "piqa": """
    Please find the best answer to the question posed. Two example completions are provided next.
    Question: How do I make sure that chocolate nutella tart is easy to cut before serving?
    0: Take out the tart from the fridge about 10 minutes before serving so it is easier to cute
    1: Take out the tart from the pastry wheel about 10 minutes before serving so it is easier to cute
    Answer:0

    Question: Scrub rough stains on bathroom tile.
    0: Apply scrub to a blow dryer.
    1: Apply scrub to a power drill.
    Answer1

    """,
    "swag": """
    Please find the best answer to the question posed. Four example completions are provided next.
    Activity: Philosophy and Religion
    Context: [header] How to understand the degrees of the palm [title] Determine this by how far down the palm the heart line is. [step] The further away from the fingers it is, the more sympathetic and understanding the person will be. If the heart line hugs close to the fingers, the person will be unfeeling, unsympathetic and critical.
    (A): If the heart line touches as little as a fingernail, it may not. If the heart line touches as close to the fingertips as possible and the patient gets a bit sick, the patient is more mobile and understanding of the hands and hands.
    (B): [title] Find the length of the heart line. [step] If you are in a different position, you may not know how far the heart line almost extends.
    (C): [title] Understand that the palm end of a folded handkerchief comes into to an acute angle of 5 degrees. [step] The iliac crest connecting the body to the palm lines is travelling above this point.
    (D): [title] Calculate the distance between the heart and head lines, in what is called the quadrangle, to find the degree of extroversion. [substeps] If the heart line and head lines are close together, the person will be more introverted than not.
    Answer:(D)

    Activity: Putting in contact lenses
    Context: The video begins with various words and instructions on how to insert & remove halloween contact lenses. a contact lens container
    (A): is replaced with a compact one and the lens is placed into a screen.
    (B): is then filled with contact solution and the contacts into both sides of the container.
    (C): is shown and an attachment is taken off by a dvd device.
    (D): transfers contact time with audiosol.
    Answer:(B)

    Activity: Home and Garden
    Context: [header] How to wash a baseball cap by hand [title] Get a container of water ready. [step] You can use a clean bucket, but a kitchen or bathroom sink is also fine. Fill it with water.
    (A): Run your hands under the running water using your old hands. [substeps] You should sit up straight in the bucket.
    (B): You only need a few milliliters (or gallons) of water. Water can act as a natural degreasing agent and is cheap.
    (C): It's best to use cool water. If the cap is really dirty, you can use slightly warm water.
    (D): Let it sit for 30 minutes before washing. As the water evaporates, the cap will get wrinkled and begin peeling.
    Answer:(C)

    Activity: Computers and Electronics
    Context: [header] How to get rid of baby acne [title] Wash your baby's skin with water and mild baby soap. [step] Wash your baby's face with warm water on a daily basis. For severe baby acne, a mild soap may also be used.
    (A): [substeps] Use soap formulated for babies whenever possible. Soaps meant for adolescents or adults may be too harsh for your baby's skin.
    (B): There is no specific cleansing unit in your home, however, and your baby's skin is sensitive to water and light chemicals. [substeps] If you intend to get baby acne at home, then your local supermarket might produce soaps that can be used to help this process along.
    (C): This can help replenish some of the dead skin cells and also reduce the appearance of heavy acne bumps on your baby's face. [substeps] Warm water too cold can increase the appearance of dead skin, depending on the age of your baby.
    (D): If your child's skin is very oily, use a mild baby soap or baby moisturizer formulated for oily skin. Rubbing your baby's face after washing will also reduce the likelihood of irritation and prevent wrinkles.
    Answer:(A)

    """
}
class FewSoftModel():
    def __init__(self,
                 num_shots: int,
                 tokenizer_path: str,
                 model_path: str,
                 task: str,) -> None:
        
        self.task = task
        self.num_shots = num_shots
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path

        self.tokenized_dataset = None
        self.tokenizer = None
        self.num_virtual_tokens = None
        self.LLM_model = None
        self.PEFT_model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.lr_scheduler = None
        self.optimizer = None
        self.num_epochs = None
    
    def init_LLM_n_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, token="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")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token

        # login()

        self.LLM_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            device_map='auto',
            cache_dir = "./llama13b",
            token="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"
        )

        self.num_virtual_tokens = len(self.tokenizer(f"{INNIT_DICT_FEW_SHOT[self.task]}")["input_ids"])
        print(f"num_tokens: {self.num_virtual_tokens}")

    def init_PEFT(self): 
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            # prompt_tuning_init_text=f"{INNIT_DICT_FEW_SHOT[self.task]}",
            num_virtual_tokens= 8,
            # self.num_virtual_tokens,
            tokenizer_name_or_path=self.tokenizer_path,
        )

        self.PEFT_model = get_peft_model(model=self.LLM_model, peft_config=peft_config)
        self.PEFT_model.print_trainable_parameters()

    def init_DataLoader_n_Optimizer(self):
        train_ds = self.tokenized_dataset["train"]
        eval_ds = self.tokenized_dataset["valid"]

        batch_size = 4
        # train_sampler = DistributedSampler(train_ds)
        self.train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=self.collate_fn, batch_size=batch_size, pin_memory=True)
        self.eval_dataloader = DataLoader(eval_ds, collate_fn=self.collate_fn, batch_size=batch_size, pin_memory=True) 

        lr = 3e-2
        self.num_epochs = 50

        self.optimizer = torch.optim.AdamW(self.LLM_model.parameters(), lr=lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        )
    
    def collate_fn(self, batch):
        # Use the transformers collator to handle the tokenization padding
        assert all(isinstance(sample, dict) for sample in batch), "Batch items must be of type dict."
        assert all('label' in sample for sample in batch)

        for sample in batch:
            sample['label'] = LABELS_MAP[self.task][sample['label']]

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, return_tensors='pt')
        batch = data_collator(batch)
        batch["labels"] = batch["labels"].unsqueeze(-1)

        # Convert label strings to integers
        return batch
    
    def train(self):
        training_args = TrainingArguments(
            output_dir="outputs",
            auto_find_batch_size=True,
            learning_rate=0.0035,
            num_train_epochs=8
        )
        trainer = Trainer(
            model=self.PEFT_model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["valid"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        trainer.train()
        trainer.model.save_pretrained("outputs")

    def train_custom_loop(self):
        device = "cuda"
        torch.cuda.empty_cache()
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")
            self.PEFT_model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                # print(f"prefix_labels shape: {prefix_labels.shape}")
                batch["labels"] = batch["labels"].unsqueeze(-1)
                outputs = self.PEFT_model(**batch)
                print(outputs)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.PEFT_model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.PEFT_model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(self.eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

