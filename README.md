### What this is:
This is the repository for the paper/project: [Few-Shot Prompt Tuning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/JamesJMoriceSamuelEdwardKwok.pdf)
# **FewSoftPrompting**
*Author(s): Sam Kwok, James Morice*  
*Affiliation(s): Stanford University Department Department of Computer Science*  
*Email: [samkwok@stanford.edu](mailto:your.email@example.com)*  

---

## **Abstract**
Fine-tuning is becoming increasingly expensive and time consuming due to the
scaling of language models. As a result, different approaches to improve model
performance have arisen. We will take two of these approaches and combine
them. The first approach is prompt-tuning, a method of parameter efficient finetuning (PEFT). The second method is few-shot prompting. In this paper we
demonstrate novel experimental results: while few-shot prompt-tuning for small
language models doesn’t outperform traditional few-shot prompting for causal
text generation tasks, small language models are able to consistently generate
task-relevant text for option-based, question answering tasks in zero-shot settings
when they are prompt-tuned in a few-shot contexts.

---

## **Repository Structure**
```plaintext
~/
│
├── data/                 
│   ├── arc_c       # Arc-Challenge dataset
│   ├── arc_e       # Arc-Easy dataset
│   ├── wino        # Winogrande dataset
│
├── results/              
│   ├── graphs/           # Visualization of the results per dataset
│   ├── 5-tokens/        # Contains model outputs for n-shots per dataset, for a soft-prompt of length 5
│   ├── 10-tokens/        # Contains model outputs for n-shots per dataset, for a soft-prompt of length 10
│   └── 20-tokens/        # Contains model outputs for n-shots per dataset, for a soft-prompt of length 20
│
├── src/
│   ├── config/           # Contains the .yaml config file (for use with hydra)
│   ├── evalute.py        # Queries the FewSoftPrompted model specified in config to get results
│   ├── train.py          # Trains a FewSoftPrompted model given specifics in config
│   ├── fewShotModel.py   # Contains the FewShotModel class 
│   └── main.py           # All functionality runs through main, aside from plot generation and preprocessing
│
├── util/                 
│   ├── preprocess.py      # Converts datasets into usable format
│   ├── processed_to_hf.py # Converts preprocessed datasets to tokenized hf datasets
│   ├── plots.py           # Code for graphs
│   └── get_scores.py      # Returns evaluation metrics from model outputs
```
## **How to use this Repository**
If you wish to train your own model:
- Modify config.yaml to specify what model you would like to use as your base (from hf)
- Specify FSP characteristics (n-shots to train on, SP length, etc)
- Run main.py, and results will populate
Before you run:
- Make sure that you have a hf account + token
- Replace instances of account and token with your info
- Specify output directories in main if desired

Reach out with questions!
 

