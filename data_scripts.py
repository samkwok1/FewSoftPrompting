import pandas as pd
import sys
from datasets import Dataset
import random


def get_prompt(task):
    assert task in ["sst", "quora", "sts"], "Ensure task is one of sst, quora, or sts. Task is currently: {}".format(task)
    return "Prompt: This is a prompt."

def get_file(task):
    assert task in ["sst", "quora", "sts"], "Ensure task is one of sst, quora, or sts. Task is currently: {}".format(task)
    if task == "sst":
        return "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/ids-sst-train.csv", ['sentence', 'sentiment']
    elif task == "quora":
        return "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/quora-train.csv", ['sentence1', 'sentence2', 'is_duplicate']
    elif task == "sts":
        return "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/sts-train.csv", ['sentence1', 'sentence2', 'similarity']
    assert False, "get_file did not return anything."

def get_example(df, task, label_name, prompt, num_shots):
    '''
        df: pandas data frame from csv (these are the inputs and, when not testing, the labels)
        num_shots: int for how many examples to include
        prompt: a string describing the task/prompt we want to feed to LM

        Function returns string of n-shot prompting with n examples
    
    Examples should be of the following form (for SST):
        Prompt: [insert prompt here]\n
        Q: relevant_df.iloc[random_sample]['sentence']\n
        A: relevant_df.iloc[same_random_sample]['sentiment']\n
        .
        .
        .
        Q: relevant_df.iloc[random_sample]['sentence']\n
        A: 
    '''
    # initialize example with prompt
    example = "{}\n".format(prompt)

    # randomly generate n shots of query, answer pairs...a single query is either one sentence for sst or two sentences for {quora, sts}
    for _ in range(num_shots):
        random_index = random.randint(0, len(df) - 1)

        # case distinction by task: either query is two sentences or one sentence
        if task in ['quora', 'sts']:
            example += "    sentence1: {}\n".format(df.iloc[random_index]['sentence1'])
            example += "    sentence2: {}\n".format(df.iloc[random_index]['sentence2'])
        
        # if task is sst only one sentence and answer is sentiment
        elif task in ['sst']:
            example += "    sentence: {}\n".format(df.iloc[random_index]['sentence'])
        else:
            assert False, "task is not one of quora, sts, sst"

        example += "    {}: {}\n".format(label_name, df.iloc[random_index][label_name])
    return example


def dataset_creator(task, dst, num_shots):
    # get file path, list of relevant columns, and prompt for a given task
    src, relevant_columns = get_file(task)
    prompt = get_prompt(task)
    label_name = relevant_columns[-1]

    # Load CSV file into a pandas DataFrame and extract only the relevant columns
    df = pd.read_csv(src, sep='\t')
    df = df[relevant_columns]

    # create n-shot example for each row of data frame as well as current question
    for i in range(len(df)):
        example = get_example(df, task, label_name, prompt, num_shots)
        
        # annoying case distinction again: either we have two sentences or one and further if we have two either the answer is for duplication or similarity
        if task in ['quora', 'sts']:
            example += "    sentence1: {}\n".format(df.iloc[i]['sentence1'])
            example += "    sentence2: {}\n".format(df.iloc[i]['sentence2'])

        # if task is sst only one sentence
        elif task in ['sst']:
            example += "    sentence: {}\n".format(df.iloc[i]['sentence'])
        
        # error
        else:
            assert False, "task is not one of quora, sts, sst"
        
        # add blank area for label for current query answer
        example += "    {}: \n".format(label_name)
        print(example)
    

    # hf_dataset = Dataset.from_pandas(df)


def main():
    # arg handling
    args = sys.argv[1:]
    assert len(args) == 3, "Ensure args are of the form: python data_scripts.py {sst, quora, sts} num_shots dst. This script takes in three args."
    task = args[0]
    num_shots = args[1]
    dst = int(args[2])

    # create few shot prompt dataset
    dataset_creator(task, dst=None, num_shots=3)
    

if __name__ == "__main__":
    main()