import pandas as pd
import sys
from datasets import Dataset
import random
import time


def get_prompt(task):
    assert task in ["sst", "quora", "sts"], "Ensure task is one of sst, quora, or sts. Task is currently: {}".format(task)
    return "Prompt: This is a prompt."

def get_file(task):
    assert task in ["sst", "quora", "sts"], "Ensure task is one of sst, quora, or sts. Task is currently: {}".format(task)

    if task == "sst":
        train_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/ids-sst-train.csv"
        validation_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/ids-sst-dev.csv"
        test_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/ids-sst-test-student.csv"
        return train_path, validation_path, test_path, ['sentence', 'sentiment']
    
    elif task == "quora":
        train_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/quora-train.csv"
        validation_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/quora-dev.csv"
        test_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/quora-test-student.csv"
        return train_path, validation_path, test_path, ['sentence1', 'sentence2', 'is_duplicate']
    
    elif task == "sts":
        train_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/sts-train.csv"
        validation_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/sts-dev.csv"
        test_path = "/Users/James/Desktop/224N_Final_Project/FewSoftPrompting/data/sts-test-student.csv"
        return train_path, validation_path, test_path, ['sentence1', 'sentence2', 'similarity']

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
    train_path, validation_path, test_path, relevant_columns = get_file(task)
    prompt = get_prompt(task)
    label_name = relevant_columns[-1]

    # Load CSV file into a pandas DataFrame and extract only the relevant columns
    df_train = pd.read_csv(train_path, sep='\t')
    df_train = df_train[relevant_columns]  # all columns relevant
    df_valid = pd.read_csv(validation_path, sep='\t')
    df_valid = df_valid[relevant_columns]  # all columns relevant
    df_test = pd.read_csv(test_path, sep='\t')
    df_test = df_test[relevant_columns[:-1]]  # all but last column relevant

    # init dict for train, validation, test
    my_dict = {'train': [], 
               'validation': [], 
                'test': []}

    # create n-shot example for each row of data frame as well as current question FOR TRAINING DATA
    for i in range(len(df_train)):
        # initialize few shot prompt examples
        example = get_example(df_train, task, label_name, prompt, num_shots)
        
        # annoying case distinction again: either we have two sentences or one and further if we have two either the answer is for duplication or similarity
        if task in ['quora', 'sts']:
            example += "    sentence1: {}\n".format(df_train.iloc[i]['sentence1'])
            example += "    sentence2: {}\n".format(df_train.iloc[i]['sentence2'])

        # if task is sst only one sentence
        elif task in ['sst']:
            example += "    sentence: {}\n".format(df_train.iloc[i]['sentence'])
        
        # error
        else:
            assert False, "task is not one of quora, sts, sst"
        
        # add blank area for label for current query answer
        example += "    {}: \n".format(label_name)

        # build up dictionary for train part: text --> example, label --> query's answer
        my_dict['train'].append({'text': example, 'label': df_train.iloc[i][label_name]})
    
    # create 0-shot example for each row of data frame for VALIDATION DATA
    for i in range(len(df_valid)):
        # initialize zero shot example (prompt only)
        example = "{}\n".format(prompt)
        
        # case distinction
        if task in ['quora', 'sts']:
            example += "    sentence1: {}\n".format(df_valid.iloc[i]['sentence1'])
            example += "    sentence2: {}\n".format(df_valid.iloc[i]['sentence2'])
        
        # if task is sst only one sentence
        elif task in ['sst']:
            example += "    sentence: {}\n".format(df_valid.iloc[i]['sentence'])
        
        # error
        else:
            assert False, "task is not one of quora, sts, sst"

        # add blank area for label for current query answer
        example += "    {}: \n".format(label_name)

        # build up dictionary for train part: text --> example, label --> query's answer
        my_dict['validation'].append({'text': example, 'label': df_valid.iloc[i][label_name]})
    
    for i in range(len(df_test)):
        # initialize zero shot example (prompt only)
        example = "{}\n".format(prompt)
        
        # case distinction
        if task in ['quora', 'sts']:
            example += "    sentence1: {}\n".format(df_test.iloc[i]['sentence1'])
            example += "    sentence2: {}\n".format(df_test.iloc[i]['sentence2'])
        
        # if task is sst only one sentence
        elif task in ['sst']:
            example += "    sentence: {}\n".format(df_test.iloc[i]['sentence'])
        
        # error
        else:
            assert False, "task is not one of quora, sts, sst"

        # add blank area for label for current query answer
        example += "    {}: \n".format(label_name)

        # build up dictionary for train part: text --> example, label --> query's answer
        my_dict['test'].append({'text': example})
    
    print(my_dict['train'][-2:])
    print(my_dict['validation'][-2:])
    print(my_dict['test'][-2:])


def main():
    # arg handling
    args = sys.argv[1:]
    assert len(args) == 3, "Ensure args are of the form: python data_scripts.py {sst, quora, sts} num_shots dst. This script takes in three args."
    task = args[0]
    dst = args[1]
    num_shots = int(args[2])
    

    # create few shot prompt dataset
    dataset_creator(task=task, dst=None, num_shots=num_shots)
    

if __name__ == "__main__":
    main()