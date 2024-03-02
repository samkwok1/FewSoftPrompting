import pandas as pd
import sys
from datasets import Dataset, DatasetDict, load_dataset
import random
import os
import csv


def get_prompt(task):
    if task == "piqa":
        return "You are a helpful AI assistant that is being asked a question by a user. Choose the response that best satisfies the question. You must choose either 0 or 1. If you are unsure, or if the question and/or options are ambiguous, you must guess between the two options. Importantly, your response must begin with and be limited to ONLY your numeric answer (one single character), or else the world will end. There is exactly one correct answer."
    elif task == "arc":
        return "You are a helpful AI assistant that is being asked a question by a user. Choose the response that best satisfies the question. Your answer must be either 0, 1, 2, or 3. If you are unsure, or if the question and/or options are ambiguous, you must guess between the two options. Importantly, your response must begin with and be limited to ONLY your numeric answer (one single character), or else the world will end. There is exactly one correct answer."
    else:  # task is wino
        return "You are a helpful AI assistant that is being asked a question by a user. Choose the response that best completes the question. You must choose either 0 or 1. If you are unsure, or if the question and/or options are ambiguous, you must guess between the two options. Importantly, your response must begin with and be limited to ONLY your numeric answer (one single character), or else the world will end. There is exactly one correct answer."


def get_label_name(task):
    if task == "piqa":
        return "label"
    elif task == "arc":
        return "answerKey"
    else:
        return "answer"


def get_columns(task):
    assert task in ['piqa', 'arc', 'wino'], "Please ensure task is one of piqa, arc, or wino."

    if task == 'piqa':
        return ['goal', 'sol1', 'sol2', 'label'], ['Question', '0', '1', 'Answer']
    elif task == 'arc':  # gotta figure out arc
        return ['id', 'question', 'choices', 'answerKey'], ['Context', 'Question', '1', '2', '3', 'Answer']
    elif task == 'wino':
        return ['sentence', 'option1', 'option2', 'answer'], ['Sentence', '0', '1', 'Answer']
    else:
        assert False, "get_columns did not return when it should have"


def zero_shot(hf_dataset, task_columns, column_names, task_name):
    '''
    Given a hugging face dataset, this function should return a list of dictionaries where the first key
    is 'text' and the associated value is the zero_shot example and the second key is 'label' with value
    equal to the label for the current query (for test_datasets this will eithher be -1 or ''). Each example
    should have the following form:
    Prompt: prompt\n

        Column1: column1 text
        Column2: column2 text
        .
        .
        .
        Column_n: column_n text
        Label: label value
    '''
    hf_examples = []
    for i in range(len(hf_dataset)):
        # how example is always started
        example = ""
        for j, column in enumerate(task_columns[:-1]):
            example += "{}: {}\n".format(column_names[j], hf_dataset[i][column])
        example += "{}:".format('Answer')
        hf_examples.append({'prompt': example, 'label': hf_dataset[i]['answer']})
        print(example)
    print(hf_examples)
    return hf_examples


def few_shot(hf_dataset, task_columns, column_names, task_name, num_shots):
    '''
    Given a hugging face (training) dataset, this function should randomly sample num_shot examples to
    display as complete examples after the prompt statement. After these complete examples, it should
    include the current incomplete example and allow a space for the model to respond with its answer.
    We treat this entire structure as a single 'example' and build a list of dictionaries where
    len(list_of_dicts) = len(hf_dataset['label]). Each dictionary will have two keys: one will be 'text'
    with its associated value being the generated example and the other key will be 'label' with its
    associated value being the true label for the incomplete part of the example. We return this built
    list of dictionaries
    '''
    if task_name == "wino":
        return zero_shot(hf_dataset, task_columns, column_names, task_name)
    # initialize list of dictionaries as empty
    hf_examples = []

    # for every training example construct an n shot example
    for i in range(len(hf_dataset)):
        # init current example with prompt
        example = ""
        
        # for current example we will generate num_shots complete examples
        for _ in range(num_shots):
            # random index for random sampling to create few shot
            random_index = random.randint(0, len(hf_dataset) - 1)

            # taking care of arc case
            # if task_name == "arc":  buggy rn
            #     example += "{}: {}\n".format('Question', hf_dataset[random_index]['question'])
            #     example += "{}: {}\n".format('0', hf_dataset[random_index]['choices']['text'][0])
            #     example += "{}: {}\n".format('1', hf_dataset[random_index]['choices']['text'][1])
            #     example += "{}: {}\n".format('2', hf_dataset[random_index]['choices']['text'][2])
            #     example += "{}: {}\n".format('3', hf_dataset[random_index]['choices']['text'][3])
            #     example += "{}\n".format(get_prompt(task_name))
            #     example += "{}:{}\n".format('Answer', hf_dataset[random_index][get_label_name(task_name)])
            
            # taking care of wino case
            if task_name == "wino":
                example += "{}: {}\n".format('Sentence', hf_dataset[random_index]['sentence'])
                example += "{}: {}\n".format('0', hf_dataset[random_index]['option1'])
                example += "{}: {}\n".format('1', hf_dataset[random_index]['option2'])
                example += "{}\n".format(get_prompt(task_name))
                example += "{}:{}\n".format('Answer', str(int(hf_dataset[random_index][get_label_name(task_name)])-1))
            
            # take care of piqa case
            else:
                for j, column in enumerate(task_columns[:-1]):
                    example += "{}: {}\n".format(column_names[j], hf_dataset[random_index][column])
                example += "{}\n".format(get_prompt(task_name))
                example += "{}:{}\n".format('Answer', hf_dataset[random_index][get_label_name(task_name)])
        
        # buggy arc case
        if task_name == "arc":
            example += "{}: {}\n".format('Question', hf_dataset[i]['question'])
            example += "{}: {}\n".format('0', hf_dataset[i]['choices']['text'][0])
            example += "{}: {}\n".format('1', hf_dataset[i]['choices']['text'][1])
            example += "{}: {}\n".format('2', hf_dataset[i]['choices']['text'][2])
            example += "{}: {}\n".format('3', hf_dataset[i]['choices']['text'][3])
            example += "{}\n".format(get_prompt(task_name))
            example += "{}:".format('Answer')
            hf_examples.append({'prompt': example, 'label': hf_dataset[i][get_label_name(task_name)]})
        
        # take care of wino case
        elif task_name == "wino":
            example += "{}: {}\n".format('Sentence', hf_dataset[i]['sentence'])
            example += "{}: {}\n".format('0', hf_dataset[i]['option1'])
            example += "{}: {}\n".format('1', hf_dataset[i]['option2'])
            example += "{}\n".format(get_prompt(task_name))
            example += "{}:".format('Answer')
            hf_examples.append({'prompt': example, 'label': str(int(hf_dataset[i][get_label_name(task_name)])-1)})

        else:  # functional piqa case
            for j, column in enumerate(task_columns[:-1]):
                example += "{}: {}\n".format(column_names[j], hf_dataset[i][column])
            # now example is n_shot complete examples and an incomplete examples. Now we just need the space for answer:
            example += "{}\n".format(get_prompt(task_name))
            example += "{}:".format('Answer')
            hf_examples.append({'prompt': example, 'label': hf_dataset[i][get_label_name(task_name)]})
        # now turn this into a dictionary and add to list
    return hf_examples

def few_shot_to_csv(hf_dataset, task_columns, column_names, task_name, num_shots, split):
    data = few_shot(hf_dataset, task_columns, column_names, task_name, num_shots)
    fields = list(data[0].keys())
    csv_dir_path = "datasets/FewSoftPrompting/preprocessed/{}/{}".format(task_name, split)
    os.makedirs(csv_dir_path, exist_ok=True)
    with open(f"{csv_dir_path}/{num_shots}shot.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, quoting=csv.QUOTE_NONNUMERIC)
        
        # Write the header
        writer.writeheader()
        
        # Write the data
        for row in data:
            writer.writerow(row)


def make_dataset(task, save):
    '''
    piqa: has train, dev, and test splits. for test all labels come as -1. need to test on leaderboard
    wino: has train, dev, and test splits. for test all labels come as empty. need to test on leaderboard
    arc: has train, dev, and test splits. answer key is included for test set so testing on leader_boards isn't required
    '''
    # init for purposes of defining it in conditional and being able to refer to it outside
    # test_examples = None
    # hf_test = None
    # df_test = None
    # get hard-coded prompt and columns of given task
    # prompt = get_prompt(task)
    task_columns, column_names = get_columns(task)
    
    # specify dataset name for different tasks
    dataset_name = None
    if task == 'wino':
        dataset_name = 'winogrande'
    elif task == 'piqa':
        dataset_name = 'piqa'
    elif task == 'arc':
        dataset_name = "ai2_arc"
    assert dataset_name is not None, "unable to assign name to dataset for particular task: {}".format(task)

    # load dataset from name and make the train, validation, and test datasets
    if task == "wino":
        task_dataset = load_dataset(dataset_name, "winogrande_debiased")
    elif task == "arc":
        task_dataset = load_dataset(dataset_name, "ARC-Challenge")
    else:
        task_dataset = load_dataset(dataset_name)
    train_dataset, validation_dataset, test_dataset = task_dataset['train'], task_dataset['validation'], task_dataset['test']
    print(train_dataset.features[get_label_name(task)])

    # make train list, validation list which are always used
    # train_examples = few_shot(train_dataset, task_columns, column_names, task, 2)
    # validation_examples = zero_shot(validation_dataset, task_columns, column_names, task)
    
    # do few shot for each of these and save to csv
    if save:
        for nshots in [0, 1, 3, 5]:
            few_shot_to_csv(train_dataset, task_columns, column_names, task, nshots, "train")
            few_shot_to_csv(validation_dataset, task_columns, column_names, task, nshots, "validation")
        few_shot_to_csv(test_dataset, task_columns, column_names, task, 0, "test")

    # # always convert train and validation list into hf dataset and initialize our dsd with those
    # hf_train = Dataset.from_list(train_examples)
    # hf_validate = Dataset.from_list(validation_examples)
    # dsd = DatasetDict({"train": hf_train, "validation": hf_validate})  # REAL ONE...EVENTUALLY UNCOMMENT
    # # dsd = DatasetDict({"train": hf_validate, "validation": hf_validate})  # FOR DEBUGGING USE

    # # when our task isn't siqa we also need to make hf test dataset and add it to dsd
    # if task != 'siqa':
    #     test_examples = zero_shot(test_dataset, task_columns, column_names, task)
    #     hf_test = Dataset.from_list(test_examples)
    #     dsd['test'] = hf_test

    # # Convert each hf dataset to Pandas DataFrame
    # df_train = hf_train.to_pandas()
    # df_validate = hf_validate.to_pandas()
    # if task != "siqa":
    #     df_test = hf_test.to_pandas()

    # # Save the DataFrame to CSV
    # if save:
    #     if task != "siqa":
    #         splits = ["train", "test", "valid"]
    #         datasets = [df_train, df_validate, df_test]
    #     else:
    #         splits = ["train", "valid"]
    #         datasets = [df_train, df_validate]
    #     for dataset, dataframe in zip(splits, datasets):
    #         csv_save_path = f"{dataset}.csv"
    #         # f"./../data/processed/{task}/{num_shots}shot"
    #         os.makedirs(f"data", exist_ok=True)

    #         dataframe.to_csv(f"data/{csv_save_path}", index=False)
    # else:
    #     return dsd


def main():
    '''
        args should have the form: python hf_data_scripts.py {piqa, arc, wino} save
    '''
    # arg handling
    args = sys.argv[1:]

    task = args[0]
    assert task == 'piqa' or task == 'arc' or task == 'wino', "Please ensure task is one of \{piqa, arc, wino\}"
    save = args[1].lower()
    assert save == 'true' or save == 'false', "save should be either true or false"
    save = True if save == 'true' else False


    make_dataset(task, save)

if __name__ == "__main__":
    main()