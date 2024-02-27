import pandas as pd
import sys
from datasets import Dataset, DatasetDict, load_dataset
import random


def get_columns(task):
    assert task in ['piqa', 'siqa', 'swag'], "Please ensure task is one of piqa, siqa, or swag."

    if task == 'piqa':
        return ['goal', 'sol1', 'sol2', 'label']
    elif task == 'siqa':
        return ['context', 'question', 'answerA', 'answerB', 'answerC', 'label']
    elif task == 'swag':
        return ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label']
    else:
        assert False, "get_columns did not return when it should have"


def get_prompt(task):
    assert task in ['piqa', 'siqa', 'swag'], "Please ensure task is one of piqa, siqa, or swag."

    if task == 'piqa':
        return "Prompt: This is the PIQA prompt."
    elif task == 'siqa':
        return "Prompt: This is the SIQA prompt."
    elif task == 'swag':
        return "Prompt: This is the HellaSwag prompt."
    else:
        assert False, "get_prompt did not return when it should have"


def zero_shot(hf_dataset, columns, prompt):
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
        example = "{}\n\n".format(prompt)
        for column in columns[:-1]:
            example += "    {}: {}\n".format(column, hf_dataset[i][column])
        example += "    {}: \n\n".format('label')
        # print(example)
        hf_examples.append({'text': example, 'label': hf_dataset[i]['label']})
    # print(len(hf_examples) == len(hf_dataset['label']))
    # print(hf_examples)


def few_shot(hf_dataset, columns, prompt, num_shots):
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
    pass


def make_dataset(task, save, num_shots):
    '''
    piqa: has train, dev, and test splits. for test all labels come as -1. need to test on leaderboard
    swag: has train, dev, and test splits. for test all labels come as ''. need to test on leaderboard
    siqa: only has train, dev splits. need to download test question and test on leaderboard.
    '''
    # all datasets have last column name: label
    label_name = 'label'
    # get hard-coded prompt and columns of given task
    prompt = get_prompt(task)
    columns = get_columns(task)
    
    # specify dataset name for different tasks
    dataset_name = None
    if task == 'swag':
        dataset_name = 'Rowan/hellaswag'
    elif task == 'piqa':
        dataset_name = 'piqa'
    elif task == 'siqa':
        dataset_name = "social_i_qa"
    assert dataset_name is not None, "unable to assign name to dataset for particular task: {}".format(task)

    # load dataset from name and make the train, validation, and test datasets
    task_dataset = load_dataset(dataset_name)
    train_dataset, validation_dataset, test_dataset = task_dataset['train'], task_dataset['validation'], task_dataset['test']

    # make train list, validation list which are always used
    train_examples = few_shot(train_dataset, columns, prompt, num_shots)  # need to actually implement this!!!
    validation_examples = zero_shot(validation_dataset, columns, prompt)


    # always convert train and validation list into hf dataset and initialize our dsd with those
    hf_train = Dataset.from_list(train_examples)
    hf_validate = Dataset.from_list(validation_examples)
    dsd = DatasetDict({"train": hf_train, "validation": hf_validate})

    # when our task isn't siqa we also need to make hf test dataset and add it to dsd
    if task != 'siqa':
        test_examples = zero_shot(test_dataset, columns, prompt)
        hf_test = Dataset.from_list(test_examples)
        dsd['test'] = hf_test
    
    # print dsd to make sure it looks how it needs to and save it when required
    print(dsd)
    if save:
        dsd.save_to_disk("../data/processed/hf-{}".format(task))


def main():
    '''
        args should have the form: python hf_data_scripts.py {piqa, siqa, swag} save num_shots
    '''
    # arg handling
    args = sys.argv[1:]

    task = args[0]
    assert task == 'piqa' or task == 'siqa' or task == 'swag', "Please ensure task is one of \{piqa, siqa, swag\}"
    save = args[1].lower()
    assert save == 'true' or save == 'false', "save should be either true or false"
    save = True if save == 'true' else False
    num_shots = int(args[2])


    # create hf dict dataset with train, validate, test splits for piqa and swag and train, validate splits for siqa
    make_dataset(task, save, num_shots)

if __name__ == "__main__":
    main()