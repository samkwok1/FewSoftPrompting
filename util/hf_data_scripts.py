import pandas as pd
import sys
from datasets import Dataset, DatasetDict, load_dataset
import random
import os



def get_letter(task, label):
    if task == "piqa":
        return '0' if label == 0 else '1'
    elif task == "siqa":
        return '1' if str(label) == '1' else ('2' if label == '2' else '3')
    else:  # task is 'swag'
        return '0' if str(label) == '0' else ('1' if label == '1' else ('2' if label == '2' else '3'))


def get_columns(task):
    assert task in ['piqa', 'siqa', 'swag'], "Please ensure task is one of piqa, siqa, or swag."

    if task == 'piqa':
        return ['goal', 'sol1', 'sol2', 'label'], ['Question', '(A)', '(B)', 'Answer']
    elif task == 'siqa':
        return ['context', 'question', 'answerA', 'answerB', 'answerC', 'label'], ['Context', 'Question', '(A)', '(B)', '(C)', 'Answer']
    elif task == 'swag':
        return ['activity_label', 'ctx', 'endings', 'label'], ['Activity', 'Context', '(A)', '(B)', '(C)', '(D)', 'Answer']
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

        # either task is piqa/siqa or task is swag...swag is annoying because it has as one column a list of strings
        if task_name == 'swag':
            # ['activity_label', 'ctx', 'endings', 'label'], ['Activity', 'Context', '(A)', '(B)', '(C)', '(D)', 'Answer']
            example += "    {}: {}\n".format('Activity', hf_dataset[i]['activity_label'])
            example += "    {}: {}\n".format('Context', hf_dataset[i]['ctx'])
            example += "    {}: {}\n".format('0', hf_dataset[i]['endings'][0])
            example += "    {}: {}\n".format('1', hf_dataset[i]['endings'][1])
            example += "    {}: {}\n".format('2', hf_dataset[i]['endings'][2])
            example += "    {}: {}\n".format('3', hf_dataset[i]['endings'][3])
        else:
            for j, column in enumerate(task_columns[:-1]):
                example += "    {}:{}\n".format(column_names[j], hf_dataset[i][column])
        example += "    {}:".format('Answer')

        hf_examples.append({'prompt': example, 'label': hf_dataset[i]['label']})
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
    # initialize list of dictionaries as empty
    hf_examples = []

    # for every training example construct an n shot example
    for i in range(len(hf_dataset)):
        # init current example with prompt
        example = ""
        
        # for current example we will generate num_shots complete examples
        for _ in range(num_shots):
            random_index = random.randint(0, len(hf_dataset) - 1)

            # either task is piqa/siqa or task is swag...swag is annoying because it has as one column a list of strings
            if task_name == 'swag':
                # ['activity_label', 'ctx', 'endings', 'label'], ['Activity', 'Context', '(A)', '(B)', '(C)', '(D)', 'Answer']
                example += "    {}: {}\n".format('Activity', hf_dataset[random_index]['activity_label'])
                example += "    {}: {}\n".format('Context', hf_dataset[random_index]['ctx'])
                example += "    {}: {}\n".format('0', hf_dataset[random_index]['endings'][0])
                example += "    {}: {}\n".format('1', hf_dataset[random_index]['endings'][1])
                example += "    {}: {}\n".format('2', hf_dataset[random_index]['endings'][2])
                example += "    {}: {}\n".format('3', hf_dataset[random_index]['endings'][3])

            # generate single example using random int and the column names and values at this random int row
            else:
                for j, column in enumerate(task_columns[:-1]):
                    example += "    {}: {}\n".format(column_names[j], hf_dataset[random_index][column])
            example += "    {}:{}\n\n".format('Answer', hf_dataset[random_index]['label'])
        
        # after generating num_shot complete examples we will add our incomplete example
        if task_name == 'swag':
            example += "    {}:{}\n".format('Activity', hf_dataset[i]['activity_label'])
            example += "    {}:{}\n".format('Context', hf_dataset[i]['ctx'])
            example += "    {}:{}\n".format('0', hf_dataset[i]['endings'][0])
            example += "    {}:{}\n".format('1', hf_dataset[i]['endings'][1])
            example += "    {}:{}\n".format('2', hf_dataset[i]['endings'][2])
            example += "    {}:{}\n".format('3', hf_dataset[i]['endings'][3])
        else:
            for j, column in enumerate(task_columns[:-1]):
                example += "    {}:{}\n".format(column_names[j], hf_dataset[i][column])
        
        # now example is n_shot complete examples and an incomplete examples. Now we just need the space for answer:
        example += "    {}:".format('Answer')

        # now turn this into a dictionary and add to list
        hf_examples.append({'prompt': example, 'label': hf_dataset[i]['label']})
    return hf_examples



def make_dataset(task, save, num_shots):
    '''
    piqa: has train, dev, and test splits. for test all labels come as -1. need to test on leaderboard
    swag: has train, dev, and test splits. for test all labels come as ''. need to test on leaderboard
    siqa: only has train, dev splits. need to download test question and test on leaderboard.
    '''
    # init for purposes of defining it in conditional and being able to refer to it outside
    test_examples = None
    hf_test = None
    df_test = None
    # get hard-coded prompt and columns of given task
    # prompt = get_prompt(task)
    task_columns, column_names = get_columns(task)
    
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
    train_dataset, validation_dataset, test_dataset = task_dataset['train'], task_dataset['validation'], None
    if task != 'siqa':
        test_dataset = task_dataset['test']
    print(train_dataset.features["label"])

    # make train list, validation list which are always used
    train_examples = few_shot(train_dataset, task_columns, column_names, task, num_shots)  # need to check implementation of this!!!
    validation_examples = zero_shot(validation_dataset, task_columns, column_names, task)


    # always convert train and validation list into hf dataset and initialize our dsd with those
    hf_train = Dataset.from_list(train_examples)
    hf_validate = Dataset.from_list(validation_examples)
    dsd = DatasetDict({"train": hf_train, "validation": hf_validate})  # REAL ONE...EVENTUALLY UNCOMMENT
    # dsd = DatasetDict({"train": hf_validate, "validation": hf_validate})  # FOR DEBUGGING USE

    # when our task isn't siqa we also need to make hf test dataset and add it to dsd
    if task != 'siqa':
        test_examples = zero_shot(test_dataset, task_columns, column_names, task)
        hf_test = Dataset.from_list(test_examples)
        dsd['test'] = hf_test

    # Convert each hf dataset to Pandas DataFrame
    df_train = hf_train.to_pandas()
    df_validate = hf_validate.to_pandas()
    if task != "siqa":
        df_test = hf_test.to_pandas()

    # # Save the DataFrame to CSV
    if save:
        if task != "siqa":
            splits = ["train", "test", "valid"]
            datasets = [df_train, df_validate, df_test]
        else:
            splits = ["train", "valid"]
            datasets = [df_train, df_validate]
        for dataset, dataframe in zip(splits, datasets):
            csv_save_path = f"./../data/processed/{task}/{num_shots}shot"
            os.makedirs(csv_save_path, exist_ok=True)
            dataframe.to_csv(f"{csv_save_path}/{dataset}.csv", index=False)
    else:
        return dsd


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