import pandas as pd
import sys
from datasets import Dataset
import random


def get_example(df, num_shots, prompt):
    '''
        df: pandas data frame from csv (these are the inputs and, when not testing, the labels)
        num_shots: int for how many examples to include
        prompt: a string describing the task/prompt we want to feed to LM

        Function returns string of n-shot prompting with n examples
    
    Examples should be of the following form:
        Prompt: [insert prompt here]\n
        Q: relevant_df.iloc[random_sample]['sentence']\n
        A: relevant_df.iloc[same_random_sample]['sentiment']\n
        .
        .
        .
        Q: relevant_df.iloc[random_sample]['sentence']\n
        A: 
    '''
    example = "{}\n".format(prompt)
    for i in range(num_shots):
        random_index = random.randint(0, len(df) - 1)
        example += "    Query: {}\n".format(df.iloc[random_index]['sentence'])
        example += "    Answer: {}\n".format(df.iloc[random_index]['sentiment'])
    return example


def dataset_creator(src, dst, num_shots):
    # Load your CSV file into a pandas DataFrame
    df = pd.read_csv(src, sep='\t')

    # for SST train/dev only two relevant columns: sentence and sentiment
    relevant_columns = ['sentence', 'sentiment']
    df = df[relevant_columns]

    prompt = '''Prompt: Perform sentiment analysis on the last question using the prior questions as examples. 
        Assign it a value with 0 being negative, 1 being slightly negative,  2 being neutral, 3 being slightly positive, and 4 being positive'''
    for i in range(len(df)):
        example = get_example(df, num_shots, prompt)
        example += "    Query: {}\n".format(df.iloc[i]['sentence'])
        example += "    Answer: \n"
        # print((example, df.iloc[i]['sentiment']))
        print(example)
    

    # hf_dataset = Dataset.from_pandas(df)


def main():
    args = sys.argv[1:]
    src = args[0]
    # dst = args[1]

    dataset_creator(src, dst=None, num_shots=3)
    


if __name__ == "__main__":
    main()