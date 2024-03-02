import csv
import sys
import pandas as pd

messages = [[
    {"role": "user", "content": "Question: how to slice meat easier\n0: place the meat in the oven and broil for 10 to 15 minutes to stiffen it up\n1: place the meat in the freezer for 10 to 15 minutes to stiffen it up\nAnswer:"},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": "Question: How to clean a mirror\n0: Using windex and paper towels or newspaper.\n1: Using shoe polish and paper towels or newspaper.\nAnswer:"},
    {"role": "assistant", "content": "0"},

    {"role": "user", "content": "Question: To feel more comfortable when out in public,\n0: dress like you want to be successful when you go out.\n1: dress like you do not care about your surroundings.\nAnswer:"},
    {"role": "assistant", "content": "0"},
    
    {"role": "user", "content": "Question: How do I ready a guinea pig cage for it's new occupants?\n0:Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.\n1:Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.\nAnswer:"}
],
[   {"role": "user", "content": "Question: how to slice meat easier\n0: place the meat in the oven and broil for 10 to 15 minutes to stiffen it up\n1: place the meat in the freezer for 10 to 15 minutes to stiffen it up\nAnswer:"},
    {"role": "assistant", "content": "1"},
    {"role": "user", "content": "Question: How to clean a mirror\n0: Using windex and paper towels or newspaper.\n1: Using shoe polish and paper towels or newspaper.\nAnswer:"},
    {"role": "assistant", "content": "0"},
    {"role": "user", "content": "Question: To feel more comfortable when out in public,\n0: dress like you want to be successful when you go out.\n1: dress like you do not care about your surroundings.\nAnswer:"},
    {"role": "assistant", "content": "0"},
    {"role": "user", "content": "Question: How do I ready a guinea pig cage for it's new occupants?\n0:Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.\n1:Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.\nAnswer:"}]
]


def turn_csv_to_messages_dict(path):
    # only for validation set
    #messages_dict will have this structure:
    """
    messages_dict = {
        '0': List[List[dict]],
        '1': List[List[dict]],
        '3': List[List[dict]],
        '5': List[List[dict]]
    }
    """
    messages_dict = {}
    # for each shot value
    for shot in ['0', '1', '3', '5']:
        messages_dict[shot] = []
        csv_path = path
        with open(csv_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            # for each row in csv...has format {'prompt': 'text', 'label': true_label}
            for row in csv_reader:
                # extract prompt and label for eachh row
                prompt = row["prompt"]
                label = row["label"]
                # split the prompt by answer: ['Question:...\n0:...\n1:...\n', 'int(label)', 'Question:...\n0:...\n1:...\n']
                sections = prompt.split('Answer:')
                messages = []
                print(sections)
                for i, section in enumerate(sections):
                    # this happens on our incomplete example
                    if section == "":
                        break
                            
                    # if you're on the first section, start it from 0
                    if i == 0:
                        messages.append({"role": "user", "content": section + "Answer:"})
                    else:
                        messages.append({"role": "assistant", "content": section[0]})
                        messages.append({"role": "user", "content": section[2:] + "Answer:"})
                
                messages_dict[shot].append(messages)
    return messages_dict
 
                
                
                
        
    
    








def main():
    args = sys.argv[1:]
    #args 1 is the dataset
    task = args[0] # either piqa, siqa, or swag
    turn_df_to_messages_dict(task)

if __name__ == "__main__":
    main()