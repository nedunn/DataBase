import os
import pandas as pd
from collections import Counter
from prettytable import PrettyTable

def file_types(path):
    file_types=set()
    for filename in os.listdir(path):
        extension=os.path.splitext(filename)[1]
        if extension:
            file_types.add(extension)
    return file_types

def unique_count(lst):
    result={}
    for item in lst:
        if item in result:
            result[item] += 1
        else:
            result[item]=1
    return result

def find_index(data, target):

    if isinstance(data,list):
        rounded_data=[int(round(value)) for value in data]
        if target in rounded_data:
            idx=rounded_data.index(target)
            return idx
    elif isinstance(data, pd.Series):
        rounded_data=data.round().astype(int)
        if target in rounded_data.values:
            idx=rounded_data[rounded_data==target].index[0]
            return idx
    else:

        return 'Error\nEnsure data input is a series or a list containing the target value.'

def dict_summary(dict):
    stmt=f'There are {len(dict)} entires and '
    
    count_dict=Counter(list(dict.values()))
    stmt+=f'{len(count_dict)} unique values in the dictionary.'

    pt=PrettyTable()
    pt.field_names=['Dict Value','Frequency in Dict']
    for value, count in count_dict.items():
        # print(f'{count} {value}')
        pt.add_row([value,count])
    return stmt, pt

