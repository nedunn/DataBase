import os
import pandas as pd
from collections import Counter
from prettytable import PrettyTable
import numpy as np

def file_types(path):
    file_types=set()
    for filename in os.listdir(path):
        extension=os.path.splitext(filename)[1]
        if extension:
            file_types.add(extension)
    return file_types

def closest_number(target, lst):
    if target == None:
        return None
    else:
        return lst[np.abs(lst - target).argmin()]

def invert_dict(mydict):
    newdict={}
    for key, value in mydict.items():
        if value not in newdict:
            newdict[value]=[key]
        else:
            newdict[value].append(key)
    return newdict

def count_duplicates(lst):
    """
    Counts the number of duplicate values in a list and returns a dictionary of values with their duplicate counts.
    Values with a count of 2 or less are dropped from the dictionary.

    Args:
        lst (list): The input list to count duplicates from.

    Returns:
        dict: A dictionary containing values with their corresponding duplicate counts, excluding values with a count of 2 or less.

    Examples:
        >>> my_list = [1, 2, 3, 3, 4, 5, 5, 5]
        >>> duplicate_counts = count_duplicates(my_list)
        >>> print(duplicate_counts)
        {5: 3}
    """
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    
    duplicate_counts = {key: value for key, value in counts.items() if value <= 2}
    return duplicate_counts

def count_duplicates(lst):
    """
    Counts the number of duplicate values in a list and returns a dictionary of values with their duplicate counts.

    Args:
        lst (list): The input list to count duplicates from.

    Returns:
        dict: A dictionary containing values with their corresponding duplicate counts.

    Examples:
        >>> my_list = [1, 2, 3, 3, 4, 5, 5, 5]
        >>> duplicate_counts = count_duplicates(my_list)
        >>> print(duplicate_counts)
        {3: 2, 5: 3}
    """
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1

    duplicate_counts = {key: value for key, value in counts.items() if value > 1}
    return duplicate_counts

def modify_duplicates(mylist):
    """
    Modifies the duplicate values in a list by replacing the second occurrence with the original value plus 0.5.

    Args:
        mylist (list): The input list to modify.

    Returns:
        list: A new list with the modifications applied.

    Examples:
        >>> mylist = [638, 639, 640, 641, 641, 642, 643, 644, 645, 646, 647, 648, 649, 649, 650, 651]
        >>> modified_list = modify_duplicates(mylist)
        >>> print(modified_list)
        [638, 639, 640, 641, 641.5, 642, 643, 644, 645, 646, 647, 648, 649, 649.5, 650, 651]
    """
    modified_list = []
    duplicates = {}

    for item in mylist:
        if item in duplicates:
            duplicates[item] += 1
            if duplicates[item] == 2:
                modified_list.append(item + 0.5)
            else:
                modified_list.append(item)
        else:
            duplicates[item] = 1
            modified_list.append(item)

    return modified_list

def list_align(lst, target_value):
        # Find the list where the target_value is closest to the beginning
    start_index = min(range(len(lst)), key=lambda i: abs(lst[i][0] - target_value))

    aligned_matrix = []
    max_length = max(len(row) for row in lst)

    for row in lst:
        # Add NaN values to the front of each row until the target value is aligned
        num_nans_front = row.index(target_value)
        aligned_row = [np.nan] * (max_length - len(row) + num_nans_front) + row

        # Pad the remaining values with NaN
        aligned_row += [np.nan] * (max_length - len(aligned_row))

        aligned_matrix.append(aligned_row)

    return aligned_matrix

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

 