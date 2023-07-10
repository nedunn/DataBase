import os
import pandas as pd
from collections import Counter
from prettytable import PrettyTable
import numpy as np
import plotly.graph_objects as go

def save_fig(fig, path='/mnt/c/Users/16162/Desktop/temp/',name='temp_fig', type='.svg',width=1600,height=1200):
    """
    Saves a figure object as an image file, default file type is .svg.

    Parameters:
        fig (plotly.graph_objects.Figure): The figure object to be saved.
        path (str): The path to the directory where the file will be saved. Defaults to '/mnt/c/Users/16162/Desktop/temp/'.
        name (str): The name of the file to be saved. Defaults to 'temp_fig'.
        type (str): The file extension/type of the saved file. Defaults to '.svg'.
        width (int): The width of the saved image in pixels. Defaults to 1600.
        height (int): The height of the saved image in pixels. Defaults to 1200.

    Returns:
        str: A message indicating the status of the save operation.

    Raises:
        FileNotFoundError: If the specified path does not exist.

    """
    override_ans='yes'
    savefile_path=f'{path}{name}{type}'
    
    # Check path
    if os.path.isdir(path) == False:
        return 'Save To Path Error: looks like the path doesnt exist. \n\tCheck that. \n\tIf path does exist... `\_()_/`'

    # Check if file exists
    if os.path.exists(savefile_path):
        ans=input(f'The file name ({name}) already exists in the path you have specified.\n\tContinue and OVERRIDE previous file, type \'{override_ans}\'.\n\Press \'enter\' to abort save.')
        if ans == override_ans:
            fig.write_image(savefile_path, width=width, height=height)
            return f'File was saved as {savefile_path}.'
        else:
            return 'File was not saved.'
    else:
        fig.write_image(savefile_path, width=width, height=height)
        return f'File was saved as {savefile_path}.'

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
    '''Modifies the duplicate values in a list by replacing each duplicate occurrence with a new list of rounded values.

    Args:
        mylist (list): The input list to modify.

    Returns:
        list: A new list with the modifications applied. Duplicate values are replaced with rounded values.

    Examples:
        >>> mylist = [1, 2, 3, 3, 4, 4, 4]
        >>> modified_list = modify_duplicates(mylist)
        >>> print(modified_list)
        [1, 2, 3, 3.5, 4, 4.5, 5]

        >>> mylist = [1.25, 1.25, 1.25]
        >>> modified_list = modify_duplicates(mylist)
        >>> print(modified_list)
        [1.25, 1.26, 1.27]'''
    modified_list = []
    duplicates = {}
    #Inventory duplicates in list
    for item in mylist:
        if item in duplicates:
            duplicates[item] += 1
        else:
            duplicates[item] = 1
    #Build new list
    for item in duplicates:
        #Items with duplicates
        if duplicates[item] > 1:
            #Make list of replacement numbers for the duplicats
            nums=np.linspace(item, item+.9, duplicates[item])
            new_nums=np.around(nums,decimals=2) #round so that decimals arent crazy
            #Add numbers to new list
            for num in new_nums:
                modified_list.append(num)
        #Items without duplicates
        elif duplicates[item]==1:
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

def map_index(df,namedic): 
    '''Replaces the index of a DataFrame with corresponding values from a dictionary.

    Args:
        df (pandas.DataFrame): The DataFrame for which the index is to be replaced.
        namedic (dict): A dictionary containing the mapping of index values to new names.

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with the index replaced by the corresponding
        values from the dictionary.

    Example:
        >>> df = pd.DataFrame({'Col1': [1, 2, 3]}, index=[2, 3, 1])
        >>> namedic = {1: 'A', 2: 'B', 3: 'C'}
        >>> result = index_from_dic(df, namedic)
        >>> print(result)
           Col1
        A     1
        B     2
        C     3'''
    res=df.copy(deep=True)
    res.index=res.index.map(namedic.get)
    return res

def num_list_summary(lst):
    summary = {}
    summary['num_lists'] = len(lst)
    summary['list_types'] = [
        {
            'list': sublist,
            'max_value': max(sublist),
            'min_value': min(sublist),
        }
        for sublist in set(map(tuple, lst))
    ]
    summary['num_unique_lists'] = len(summary['list_types'])
    return summary
