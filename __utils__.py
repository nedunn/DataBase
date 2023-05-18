import os

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