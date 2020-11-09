import pickle
import json
import os
import re

import json

def isFile(filename):
    """
    :param filename: filename
    :return: boolean
    """
    return os.path.isfile(filename)

def make_dirs(path):
    """ Create directory at given path 
    
    :param path: path name
    """
    if not os.path.exists(path):
        os.makedirs(path)

def dump_to_json(data, filepath, indent=4, sort_keys=True):
    """ Dump dictionary to json file to a given filepath name 
    
    :param data: python dictionary
    :param filepath: filepath name
    :param indent: indent keys in json file
    :param sort_keys: boolean flag to sort keys
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)

def dump_to_txt(data, filepath):
    """ Dump data to txt file format to a given filepath name 
    
    :param filepath: filepath name
    """
    with open(filepath, "w") as file :
        file.write(data)

def read_txt_as_list(filepath):
    """ Load txt file content into list 
    
    :param filepath: filepath name
    :return: list of tokens
    """
    f = open(filepath, 'r+')
    data = [line.rstrip('\n') for line in f.readlines()]
    f.close()
    return data
