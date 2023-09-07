"""
This script contains general classes and functions used for the data preparation step. The functions are sorted in
alphabetical order.
"""
import os
import re
import shutil
import logging
import numpy as np
from pydicom.multival import MultiValue


class Logger:
    def __init__(self, output_filename=None):
        logging.basicConfig(filename=output_filename, format='%(asctime)s - %(message)s', level=logging.INFO,
                            filemode='w')

    def my_print(self, message, level='info'):
        """
        Manual print operation.

        Args:
            message: input string.
            level: level of logging.

        Returns:

        """
        if level == 'info':
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            logging.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            logging.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        print(print_message)

    def close(self):
        logging.shutdown()


def copy_folder(src, dst):
    """
    Copy contents of source (src) folder to destination (dst) folder.

    Note that the contents of the src folder are copied, but not the src folder itself! Therefore, it is recommended
    to specify a new folder name in the 'dst' argument. The destination folder does NOT need to exist, because of
    'dirs_exist_ok' argument in shutil.copytree().

    For example, if we want to copy the folder C:/foo to folder D:/bar, then use:
        src = 'C:/foo'
        dst = 'D:/bar/foo' (instead of 'D:/bar')

    Args:
        src: source folder, containing contents to be copied to the destination folder
        dst: destination folder, i.e. folder that will get copies of the folder+files from the source folder

    Returns:

    """
    # Recursively copy an the content of a directory tree rooted at src to a directory named dst. The 'dirs_exist_ok'
    # argument dictates whether to raise an exception in case dst or any missing parent directory already exists.
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src, dst):
    """
    Copy source (src) file to destination (dst) file.

    Note that renaming is possible.

    Args:
        src: source file to be copied to the destination file
        dst: destination file (potentially renamed) from source file.

    Returns:

    """
    shutil.copy(src, dst)


def create_folder_if_not_exists(folder, logger=None):
    """
    Create folder if it does not exist yet.

    It is also possible to create subfolders. For example, if path D:/foo exists and we want to create D:/foo/bar/baz,
    but D:/foo/bar does not exist, then it is still possible to directly create D:/foo/bar/baz by
    create_folder_if_not_exists('D:/foo/bar/baz').

    Args:
        folder:
        logger:

    Returns:

    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if logger is not None:
        logger.my_print('Creating folder: {}'.format(folder))


def get_all_folders(path):
    """
    Get all folders, including sub-folders, in path.

    Args:
        path (str):

    Returns:
        list of folders in path.
    """
    return [x[0] for x in os.walk(path)]


def get_folders_diff_or_intersection(list_1, list_2, mode):
    """
    mode='diff': Get list of folders (as strings) that are present in list_1, but not in list_2.
    mode='intersection': Get list of folders (as strings) that are present in both list_1 and list_2.
    Important: no subfolders are considered!

    Args:
        list_1 (list): folders (as strings) in list_1
        list_2 (list): folders (as strings) in list_2
        mode (str): either 'diff' (difference) or 'intersection'

    Returns:

    """
    if mode == 'diff':
        output = [x for x in list_1 if x not in list_2]
    elif mode == 'intersection':
        output = [x for x in list_1 if x in list_2]
    else:
        raise ValueError('Mode {} is not valid.'.format(mode))

    # Sort list
    output = sort_human(output)

    return output


def list_to_txt_str(l, sep='\n'):
    """
    Convert a list to string. For example, if l = ['a', 'b', 'c', 1] and sep='\n', then the output will be:
    txt = 'a\nb\nc\n1\n'

    Args:
        l: list with elements
        sep: separator

    Returns:

    """
    # Initiate txt
    txt = ''

    for i in l:
        txt += str(i) + sep

    return txt


def move_folder_or_file(src, dst):
    """
    Move source (src) folder/file to destination (dst) folder.

    Args:
        src: source file to be moved to the destination folder
        dst: destination folder

    Returns:

    """
    shutil.move(src, dst)


def round_nths_of_list(in_list, n, decimal):
    """
    Round every n^{th} element of `in_list` to decimcal.

    Args:
        in_list (list):
        n (int):
        decimal (int):

    Returns:

    """
    return [round(x, decimal) if (i + 1) % n == 0 else x for i, x in enumerate(in_list)]


def set_default(obj):
    """
    Set JSON defaults: determine alternative datatype for datatypes that are invalid for JSONs. Otherwise,
    Python will raise an error.

    Args:
        obj:

    Returns:

    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, MultiValue):
        return list(obj)
    if np.issubdtype(obj, np.unsignedinteger):
        return int(obj)
    raise TypeError


def sort_human(l):
    """
    Sort the input list. However normally with l.sort(), e.g., l = ['1', '2', '10', '4'] would be sorted as
    l = ['1', '10', '2', '4']. The sort_human() function makes sure that l will be sorted properly,
    i.e.: l = ['1', '2', '4', '10'].
    
    Source: https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically

    Args:
        l: to-be-sorted list

    Returns:
        l: properly sorted list
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

