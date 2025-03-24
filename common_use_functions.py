"""
Module with the defined common use functions (normally file system operation functions and some pratical use functions)
"""
import csv
import json
import shutil
import requests
from common_imports import np, pd, os, Image

"""
Functions of file systems operations
"""
def write_csv(data, path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerows(data)
        
        f.close()

def content_existence(path):
    return os.path.exists(path)

def contents_of_folder(folder_path):
    return os.listdir(folder_path)

def create_directory(dir, display=True):
    """
    display: Boolean indicating the folder creation information ("success" or "already exists")
    """
    try:
        # Create target Directory
        os.mkdir(dir)
        if display:
            print("Directory " , dir ,  " Created ") 
    except FileExistsError:
        if display:
            print("Directory " , dir ,  " already exists")

def erase_files_from_folder(folder_path):
    shutil.rmtree(folder_path)

def erase_one_file(file_path):
    # If file exists, delete it.
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        # If it fails, inform the user.
        print("Error: %s file not found" % file_path)

def read_content_path(path):
    contents = {}
    for file in os.listdir(path):
        contents[file] = os.path.join(path, file)
    return contents

def path_join(*path_parts):
    return os.path.join(*path_parts)

def store_list_as_json(path, list_to_save):
    with open(path, 'w') as fp:
        json.dump(list_to_save, fp)
        fp.close()

def store_dict_as_json(path, dict_to_save):
    with open(path, 'w') as fp:
        json.dump(dict_to_save, fp)
        fp.close()

def load_json(fp):
    return json.load(fp)

def load_json_by_path(path):
    loaded_json = None
    with open(path) as fp:
        loaded_json = json.load(fp)
        fp.close()
    return loaded_json

def read_csv_to_pd_df(csv_path, first_line_as_head=True, display=False):
    df = None
    if first_line_as_head:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)

    if display:
        print(df.to_string())
    return df

def save_list_to_csv(csv_path, list, index=None, headers=None, save_index=False, save_headers=True, sep=','):
    if index is not None:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers, index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list), index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
    else:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list)).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')

def save_df_to_csv(csv_path, df, save_index=False, save_headers=True, sep=','):
    df.to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+')

def save_content_to_file(data, file_path, with_newline=True):
    # Data should be a list of string
    with open(file_path, 'w+', encoding='UTF-8') as f:  
        if with_newline:
            f.write('\n'.join(data) + '\n')
        else:
            f.writelines(data)

def download_image_from_url(img_url, save_path=None):
    """
    The save_path should contains the filename of the image
    """
    img = Image.open(requests.get(img_url, stream = True).raw)
    if save_path is not None:
        img.save(save_path)
    return img