# -*- coding: utf-8 -*- 
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, IO
import os

from torch import full

def get_id_by_file_path(
    file_path: Optional[str] = None,
    ) -> Optional[str]:
    return file_path.split("\\")[-1].split(".")[0]


# get all data in origin folder by format 
def get_list_by_format(
    folder_path: Union[str, Path, IO] = None,
    formats: Optional[tuple] = None,
    ) -> Optional[List[List]]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")


    return_list = {}

    if formats:
        for tmp_format in list(formats):
            tmp_list = []
            dir_list = os.listdir(folder_path)
            for dir in dir_list:
                full_path = os.path.join(folder_path, dir)
                if os.path.isdir(full_path):
                    dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
                if os.path.isfile(full_path) and full_path.endswith(tmp_format):
                    tmp_list.append(full_path)
        return_list[tmp_format] = tmp_list


def get_mask_list(
    folder_path: Union[str, Path, IO] = None,
    ) -> Optional[Dict]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")

    return_list = {}
    current_id = None
    tmp_mask_list = []


    dir_list = os.listdir(folder_path)
    for dir in dir_list:
        full_path = os.path.join(folder_path, dir)
        if os.path.isdir(full_path):
            dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
        if os.path.isfile(full_path) and full_path.endswith("png") and "\\masks" in full_path:
           if current_id is None:
               current_id = full_path.split("_")[1][4:] 
           else:
               print(current_id != full_path.split("_")[1][4:])
               print(full_path.split("_")[1][4:])
               if current_id != full_path.split("_")[1][4:]:
                   tmp_mask_list.sort(key=lambda x: int(x.split('-')[-1].split('.')[0])) 
                   return_list[current_id] = tmp_mask_list
                   current_id = full_path.split("_")[1][4:]
                   tmp_mask_list = []
               else:
                   tmp_mask_list.append(full_path)
    
    return return_list





if __name__ == '__main__':
    path = "E:\\benis\\Documents\\Arbeit\\Arbeit\\Augenklinik\\Projekt Sarah\\Macustar\\V1\\testdaten\\MACUSTAR_313-001-0001_R_2018-3-26"
    id_list = get_list_by_format(path,".vol")
    mask_list = get_mask_list(path)


    #print(get_id_by_file_path(id_list[".vol"][1]))