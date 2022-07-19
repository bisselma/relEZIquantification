# -*- coding: utf-8 -*- 
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, IO
import os
import cv2 
import numpy as np
from torch import full
import matplotlib.pyplot as plt 
from scipy.ndimage import shift
from scipy.ndimage.morphology import binary_dilation
import eyepy as ep

import rel_ez_intensity.base

def get_id_by_file_path(
    file_path: Optional[str] = None,
    ) -> Optional[str]:
    return file_path.split("\\")[-1].split(".")[0]


def get_vol_list(
    folder_path: Union[str, Path, IO] = None,
    ) -> Optional[Dict]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")

    return_list = {}


    dir_list = os.listdir(folder_path)
    for dir in dir_list:
        full_path = os.path.join(folder_path, dir)
        if os.path.isdir(full_path):
            dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
        if os.path.isfile(full_path) and full_path.endswith(".vol"):
            pid = full_path.split("\\")[-2].split("_")[1][4:]
            return_list[pid] = full_path
    return return_list
            


def get_rpedc_list(
    folder_path: Union[str, Path, IO] = None,
    ) -> Optional[Dict]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")

    return_list = {}


    dir_list = os.listdir(folder_path)
    for dir in dir_list:
        full_path = os.path.join(folder_path, dir)
        if os.path.isdir(full_path):
            dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
        if os.path.isfile(full_path) and full_path.endswith(".tif") and "RPEDC_thickness-map" in full_path:
            return_list[full_path.split("\\")[-1].split("_")[1][4:]] = full_path
    return return_list

def get_rpedc_map(
    file_path: Union[str, Path, IO] = None,
    scan_size: Optional[tuple] = None,
    mean_rpedc: Optional[rel_ez_intensity.base.OCTMap] = None,
    laterality: Optional[str] = None,
    translation: Optional[tuple] = None
    ) -> np.ndarray:

    maps = cv2.imread(file_path, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    
    
    if laterality == "OS":
        maps = np.flip(maps, 1)
        
    maps_shifted = shift(maps, translation)
    # substract mean thickness of rpedc plus 3 times std (Duke/AREDS Definition) 
    sub = maps_shifted - (mean_rpedc.octmap["mean"] + (4. * mean_rpedc.octmap["std"])) 


    sub = np.logical_or(sub > 0., maps_shifted <= 0.01)
    sub_resized = cv2.resize(sub.astype(np.uint8), scan_size[::-1], cv2.INTER_LINEAR) # cv2.resize get fx argument befor fy, so  the tuple "scan_size" must be inverted

    
    # structure element should have a radius of 100 um (Macustar-format (
    # ~30 um per bscan => ~ 3 px in y-direction and 2 * 3 px to get size of rectangle in y-dir. 
    # ~ 10 um per ascan => ~ 10 px in x-direction and 2 * 10 px to get size of rectangle in x-dir. 
    struct = np.ones((4, 10), dtype=bool)
    sub_dilation = binary_dilation(sub_resized, structure = struct)   
    

    return sub_dilation.astype(bool) 

# get all data in origin folder by format 
def get_list_by_format(
    folder_path: Union[str, Path, IO] = None,
    formats: Optional[tuple] = None,
    ) -> Optional[Dict]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")


    return_list = {}

    if formats:
        for tmp_format in list(formats):
            
            if tmp_format == ".vol":
                dir_list = os.listdir(folder_path)
                tmp_dict = {}          
                for dir in dir_list:               
                    full_path = os.path.join(folder_path, dir)
                    if os.path.isdir(full_path):
                        dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
                    if os.path.isfile(full_path) and full_path.endswith(tmp_format):
                        tmp_dict[full_path.split("\\")[-2].split("_")[1][4:]] = full_path                      
                return_list[tmp_format] = tmp_dict
            else:
                dir_list = os.listdir(folder_path)
                tmp_list = []
                for dir in dir_list:               
                    full_path = os.path.join(folder_path, dir)
                    if os.path.isdir(full_path):
                        dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
                    if os.path.isfile(full_path) and full_path.endswith(tmp_format):
                        tmp_list.append(full_path)                     
                return_list[tmp_format] = tmp_list               
            
    return return_list


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
               current_id = full_path.split("_")[-4][4:] 
               tmp_mask_list.append(full_path)
           else:
               if current_id != full_path.split("_")[-4][4:]:
                   tmp_mask_list.sort(key=lambda x: int(x.split('-')[-1].split('.')[0])) 
                   return_list[current_id] = tmp_mask_list[::-1]
                   current_id = full_path.split("_")[-4][4:]
                   tmp_mask_list = [full_path]
               else:
                   tmp_mask_list.append(full_path)
    
    tmp_mask_list.sort(key=lambda x: int(x.split('-')[-1].split('.')[0])) 
    return_list[current_id] = tmp_mask_list[::-1]
    return return_list


def get_seg_by_mask(mask_path, n):
    """
    Args:

    mask_path (str): file path to read in mask image
    n (int): layer number

    """
    
    mask = cv2.imread(mask_path, 0)

    layer = np.zeros((1, mask.shape[1])).astype(np.int16)

    for i, col in enumerate(mask.T):
        idxs = np.where(col == n)[-1]
        if any(idxs):
            layer[0,i] = idxs[0]
    
    return layer[0,:]



if __name__ == '__main__':
    path = "E:\\benis\\Documents\\Arbeit\\Arbeit\\Augenklinik\\GitLab\\test_data\\macustar"
    id_list = get_list_by_format(path,".vol")
    mask_list = get_mask_list(path)


    #print(get_id_by_file_path(id_list[".vol"][1]))