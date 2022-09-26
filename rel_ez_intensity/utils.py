# -*- coding: utf-8 -*- 
from datetime import date
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, IO
import os
import cv2 
import numpy as np
from torch import full
import matplotlib.pyplot as plt 
from scipy.ndimage import shift
import eyepy as ep
import pandas as pd



# key (id): (Eccentricity, Angularposition)
grid_iamd = {
    1:  (0, 0),

    2:  (1, 0),    3:  (1, 90),   4:  (1, 180),  5:  (1, 270),
    
    6:  (3, 0),    7:  (3, 30),   8:  (3, 60),   9:  (3, 90),
    10: (3, 120), 11:  (3, 150), 12:  (3, 180), 13:  (3, 210),
    14: (3, 240), 15:  (3, 270), 16:  (3, 300), 17:  (3, 330),

    18: (5, 0),   19:  (5, 30),  20:  (5, 60),  21:  (5, 90),
    22: (5, 120), 23:  (5, 150), 24:  (5, 180), 25:  (5, 210),
    26: (5, 240), 27:  (5, 270), 28:  (5, 300), 29:  (5, 330),

    30: (7, 0),   31:  (7, 90),  32:  (7, 180), 33:  (7, 270),
}


def get_microperimetry(
        file_path: Union[str, Path, IO] = None,
        pid: str = None,
        visit: int = None,
        laterality: str = None,
        mode: str = None):

        
    df = pd.read_excel(file_path)
    idx = np.where(df["UNIQUE_ID"] == pid + "-V" + str(visit) + "-" + laterality + "-" + mode)[0]
    if len(idx) == 1:
        data = df.loc[idx[0]]
    else:
        raise ValueError("ID is not in data %s" % file_path)
        
    micro = np.array(data[
            np.logical_and(
                np.arange(0,len(data)) > 25,
                np.logical_and(
                    np.arange(0,len(data)) < 25 + 2 * 33,
                    np.arange(0,len(data)) % 2 == 0)
                    )].array._ndarray)

    micro[micro == "<0"] = -1

    return (-1) * micro.astype(int) 

def get_microperimetry_IR_image_list(
     folder_path: Union[str, Path, IO] = None,
    ) -> Optional[Dict]:

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")

    return_list_m = {}
    return_list_s = {}


    dir_list = os.listdir(folder_path)
    for dir in dir_list:
        full_path = os.path.join(folder_path, dir)
        if os.path.isdir(full_path):
            dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
        if os.path.isfile(full_path) and full_path.endswith(".png"):
            pid = full_path.split("\\")[-1].split("_")[1] + "-" + full_path.split("\\")[-1].split("_")[2] 

            if full_path.split("\\")[-1].split("_")[5][0] == "m":
                return_list_m[pid] = full_path
            else:
                return_list_s[pid] = full_path

    return return_list_m, return_list_s

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

def get_rpd_list(
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
        if os.path.isfile(full_path) and full_path.endswith(".zip"):
            return_list[full_path.split("\\")[-1].split("_")[1][4:]] = full_path
    return return_list

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

def get2DAffineTransformationMartix_by_SIFT(img1, img2):
    # Create our SIFT detector and detect keypoints and descriptors
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
    search_params = dict(checks=1500)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            matchesMask[i]=[1,0]
            
    k1_l = []
    k2_l = []
    for i, (matchM, match) in enumerate(zip(matchesMask,matches)):
        if any(matchM):
            k1_l.append(kp1[matches[i][0].queryIdx].pt)
            k2_l.append(kp2[matches[i][0].trainIdx].pt)


    k1_l = np.float32(k1_l)
    k2_l = np.float32(k2_l)

    b = k2_l.T.flatten()[:,None]

    M = np.zeros((2*len(k2_l),6))
    M[0:len(k1_l),:3] = np.append(k1_l,np.ones((len(k1_l),1)),axis=1)
    M[len(k1_l):,3:6] = np.append(k1_l,np.ones((len(k1_l),1)),axis=1)


    # create pseudo inverse matrix 
    M_piv = np.linalg.pinv(M)
    a = M_piv@b
    A_cal = a.reshape(2,3)

    return A_cal

def get2DRigidTransformationMatrix(p, q):

    """
    Args:
        p (np.array): 2x2 Matrix with column-wise vector [x_n, y_n].T (Source)
        q (np.array): 2x2 Matrix with column-wise vector [x_n, y_n].T (Target)
    """

    # create pseudo inverse matrix 
    M = np.zeros((4,4))
    M[0:2,:-1] = np.append(p, np.ones((1,2)),axis=0).T
    M[2:,0] = p[1,:]
    M[2:,1] = -p[0,:]
    M[2:,-1] = np.ones((2,))
    
    b = q.flatten()[:,None]

    M_piv = np.linalg.pinv(M)
    r = M_piv@b

    # calculate rotation angle alpha
    alpha = np.arcsin(-r[1,0])[...]


    R = np.array(
        [[np.cos(alpha), -np.sin(alpha), r[2,0]],
        [np.sin(alpha), np.cos(alpha), r[3,0]]])

    return R 


if __name__ == '__main__':
    path = "E:\\benis\\Documents\\Arbeit\\Arbeit\\Augenklinik\\GitLab\\test_data\\macustar"
    id_list = get_list_by_format(path,".vol")
    mask_list = get_mask_list(path)


    #print(get_id_by_file_path(id_list[".vol"][1]))