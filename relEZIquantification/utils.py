# -*- coding: utf-8 -*- 
from datetime import date
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, IO
import os
import cv2 
import numpy as np
import torch
from torch import full
from torchvision import transforms
import matplotlib.pyplot as plt 
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift
from scipy.interpolate import griddata
import eyepy as ep
from PIL import Image


from relEZIquantification import superretina  
from relEZIquantification.superretina.model.super_retina import SuperRetina
from relEZIquantification.superretina.common.common_util import pre_processing, simple_nms, remove_borders, sample_keypoint_desc

# global config data for superretina
model_image_width = 768
model_image_height = 768
use_matching_trick = True

nms_size = 10
nms_thresh = 0.01
knn_thresh = 0.9

# modul load
device = "cuda:0"
device = torch.device(device if torch.cuda.is_available() else "cpu")
model_save_path = Path(os.path.dirname(superretina.__file__) + "\save\SuperRetina.pth")
checkpoint = torch.load(model_save_path, map_location=device)
model = SuperRetina()
model.load_state_dict(checkpoint['net'])
model.to(device)


# Input a batch with two images to SuperRetina 
def model_run(query_tensor, refer_tensor):
    inputs = torch.cat((query_tensor.unsqueeze(0), refer_tensor.unsqueeze(0)))
    inputs = inputs.to(device)

    with torch.no_grad():
        detector_pred, descriptor_pred = model(inputs)

    scores = simple_nms(detector_pred, nms_size)

    b, _, h, w = detector_pred.shape
    scores = scores.reshape(-1, h, w)

    keypoints = [
        torch.nonzero(s > nms_thresh)
        for s in scores]

    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, 4, h, w)
        for k, s in zip(keypoints, scores)]))

    keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

    descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu().data
                   for k, d in zip(keypoints, descriptor_pred)]
    keypoints = [k.cpu() for k in keypoints]
    return keypoints, descriptors

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
        df,
        pid: str = None,
        visit: int = None,
        laterality: str = None,
        mode: str = None):

    idx = np.where(df["UNIQUE_ID"] == pid + "-V" + str(visit) + "-" + laterality + "-" + mode)[0]
    if len(idx) == 1:
        data = df.loc[idx[0]]
    else:
        return None 

    micro = np.array(data[
            np.logical_and(
                np.arange(0,len(data)) > 25,
                np.logical_and(
                    np.arange(0,len(data)) < 25 + 2 * 33,
                    np.arange(0,len(data)) % 2 == 0)
                    )].array._ndarray)

    micro[micro == "<0"] = -1

    return micro 

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

def get_microperimetry_maps(ir_path, lat, radius, slo_img, scan_size, stackwidth, stimuli, x, y):

            # create binary image with iamd grid 
            if stimuli is None:
                return np.full((scan_size[0], scan_size[1] // stackwidth), np.nan),  np.full((scan_size[0], scan_size[1] // stackwidth), np.nan)      
            else:
                mask_iamd = np.full((scan_size[0], scan_size[1] // stackwidth), np.nan)
                stimuli_map = np.full_like(mask_iamd, np.nan)

            # get microperimetry IR image m and s
            img1_raw = cv2.imread(ir_path,0)
            (h_micro, w_micro) = img1_raw.shape[:2]

            # rotated IR image 
            (cX, cY) = (w_micro // 2, h_micro // 2)
            M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
            img1_raw = cv2.warpAffine(img1_raw, M, (w_micro, h_micro))

            # flip microperimetry-IR image
            if lat == "OS":
                img1_raw = np.flip(img1_raw, 1)

            # crop image to 30° x 30° around centered fovea 3.25° offset on each side
            offset = int((h_micro/36) * 3)
            img1 = img1_raw[offset:-offset,offset:-offset]
            img1 = cv2.resize(img1,slo_img.shape)



            # calculate affine transformation matrix A
            H = get2DProjectiveTransformationMartix_by_SuperRetina(img1, slo_img)
        

            # transform grid
            grid_coords = np.zeros((3,len(x)))
            grid_coords[0,:] = x
            grid_coords[1,:] = y
            grid_coords[2,:] = np.ones((1,len(x)))

            grid_coords_transf = H @ grid_coords


            x_new = (grid_coords_transf[0,:] * (30/ 768)) 
            y_new = ((grid_coords_transf[1,:] - 64) * (25 / 640))

            
            yy,xx = np.mgrid[:241,:(768 // stackwidth)]

            xx = xx * (30/(768 // stackwidth))
            yy = yy * (25/241)

            for idx in range(33):            
                mask_iamd[((yy - y_new[idx]) ** 2) + ((xx - x_new[idx])**2) <= radius ** 2] = idx + 1
                stimuli_map[((yy - y_new[idx]) ** 2) + ((xx - x_new[idx])**2) <= radius ** 2] = stimuli[idx]


            return mask_iamd, stimuli_map


def get_id_by_file_path(
    file_path: Optional[str] = None,
    ) -> Optional[str]:
    return file_path.split("\\")[-1].split(".")[0]

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

def get_vol_list(
    folder_path: Union[str, Path, IO] = None,
    project: str = None
    ):

    if not os.path.exists(folder_path):
        raise NotADirectoryError("directory: " +  folder_path + " not exist")

    path_list = {}
    vid_list = {}


    dir_list = os.listdir(folder_path)
    for dir in dir_list:
        full_path = os.path.join(folder_path, dir)
        if os.path.isdir(full_path):
            dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
        if os.path.isfile(full_path) and full_path.endswith(".vol"):
            if project == "macustar":
                pid = full_path.split("\\")[-2].split("_")[1][4:]
                path_list[pid] = full_path
            elif project == "mactel":
                pid = full_path.split("\\")[-1].split("_")[-1].split(".")[0]   
                path_list[pid] = full_path  
                vid_list[pid] =  full_path.split("\\")[-4]         
    return path_list, vid_list
            
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

def get_roi_masks(bscan, ref_layer, scan_size, seg_mask):
    
    # create raw_roi and seg_mask_roi
    raw_roi = np.full((53,scan_size[1]), np.nan)
    seg_mask_roi = np.full((53,scan_size[1]), np.nan)
    for col_idx in range(scan_size[1]):
        idxs_ref = np.where(seg_mask[:, col_idx] == ref_layer)[0]
        if len(idxs_ref) > 0:
            raw_roi[:, col_idx] = bscan[idxs_ref[0] -48: idxs_ref[0] +5, col_idx]
            seg_mask_roi[:, col_idx] = seg_mask[idxs_ref[0] -48: idxs_ref[0] +5, col_idx]
    
    return raw_roi, seg_mask_roi

def get2DProjectiveTransformationMartix_by_SuperRetina(query_image, refer_image):


    image_transformer = transforms.Compose([
        transforms.Resize((model_image_height, model_image_width)),
        transforms.ToTensor()])

    knn_matcher = cv2.BFMatcher(cv2.NORM_L2)

    # image shape 
    image_height, image_width = refer_image.shape

    query_image = pre_processing(query_image)
    refer_image = pre_processing(refer_image)
    query_image = (query_image * 255).astype(np.uint8)
    refer_image = (refer_image * 255).astype(np.uint8)

    # scaled image size
    query_tensor = image_transformer(Image.fromarray(query_image))
    refer_tensor = image_transformer(Image.fromarray(refer_image))

    keypoints, descriptors = model_run(query_tensor, refer_tensor)

    query_keypoints, refer_keypoints = keypoints[0], keypoints[1]
    query_desc, refer_desc = descriptors[0].permute(1, 0).numpy(), descriptors[1].permute(1, 0).numpy()

    # mapping keypoints to scaled keypoints
    cv_kpts_query = [cv2.KeyPoint(int(i[0] / model_image_width * image_width),
                              int(i[1] / model_image_height * image_height), 30)  # 30 is keypoints size, which can be ignored
                    for i in query_keypoints]
    cv_kpts_refer = [cv2.KeyPoint(int(i[0] / model_image_width * image_width),
                              int(i[1] / model_image_height * image_height), 30)
                    for i in refer_keypoints]  

    # keypoint matching
    goodMatch = []
    status = []
    try:
        matches = knn_matcher.knnMatch(query_desc, refer_desc, k=2)
        for m, n in matches:
            if m.distance < knn_thresh * n.distance:
                goodMatch.append(m)
                status.append(True)
            else:
                status.append(False)
    except Exception:
        pass

    # find homographic matrix H_m 
    H_m = None
    good = goodMatch.copy()
    if len(goodMatch) >= 4:
        src_pts = [cv_kpts_query[m.queryIdx].pt for m in good]
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in good]
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

        H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        good = np.array(good)[mask.ravel() == 1]
        status = np.array(status)
        temp = status[status==True]
        temp[mask.ravel() == 0] = False
        status[status==True] = temp

    return H_m

def get2DAffineTransformationMartix_by_SIFT(img1, img2):
    # Create our SIFT detector and detect keypoints and descriptors
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
    search_params = dict(checks=2000)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # counter 
    count = 0
    dist_factor = 0.5
    # ratio test as per Lowe's paper
    while True:
        for i,(m,n) in enumerate(matches):
            if m.distance < dist_factor * n.distance:
                matchesMask[i]=[1,0]
                count+=1
        
        if count >= 3:
            break
        else:
            count = 0
            dist_factor+=0.01
            matchesMask = [[0,0] for i in range(len(matches))]

    
            
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

def sample_circle(x,y, radius, field, no_dc_ratio):
    yy,xx = np.mgrid[:field.shape[0], :field.shape[1]]

    mask = ((yy - y) ** 2) + ((xx - x)**2) < radius ** 2


    if len(np.where(np.isnan(field[mask]))[0]) / len(np.where(mask)[0]) <= no_dc_ratio:
        mean_i = np.nanmean(field[mask])
        if mean_i > 0 and mean_i != np.nan:
            return np.log(mean_i)
        else:
            return np.nan
    else:
        return np.nan

def exc_circle(x,y, radius, field):
    yy,xx = np.mgrid[:field.shape[0], :field.shape[1]]

    mask = ((yy - y) ** 2) + ((xx - x)**2) < radius ** 2

    return len(np.where(field[mask])[0]) / len(np.where(mask)[0])

def interpolate_grid(x,y,z,w,h, point_radius):

    yy,xx = np.mgrid[:h, :w]

    zi = griddata((x,y),z,(xx,yy),method='linear')


    return zi


def show_grid_over_relEZIMap(
    img,
    rel_ez_i_map,
    x,
    y,
    center,
    stimuli,
    H_matrix,
    eccentricity,
    ppd, # pixel per degree
    savefig,
    pid,
    showfig,
    path
    ):
    # create rel_ezi heatmap illustration image
    rel_ez_i_map_ill = np.zeros_like(img)
    rel_ez_i_map_ill[64:-64, :] = rel_ez_i_map
    
    # calculate transformed intensity map
    w, h = rel_ez_i_map_ill.shape  
    rel_ez_i_map_ill = cv2.warpAffine(rel_ez_i_map_ill, H_matrix, (w, h))    


    rads = np.arange(0,360,20) *  np.pi / 180
    ecc  = np.ones((1, len(rads))) * eccentricity * ppd
    x_ill = ((np.sin(rads) * ecc) + center[0])
    y_ill = ((np.cos(rads) * ecc) + center[1])

    x_ill = np.append(x_ill, x[None,:])        
    y_ill = np.append(y_ill, y[None,:])



    # radius according to Goldmann III 0.43° diameter
    radius = (img.shape[0]/30) * (0.43/2)

    rel_EZI = np.array([sample_circle(x_i,y_i, radius, rel_ez_i_map_ill, 0.8) for x_i, y_i in zip(x_ill,y_ill)])

    rel_EZI[np.isnan(rel_EZI)] = np.nanmin(rel_EZI) # set nan areas to min 


    plt.figure(figsize = (10,10))
    ax = plt.gca()


    plt.axis('off')

    int_grid = interpolate_grid(
                    x= x_ill,
                    y= y_ill,
                    z= rel_EZI,
                    w = img.shape[1],
                    h = img.shape[0],
                    point_radius=0
                    )

    plt.imshow(img, cmap="gray")
    plt.imshow(int_grid, cmap = "RdYlGn", alpha=0.5, vmax= 4)
    print(x,y,stimuli)
    im = ax.scatter(x,y, c=stimuli, cmap = "RdYlGn_r",vmin=-30, vmax=0.5 * np.nanmax(stimuli))
    for i, val in enumerate(stimuli):
        plt.annotate(str(val), (x[i], y[i]))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  
    cbar = plt.colorbar(im, cax=cax)
    ytl_obj = plt.getp(cax, 'yticklabels')                         
    plt.setp(ytl_obj, color="black")                    
    cbar.set_label("dB", color="black", weight='bold')  

    if not showfig:
        plt.ioff()
    if savefig:
        plt.savefig(path + "\\" + pid + ".png")

if __name__ == '__main__':
    pass