import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from rel_ez_intensity.seg_utils import Layer, get_flatten_seg
import rel_ez_intensity.utils as sh
from rel_ez_intensity.getAdjacencyMatrix import get_adjacency_matrix, plot_layers
from rel_ez_intensity.getAdjacencyMatrix import (get_adjacency_matrix, sparse_matrix, find_shortest_path, get_path, sub2ind,
    ind2sub)




# This Object contains all the neccessary parameters of the segmentation 
class Params(object):
    def __init__(self):
        self.is_resize = 1.0
        self.filter_0_params = np.array([5, 5, 1])
        self.filter_params = np.array([20, 20, 2])
        self.is_os_0 = 20
        self.is_os_1 = -8
        self.rpe_0 = 13
        self.rpe_1 = 1

        # adjacency matrices parameter
        self.adjMatrixW = []
        self.adjMatrixMW = []
        self.adjMAsub = []
        self.adjMBsub = []
        self.adjMW = []
        self.adjMmW = []

        # flatten adjacency matrices parameter
        self.adjMatrixW_f = []
        self.adjMatrixMW_f = []
        self.adjMAsub_f = []
        self.adjMBsub_f = []
        self.adjMW_f = []
        self.adjMmW_f = []

 


# This function is used in get_retinal_layers
def get_retinal_layers_core(layer_name, img, params, paths_list, shift_array):

    

    adj_ma = params.adjMAsub_f
    adj_mb = params.adjMBsub_f
    adj_MW = params.adjMW_f
    adj_MmW = params.adjMmW_f


    # init region of interest
    sz_img = img.shape
    sz_img_org = np.array([sz_img[0],sz_img[1]-2])
    roi_img = np.zeros(sz_img)


    # runs through the new image with the added columns 
    for k in range(0, sz_img_org[1]):

        if layer_name == 'rpe':

 
            start_ind = img.shape[0] - params.rpe_0
            end_ind = img.shape[0] -1 


        else:
            ind_pathX = np.where(paths_list['rpe'].layerX == k)
            ind_pathX = ind_pathX[0].reshape(1, ind_pathX[0].size)


            start_ind = params.is_os_0  
            end_ind = paths_list['rpe'].layerY[ind_pathX[0, 0]] + params.is_os_1




        # error checking

        if start_ind >= end_ind:
            start_ind = end_ind - 1

        if start_ind < 0:
            start_ind = 0

        if end_ind > sz_img[0] - 1:
            end_ind = sz_img[0] - 1
        
        



        # set column_wise region of interest to 1
        roi_img[start_ind: end_ind, k+1] = 1

    # set first and last column to 1
    roi_img[:, 0] = 1
    roi_img[:, -1] = 1


    # include only the region of interest in the adjacency matrix
    ind1, ind2 = np.nonzero(roi_img[:] == 1)
    include_a = np.isin(adj_ma, sub2ind(roi_img.shape, ind1, ind2))
    include_b = np.isin(adj_mb, sub2ind(roi_img.shape, ind1, ind2))

    keep_ind = np.logical_and(include_a, include_b)

    

    ## Dark to Bright or bright to dark adjacency

    # dark to bright
    if layer_name in ['isos']:

        adjMatrixW = sparse_matrix(adj_MW[keep_ind], adj_ma[keep_ind], adj_mb[keep_ind], img)
        dist_matrix, predecessors = find_shortest_path(adjMatrixW)
        path = get_path(predecessors, len(dist_matrix) - 1)

    # bright to dark 
    else:

        adjMatrixMW = sparse_matrix(adj_MmW[keep_ind], adj_ma[keep_ind], adj_mb[keep_ind], img)
        dist_matrix, predecessors = find_shortest_path(adjMatrixMW)
        path = get_path(predecessors, len(dist_matrix) - 1)

    # get pathX and pathY
    pathY, pathX = ind2sub(sz_img, path)
    pathY = pathY[np.gradient(pathX) != 0]
    pathX = pathX[np.gradient(pathX) != 0]
    path = sub2ind(sz_img_org,pathY,pathX)

# =============================================================================
#     # if flatten image is used unshift the coordinates
#     width = sz_img_org[1]
#     if layer_name in ['isos','rpe']:
# 
#         # crop size to origin image length
#         pathY = pathY[1:-1]
#         pathX = pathX[2:]
#         y_list = np.flip(sh.path_to_y_array(list(zip(pathY,pathX)),width))
#         pathY = sh.shiftColumn(np.flip(shift_array), y_list)
#         pathX = np.flip(range(len(pathY)))
#         path = sub2ind(sz_img_org, pathY, pathX)
# =============================================================================

    if layer_name == "rpe":
        paths_list["rpe"] = Layer(layer_name, path, pathY, pathX)
    else:
        paths_list["isos"] = Layer(layer_name, path, pathY, pathX)
    
    return paths_list


# This function organize the workflow of the retinal segmentation
# It defines the boundaries between 7 retinal layers:

# IS/OS 
# RPE

def get_retinal_layers(img):
    # Parameter object:
    params = Params()

    # get Image size
    sz_img = img.shape

    # Pre-Processing ######################################
    # resize Image
    # img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)


    # smooth image 
    # 2 approaches:

    ## Gaussian Blur
    #img = cv2.GaussianBlur(img,(1,15),1,0)



    ## Median
    img = cv2.medianBlur(img, 3)
    
    

    #######################################################

    params.adjMatrixW_f, params.adjMatrixMW_f, params.adjMAsub_f, params.adjMBsub_f, params.adjMW_f, params.adjMmW_f, img_new = get_adjacency_matrix(
        img)

    # Main part ###########################################
    # Iterate through a sorted list of layer names based on knowledge about the image struture  ###########################################

    # Pre-set order 
    retinal_layer_segmentation_order = ['rpe', 'isos']

    # Iterate through the list and save the found layer boundaries
    retinal_layers = {}
    for layer in retinal_layer_segmentation_order:
        retinal_layers = get_retinal_layers_core(layer, img_new, params, retinal_layers, img)

    ########################################################

    # delete redundant columns 
    for layer in retinal_layers:
        if len(retinal_layers[layer].layerX) > sz_img[1] + 2:
            retinal_layers[layer] = get_flatten_seg(retinal_layers[layer])


    return retinal_layers
    

def save_layers_to_file(layers, filename):
    layers_as_json = "{\"img_name\" : \"" + img_name + "\", \"layers\": ["
    for i, layer in enumerate(layers):
        layers_as_json += layer.JSON()
        if i < len(layers) - 1:
            layers_as_json += (",")
    layers_as_json += ("]}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json.loads(layers_as_json), f, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    img_name = "" # example

    dir = ""


    img = cv2.imread(dir + img_name, cv2.IMREAD_ANYDEPTH)
    imglayers = get_retinal_layers(img)
    #save_layers_to_file(imglayers, str(img_name.split(".")[0]) + '.json')


    plot_layers(img, imglayers)
