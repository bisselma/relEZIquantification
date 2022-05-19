import cv2
import json
import numpy as np
import utils as sh
from getAdjacencyMatrix import get_adjacency_matrix, plot_layers
from getAdjacencyMatrix import get_adjacency_matrix, sparse_matrix, find_shortest_path, get_path, sub2ind, \
    ind2sub
import matplotlib.pyplot as plt


# path class
class Path(object):

    def __init__(self, name, path, pathY, pathX):
        self.name = name
        self.path = path
        self.pathY = pathY
        self.pathX = pathX
        self.pathYmean = np.mean(self.pathY)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path
        self.pathXmean = np.mean(self.path)

    def getPathY(self):
        return self.pathY

    def setPathY(self, pathY):
        self.pathY = pathY

    def getPathX(self):
        return self.pathX

    def setPathX(self, pathX):
        self.pathX = pathX

    def getPathYmean(self):
        return self.pathYmean

    def JSON(self):
        return "{\"name\": \"" + self.name + "\"," + "\"path_x\": " + json.dumps((self.pathX.tolist())) + "," + "\"path_y\": " + json.dumps((self.pathY.tolist())) + "}"

# This Object contains all the neccessary parameters of the segmentation 
class Params(object):
    def __init__(self):
        self.is_resize = 1.0
        self.filter_0_params = np.array([5, 5, 1])
        self.filter_params = np.array([20, 20, 2])
        self.is_os_0 = 20
        self.is_os_1 = -12
        self.rpe_0 = 10
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

 

def getHyperReflectiveLayers(inputImg, param):

    # initiate parameters
    if param.shrink_scale is None:
        shrinkScale = 0.2
    else:
        shrinkScale = param.shrink_scale

    if param.offsets is None:
        offsets = np.arange(-20, 21)
    else:
        offsets = param.offsets

    # resize the image.
    szImg = inputImg.size
    resizedImg = cv2.resize(inputImg, dsize=None, fx=shrinkScale, fy=shrinkScale,
                            interpolation=cv2.INTER_LINEAR)

    # create adjacency matrices
    adjMatrixW, adjMatrixMW, adjMAsub, adjMBsub, adjMW, adjMmW, newImg = get_adjacency_matrix(
        resizedImg)

    # create roi for getting shortestest path based on vertical gradient image.
    # get  vertical gradient image
    gy = cv2.Sobel(newImg, cv2.CV_64F, 0, 1, ksize=5)
    # normalize gradient
    gy = (gy - np.amin(gy))/(np.amax(gy)-np.amin(gy))

    # create binary mask
    szImgNew = newImg.shape
    roiImg = np.zeros(szImgNew)
    # set value 1 where pixel value is greater than the mean value of the gradient
    roiImg[gy > np.mean(gy[:])] = 1

    # find 2 layers
    paths = np.empty(2, dtype=object)
    count = 0

    while count < len(paths):

        # add columns of value 1 at both ends of the image
        roiImg[:, 0] = 1
        roiImg[:, -1] = 1

        # include only region of interst in the adjacency matrix
        ind1, ind2 = np.nonzero(roiImg[:] == 1)   # find all pixels equal 1
        indices = sub2ind(roiImg.shape, ind1, ind2)
        # Test whether each element of first array is also present in a second array
        includeA = np.isin(adjMAsub, indices)
        includeB = np.isin(adjMBsub, indices)
        keepInd = np.logical_and(includeA, includeB)

        # compile adjacency matrix
        adjMatrix = sparse_matrix(
            adjMW[keepInd], adjMAsub[keepInd], adjMBsub[keepInd], newImg)

        # apply Dijkstra algorithm
        dist_matrix, predecessors = find_shortest_path(adjMatrix)

        # construct path from the predecessor nodes retrieved from Dijkstra algorithm
        path = get_path(predecessors, len(dist_matrix)-1)

        # get rid of first few points and last few points
        pathY, pathX = ind2sub(newImg.shape, path)
        pathY = pathY[np.gradient(pathX) != 0]
        pathX = pathX[np.gradient(pathX) != 0]
        path = sub2ind(inputImg.shape,pathY,pathX)

        # block the obtained path and abit around it
        pathYArr = np.tile(pathY, (len(offsets), len(offsets)))
        pathXArr = np.tile(pathX, (len(offsets), len(offsets)))


        for i in range(offsets.size):
            pathYArr[i,:] = pathYArr[i,:] + offsets[i]
            #pathXArr[i, :] = pathXArr[i, :] + offsets[i]

        #pathXArr = pathXArr[np.logical_and(pathYArr >= 0, pathYArr < szImgNew[1])]
        pathYArr = pathYArr[np.logical_and(
            pathYArr >= 0, pathYArr < szImgNew[0])]
        pathXArr = pathXArr[np.logical_and(
            pathXArr >= 0, pathXArr < szImgNew[1])]

        pathArr = sub2ind(szImgNew, pathYArr, pathXArr)

        roiImg[pathYArr, pathXArr] = 0

        # plot the masked path
        #plot_layers(gy, [pathArr])

        paths[count] = Path("", path, pathY[1:-1], pathX[2:])

        count += 1

    # define the name of the detected layer boundary based on the mean y value. ILM lies always above IS-OS

    if paths[0].getPathYmean() > paths[1].getPathYmean():
        paths[0].name = 'isos'
        paths[1].name = 'ilm'
    else:
        paths[0].name = 'ilm'
        paths[1].name = 'isos'

    return paths


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
            ind_pathX = np.where(paths_list['rpe'].pathX == k)
            ind_pathX = ind_pathX[0].reshape(1, ind_pathX[0].size)


            start_ind = params.is_os_0  
            end_ind = paths_list['rpe'].pathY[ind_pathX[0, 0]] + params.is_os_1




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
        paths_list["rpe"] = Path(layer_name, path, pathY, pathX)
    else:
        paths_list["isos"] = Path(layer_name, path, pathY, pathX)
    
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
