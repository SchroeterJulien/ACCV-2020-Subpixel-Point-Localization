import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as utils

dwn_rate = [1/2,1/4,1/6]


version = "torch_full_v6_"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dimPred = 2 + 1
n_points = 1




#####################################
### 0. Our Approach


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 13, 1, padding=6)
        self.dowsampling1 = nn.Conv2d(32, 32, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 1, 1)

        self.dowsampling2 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(64, n_points * dimPred, 3, 1, padding=1)

        torch.nn.init.constant_(self.conv3.bias, -4.0)

    def forward(self, input):

        input = input.permute([0,3,1,2])
        x = (input - 255 / 2) / 255

        x = F.relu(self.conv1(x))
        x = F.relu(self.dowsampling1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.dowsampling2(x))
        x = torch.sigmoid(self.conv3(x))


        # Prediction Offsets
        final_y = x.shape[3]
        original_shape_y = input.shape[3]
        final_x = x.shape[2]
        original_shape_x = input.shape[2]

        x_offset, y_offset = np.meshgrid(
            np.linspace(0, original_shape_y -original_shape_y / final_y, final_y),
            np.linspace(0, original_shape_x - original_shape_x / final_x, final_x))

        x_clone = x.clone() #to allow for in-place operations
        x_clone[:, 0, :, :] *= original_shape_x / final_x # local offset
        x_clone[:, 1, :, :] *= original_shape_y / final_y # local offset
        x_clone[:, 0,:,:] += torch.tensor(y_offset[np.newaxis,:,:], device=device).float() - 0.5
        x_clone[:, 1,:,:] += torch.tensor(x_offset[np.newaxis,:,:], device=device).float() - 0.5

        x_clone = x_clone.permute(0,2,3,1)

        return x_clone.view(x_clone.shape[0], -1, dimPred)

model = Net().to(device)
model.eval()
save_dict = torch.load('models/' + version + 'model.ckpt',
                       map_location=lambda storage, location: storage)
model.load_state_dict(save_dict)


threshold = 0.4
def getHeatmapCorners(file):
    img = cv2.imread(file, 0)
    img = cv2.GaussianBlur(img, (13, 13), 1)

    if len(img.shape)>2:
        input = img[np.newaxis]
    else:
        input= np.tile(img[np.newaxis,:,:,np.newaxis],[1,1,1,3])

    padded_input = np.zeros([input.shape[0],int(4*np.ceil(input.shape[1]/4)),int(4*np.ceil(input.shape[2]/4)), input.shape[3]])
    padded_input[:,:input.shape[1],:input.shape[2],:] = input

    prediction = model(torch.tensor(padded_input, device=device).float())

    pp = prediction.cpu().data.numpy()
    pp = pp[0,pp[0,:,2]>threshold,:]
    pp = pp[:,[1, 0, 2]]

    #quick cleaning: delete potential duplicates
    pp = pp[np.argsort(-pp[:, 2]), :]
    index_list = []
    for pp_index in range(pp.shape[0]):
        min_distance = 100
        for pp_compare in range(pp_index):
            min_distance = min(min_distance,np.linalg.norm(pp[pp_index, :2] - pp[pp_compare, :2]))
        #print(min_distance)
        if min_distance>2:
            index_list.append(pp_index)
    pp = pp[np.array(index_list),:]

    return pp


files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for idx_dow in range(len(dwn_rate)):
    distance_list = []
    dist_x = []
    dist_y = []
    false_negative = 0

    for idx_test in range(len(files_list)):

        # Original
        oCam_board = \
            np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

        original_board = getHeatmapCorners(files_list[idx_test].replace('/GoPro/','/GoPro_bw/'))

        dwn_board = getHeatmapCorners(files_list[idx_test].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        dwn_board[:,:2] = dwn_board[:,:2]/dwn_rate[idx_dow]

        if False:
            plt.figure(figsize=(15, 15))
            for jj in range(1):
                plt.subplot(1, 1, jj + 1)
                img = cv2.imread(files_list[idx_test], 0)
                plt.imshow(np.tile(img[:,:,np.newaxis],[1,1,3]))

                # Plot predictions
                point = dwn_board
                for kk in range(1):
                    for ii in range(point.shape[0]):
                        plt.scatter(point[ii, 0], point[ii, 1], c='r')
                plt.axis('equal')
                plt.gca().axis('off')
                plt.tight_layout()
            plt.savefig('tmp.JPG')
            plt.close('all')

        if dwn_board.shape[0]!=0:

            # Take predictions that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5

            false_negative+=np.sum(np.logical_not(np.min(oCamdistance,axis=1)<5))

            dwn_board = dwn_board[oCamIdx]

            # Take original that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], original_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(original_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - original_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5
            original_board = original_board[oCamIdx]

            # Take distance
            distance = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            distance_x = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            distance_y = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            for kk in range(original_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    distance[kk, jj] = np.linalg.norm(original_board[kk, :2] - dwn_board[jj, :2])
                    distance_x[kk, jj] = original_board[kk, 0] - dwn_board[jj, 0]
                    distance_y[kk, jj] = original_board[kk, 1] - dwn_board[jj, 1]

            mini = np.min(distance,axis=1)
            mini_x = distance_x[np.arange(distance_x.shape[0]),np.argmin(np.abs(distance_x), axis=1)]
            mini_y =  distance_y[np.arange(distance_y.shape[0]),np.argmin(np.abs(distance_y), axis=1)]
            dist_x += mini_x[mini < 6].tolist()
            dist_y += mini_y[mini < 6].tolist()

            mini = mini[mini < 6]  # non-detection
            distance_list+= mini.tolist()

        else:
            false_negative+=oCam_board.shape[0]

    print("===================================================================================")
    print(idx_dow, np.mean(distance_list), np.quantile(np.array(distance_list),0.5), np.quantile(np.array(distance_list),0.9),false_negative)
    print("===================================================================================")




########################################################
########################################################
### 1. Heatmap baselines



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 13, 1, padding=6)
        self.conv2 = nn.Conv2d(32, 64, 1, 1)
        self.conv3 = nn.Conv2d(64, 1, 3, 1,padding=1)
        torch.nn.init.constant_(self.conv3.bias, -4.0)

    def forward(self, x):

        x = x.permute([0,3,1,2])
        x = (x - 255 / 2) / 255

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))

        return torch.squeeze(x)


# Define Graph
version = "torch_benchmark_v6_"
model = Net().to(device)
model.eval()
save_dict = torch.load('models/' + version + 'model.ckpt',
                       map_location=lambda storage, location: storage)
model.load_state_dict(save_dict)




### 1. Heatmap-based

_subpixel = False

def getHeatmapCorners(file):
    img = cv2.imread(file, 0)

    if len(img.shape)>2:
        input = img[np.newaxis]
    else:
        input= np.tile(img[np.newaxis,:,:,np.newaxis],[1,1,1,3])

    prediction = model(torch.tensor(input.astype(np.float32), device=device).float())
    pp = prediction.cpu().data.numpy()

    if False:
        plt.figure()
        plt.imshow(pp)
        plt.savefig('tmp.png')
        plt.close('all')

    if _subpixel:
        pp = np.array(utils.subpixelPrediction(np.copy(pp)))
    else:
        pp = np.array(utils.argmaxPrediction(np.copy(pp))).astype(np.float32)

    return pp


files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for idx_dow in range(len(dwn_rate)):
    distance_list = []
    dist_x = []
    dist_y = []
    false_negative = 0

    for idx_test in range(len(files_list)):

        print(idx_test)

        # Original
        oCam_board = \
            np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

        original_board = getHeatmapCorners(files_list[idx_test])

        dwn_board = getHeatmapCorners(files_list[idx_test].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        dwn_board /= dwn_rate[idx_dow]

        if dwn_board.shape[0]!=0:

            # Take predictions that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5

            false_negative+=np.sum(np.logical_not(np.min(oCamdistance,axis=1)<5))
            dwn_board = dwn_board[oCamIdx]

            # Take original that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], original_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(original_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - original_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5
            original_board = original_board[oCamIdx]

            # Take distance
            distance = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            distance_x = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            distance_y = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            for kk in range(original_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    distance[kk, jj] = np.linalg.norm(original_board[kk, :2] - dwn_board[jj, :2])
                    distance_x[kk, jj] = original_board[kk, 0] - dwn_board[jj, 0]
                    distance_y[kk, jj] = original_board[kk, 1] - dwn_board[jj, 1]

            mini = np.min(distance,axis=1)
            mini_x = distance_x[np.arange(distance_x.shape[0]),np.argmin(np.abs(distance_x), axis=1)]
            mini_y =  distance_y[np.arange(distance_y.shape[0]),np.argmin(np.abs(distance_y), axis=1)]
            dist_x += mini_x[mini < 6].tolist()
            dist_y += mini_y[mini < 6].tolist()

            mini = mini[mini < 6]  # non-detection
            distance_list+= mini.tolist()

            distance_list+= mini.tolist()

            if idx_test%10==0:
                print("-----", np.mean(distance_list))
        else:
            false_negative+=oCam_board.shape[0]

    print("===================================================================================")
    print(idx_dow, np.mean(distance_list), np.quantile(np.array(distance_list),0.5), np.quantile(np.array(distance_list),0.9),false_negative)
    print("===================================================================================")




#####################################
### 2. OpenCV

def OpenCV_corners(file):

    img = cv2.imread(file, 0)
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray=img

    # Find the chessboard corners

    if True:
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine the corner predictions

            corners = corners[:, 0, :]
        else:
            corners = np.zeros([0, 2])

    else:
        # other option
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)  # refine the corner predictions

    return corners

files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for idx_dow in range(len(dwn_rate)):
    distance_list = []
    false_negative = 0

    for idx_test in range(len(files_list)):

        # Original
        oCam_board = \
            np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

        original_board = OpenCV_corners(files_list[idx_test])

        dwn_board = OpenCV_corners(files_list[idx_test].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        dwn_board /= dwn_rate[idx_dow]


        if dwn_board.shape[0]!=0:

            # Take predictions that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5

            false_negative+=np.sum(np.logical_not(np.min(oCamdistance,axis=1)<5))
            dwn_board = dwn_board[oCamIdx]

            if dwn_board.shape[0]!=0:
                # Take original that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], original_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(original_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - original_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 5
                original_board = original_board[oCamIdx]

                # Take distance
                distance = np.zeros([original_board.shape[0], dwn_board.shape[0]])
                for kk in range(original_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        distance[kk, jj] = np.linalg.norm(original_board[kk, :2] - dwn_board[jj, :2])

                mini = np.min(distance,axis=1)
                mini = mini[mini < 6]  # non-detection

                distance_list+= mini.tolist()
            else:
                false_negative += oCam_board.shape[0]
        else:
            false_negative+=oCam_board.shape[0]

    print("----------OpenCV")
    print(idx_dow, np.mean(distance_list), np.quantile(np.array(distance_list),0.5), np.quantile(np.array(distance_list),0.9),false_negative)







#####################################
### 3. Rochade
files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for idx_dow in range(len(dwn_rate)):
    distance_list = []
    false_negative = 0

    for idx_test in range(len(files_list)):

        # Original
        oCam_board = \
            np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

        original_board = \
            np.loadtxt(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_rochade_refined_5.txt'))
        original_board-=1 #matlab

        dwn_board = np.loadtxt(
            files_list[idx_test].replace('/benchmarkData/GoPro/', '/results/GoPro_{0}/'.format(idx_dow)).replace('.JPG','_rochade_refined_5.txt'))
        dwn_board-=1 # matlab

        dwn_board /= dwn_rate[idx_dow]
        if dwn_board.shape[0]!=0:

            # Take predictions that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5

            false_negative+=np.sum(np.logical_not(oCamIdx))
            dwn_board = dwn_board[oCamIdx]

            # Take original that are a board
            oCamdistance = np.zeros([oCam_board.shape[0], original_board.shape[0]])
            for kk in range(oCam_board.shape[0]):
                for jj in range(original_board.shape[0]):
                    oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - original_board[jj, :2])
            oCammini = np.min(oCamdistance,axis=0)
            oCamIdx = oCammini < 5
            original_board = original_board[oCamIdx]

            # Take distance
            distance = np.zeros([original_board.shape[0], dwn_board.shape[0]])
            for kk in range(original_board.shape[0]):
                for jj in range(dwn_board.shape[0]):
                    distance[kk, jj] = np.linalg.norm(original_board[kk, :2] - dwn_board[jj, :2])

            mini = np.min(distance,axis=1)
            mini = mini[mini < 6]  # non-detection

            distance_list+= mini.tolist()
        else:
            false_negative+=oCam_board.shape[0]
    print(idx_dow, np.mean(distance_list), np.quantile(np.array(distance_list),0.5), np.quantile(np.array(distance_list),0.9),false_negative)


#####################################
### 3. OCamCalib

files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for idx_dow in range(len(dwn_rate)):
    distance_list = []
    false_negative = 0
    for idx_test in range(len(files_list)):

        # Original
        original_board = \
            np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

        dwn_board = np.load(
            files_list[idx_test].replace('/benchmarkData/GoPro/', '/results/GoPro_{0}/'.format(idx_dow)).replace('.JPG','_OCamCalib.npy'))

        dwn_board /= dwn_rate[idx_dow]

        distance = np.zeros([original_board.shape[0], dwn_board.shape[0]])
        for kk in range(original_board.shape[0]):
            for jj in range(dwn_board.shape[0]):
                distance[kk, jj] = np.linalg.norm(original_board[kk, :2] - dwn_board[jj, :2])

        mini = np.min(distance,axis=1)
        mini = mini[mini<6] #take out non-detection
        distance_list+= mini.tolist()

        false_negative += (54-len(mini))

    print(idx_dow, np.mean(distance_list), np.quantile(np.array(distance_list),0.5), np.quantile(np.array(distance_list),0.9),false_negative)


