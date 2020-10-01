import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as utils



import sys
term_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200,sys.float_info.epsilon)

alternative=True

def OpenCV_corners(file):
    img = cv2.imread(file, 0)
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Find the chessboard corners
    if True:
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

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

    return corners, gray

#####################################
### 2. OpenCV
import sys
if len(sys.argv)>1 and int(sys.argv[1])==0:

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
    for idx_dow in [1, 2, 4]: #range(1):

        nonCompleteBoard = 0
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for idx_test in range(len(files_list)):

            # Original
            oCam_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

            original_board, gray = OpenCV_corners(files_list[idx_test])

            if idx_dow==0:
                dwn_board=np.copy(original_board)
            else:
                dwn_board, gray = OpenCV_corners(files_list[idx_test].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
                dwn_board /= [0.75, 0.5, 0.25, 1/3, 1/6][idx_dow]

            if dwn_board.shape[0]!=0:

                # Take predictions that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 5
                dwn_board = dwn_board[oCamIdx]

                if dwn_board.shape[0]==54:
                    imgpoints.append(dwn_board[:,np.newaxis,:])
                    objpoints.append(objp)
                else:
                    nonCompleteBoard += 1
            else:
                nonCompleteBoard+=1


        if not alternative:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error

            print("------------------------------------------------------------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard, len(files_list)-nonCompleteBoard)
            print("total error: ", tot_error/len(objpoints))

        else:
            print('----NonComplete ', nonCompleteBoard, len(files_list)-nonCompleteBoard)

            error_list = []
            for kk in range(40):

                idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                img_pts = [imgpoints[i] for i in idx_choice]
                obj_pts = [objpoints[i] for i in idx_choice]

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                tot_error = 0
                for i in range(20):
                    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error

                error_list.append(tot_error/20)

            print("-------------------------------------------OpenCV-------------")
            print(idx_dow)
            print(np.mean(error_list), np.std(error_list), len(files_list)-nonCompleteBoard)




#####################################
### 3. Rochade

if len(sys.argv)>1 and int(sys.argv[1])==1:

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
    for idx_dow in [1, 2, 4]: #range(1):

        nonCompleteBoard = 0
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for idx_test in range(len(files_list)):

            # Original
            oCam_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

            original_board = \
                np.loadtxt(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_rochade_refined_5.txt'))
            original_board-=1 #matlab

            if idx_dow==0:
                dwn_board=np.copy(original_board)
            else:
                dwn_board = np.loadtxt(
                    files_list[idx_test].replace('/benchmarkData/GoPro/', '/results/GoPro_{0}/'.format(idx_dow)).replace(
                        '.JPG', '_rochade_refined_5.txt'))
                dwn_board -= 1  # matlab
                dwn_board /= [0.75, 0.5, 0.25, 1/3, 1/6][idx_dow]

            if dwn_board.shape[0]!=0:

                # Take predictions that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 5
                dwn_board = dwn_board[oCamIdx]

                if dwn_board.shape[0]==54:

                    original_openCV, _ = OpenCV_corners(files_list[idx_test])
                    CVdistance = np.zeros([original_openCV.shape[0], dwn_board.shape[0]])
                    for kk in range(original_openCV.shape[0]):
                        for jj in range(dwn_board.shape[0]):
                            CVdistance[kk, jj] = np.linalg.norm(original_openCV[kk, :2] - dwn_board[jj, :2])

                    correspondance = np.argmin(CVdistance, axis=1)
                    dwn_board = dwn_board[correspondance,:]

                    imgpoints.append(dwn_board[:,np.newaxis,:].astype(np.float32))
                    objpoints.append(objp)
                else:
                    nonCompleteBoard += 1
            else:
                nonCompleteBoard+=1

        _, gray = OpenCV_corners(files_list[0].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        if not alternative:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error

            print("------------------------------------------------------------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print("total error: ", tot_error/len(objpoints))

        else:
            error_list = []
            print('----NonComplete ', nonCompleteBoard, len(files_list) - nonCompleteBoard)
            for kk in range(40):

                idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                img_pts = [imgpoints[i] for i in idx_choice]
                obj_pts = [objpoints[i] for i in idx_choice]

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                tot_error = 0
                for i in range(20):
                    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error

                error_list.append(tot_error/20)

            print("----------------------------------------Rochade----------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print(np.mean(error_list), np.std(error_list))

if len(sys.argv)>1 and int(sys.argv[1])==2:

    #####################################
    ### 4. OCamCalib

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
    for idx_dow in [1, 2, 4]: #range(1):

        nonCompleteBoard = 0
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for idx_test in range(len(files_list)):

            # Original
            oCam_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

            # Original
            original_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))


            if idx_dow==0:
                dwn_board=np.copy(original_board)
            else:
                dwn_board = np.load(
                    files_list[idx_test].replace('/benchmarkData/GoPro/', '/results/GoPro_{0}/'.format(idx_dow)).replace(
                        '.JPG', '_OCamCalib.npy'))
                dwn_board /= [0.75, 0.5, 0.25, 1/3, 1/6][idx_dow]

            if dwn_board.shape[0]!=0:

                # Take predictions that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 5
                dwn_board = dwn_board[oCamIdx]

                if dwn_board.shape[0]==54:

                    original_openCV, _ = OpenCV_corners(files_list[idx_test])
                    CVdistance = np.zeros([original_openCV.shape[0], dwn_board.shape[0]])
                    for kk in range(original_openCV.shape[0]):
                        for jj in range(dwn_board.shape[0]):
                            CVdistance[kk, jj] = np.linalg.norm(original_openCV[kk, :2] - dwn_board[jj, :2])

                    correspondance = np.argmin(CVdistance, axis=1)
                    dwn_board = dwn_board[correspondance,:]

                    imgpoints.append(dwn_board[:,np.newaxis,:].astype(np.float32))
                    objpoints.append(objp)
                else:
                    nonCompleteBoard += 1
            else:
                nonCompleteBoard+=1

        _, gray = OpenCV_corners(files_list[0].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        if not alternative:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error

            print("------------------------------------------------------------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print("total error: ", tot_error/len(objpoints))

        else:
            error_list = []
            print('----NonComplete ', nonCompleteBoard, len(files_list) - nonCompleteBoard)
            for kk in range(10):

                idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                img_pts = [imgpoints[i] for i in idx_choice]
                obj_pts = [objpoints[i] for i in idx_choice]

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                tot_error = 0
                for i in range(20):
                    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error

                error_list.append(tot_error/20)

            print("---------------------------------------------OCamCalib------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print(np.mean(error_list), np.std(error_list))


#####################################
### 0. Our Approach



if len(sys.argv)>1 and int(sys.argv[1])==3:

    version = "torch_full_v6_"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimPred = 2 + 1
    n_points = 1


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
            input = input.permute([0, 3, 1, 2])
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
                np.linspace(0, original_shape_y - original_shape_y / final_y, final_y),
                np.linspace(0, original_shape_x - original_shape_x / final_x, final_x))

            x_clone = x.clone()  # to allow for in-place operations
            x_clone[:, 0, :, :] *= original_shape_x / final_x  # local offset
            x_clone[:, 1, :, :] *= original_shape_y / final_y  # local offset
            x_clone[:, 0, :, :] += torch.tensor(y_offset[np.newaxis, :, :], device=device).float() - 0.5
            x_clone[:, 1, :, :] += torch.tensor(x_offset[np.newaxis, :, :], device=device).float() - 0.5

            x_clone = x_clone.permute(0, 2, 3, 1)

            return x_clone.view(x_clone.shape[0], -1, dimPred)


    # Define Graph

    model = Net().to(device)
    model.eval()
    save_dict = torch.load('models/' + version + 'model.ckpt',
                           map_location=lambda storage, location: storage)
    model.load_state_dict(save_dict)


    def getHeatmapCorners(file):
        img = cv2.imread(file, 0)
        img = cv2.GaussianBlur(img, (13, 13), 1)

        if len(img.shape) > 2:
            input = img[np.newaxis]
        else:
            input = np.tile(img[np.newaxis, :, :, np.newaxis], [1, 1, 1, 3])

        prediction = model(torch.tensor(input, device=device).float())

        pp = prediction.cpu().data.numpy()
        pp = pp[0, pp[0, :, 2] > 0.2, :]
        pp = pp[:, [1, 0, 2]]

        # quick cleaning: delete potential duplicates
        pp = pp[np.argsort(-pp[:, 2]), :]
        index_list = []
        for pp_index in range(pp.shape[0]):
            min_distance = 100
            for pp_compare in range(pp_index):
                min_distance = min(min_distance, np.linalg.norm(pp[pp_index, :2] - pp[pp_compare, :2]))
            # print(min_distance)
            if min_distance > 2:
                index_list.append(pp_index)
        pp = pp[np.array(index_list), :]

        return pp


    ##########################################


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
    for idx_dow in [1,2,4]: #range(1):

        nonCompleteBoard = 0
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for idx_test in range(len(files_list)):

            print(idx_test)

            # Original
            oCam_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

            original_board = getHeatmapCorners(files_list[idx_test])

            if idx_dow==0:
                dwn_board=np.copy(original_board)
            else:
                dwn_board = getHeatmapCorners(files_list[idx_test].replace('/GoPro/', '/GoPro_dwn_{0}/'.format(idx_dow)))
                dwn_board /= [0.75, 0.5, 0.25, 1/3, 1/6][idx_dow]

            if dwn_board.shape[0]!=0:
                # Take predictions that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 6
                dwn_board = dwn_board[oCamIdx]

                if dwn_board.shape[0]==54:
                    original_openCV, _ = OpenCV_corners(files_list[idx_test])
                    CVdistance = np.zeros([original_openCV.shape[0], dwn_board.shape[0]])
                    for kk in range(original_openCV.shape[0]):
                        for jj in range(dwn_board.shape[0]):
                            CVdistance[kk, jj] = np.linalg.norm(original_openCV[kk, :2] - dwn_board[jj, :2])

                    correspondance = np.argmin(CVdistance, axis=1)
                    dwn_board = dwn_board[correspondance,:]

                    imgpoints.append(dwn_board[:,np.newaxis,:2].astype(np.float32))
                    objpoints.append(objp)
                else:
                    nonCompleteBoard += 1
            else:
                nonCompleteBoard+=1

        _, gray = OpenCV_corners(files_list[0].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        if not alternative:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error

            print("------------------------------------------------------------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print("total error: ", tot_error/len(objpoints))

        else:
            error_list = []
            print('----NonComplete ', nonCompleteBoard, len(files_list) - nonCompleteBoard)
            for kk in range(40):

                idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                img_pts = [imgpoints[i] for i in idx_choice]
                obj_pts = [objpoints[i] for i in idx_choice]

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                tot_error = 0
                for i in range(20):
                    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error

                error_list.append(tot_error/20)

            print("------------------------------------------ Ours --------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print(np.mean(error_list), np.std(error_list), len(files_list) - nonCompleteBoard)



#####################################
### 5. Heatmap

if len(sys.argv)>1 and int(sys.argv[1])==4:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimPred = 2 + 1
    n_points = 1


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, 13, 1, padding=6)
            self.conv2 = nn.Conv2d(32, 64, 1, 1)
            self.conv3 = nn.Conv2d(64, 1, 3, 1, padding=1)
            torch.nn.init.constant_(self.conv3.bias, -4.0)

        def forward(self, x):
            x = x.permute([0, 3, 1, 2])
            x = (x - 255 / 2) / 255

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.sigmoid(self.conv3(x))

            return torch.squeeze(x)


    version = "torch_benchmark_v6_"
    model = Net().to(device)
    model.eval()
    save_dict = torch.load('models/' + version + 'model.ckpt',
                           map_location=lambda storage, location: storage)
    model.load_state_dict(save_dict)

    _subpixel = False


    def getHeatmapCorners(file):
        img = cv2.imread(file, 0)

        if len(img.shape) > 2:
            input = img[np.newaxis]
        else:
            input = np.tile(img[np.newaxis, :, :, np.newaxis], [1, 1, 1, 3])

        prediction = model(torch.tensor(input.astype(np.float32), device=device).float())
        pp = prediction.cpu().data.numpy()

        if _subpixel:
            pp = np.array(utils.subpixelPrediction(np.copy(pp)))
        else:
            pp = np.array(utils.argmaxPrediction(np.copy(pp))).astype(np.float32)

        plt.figure(figsize=(15, 15))
        for kk in range(1):
            plt.subplot(1, 1, 1)
            plt.imshow(input[0])
            location_list = pp
            for kk in range(len(location_list)):
                plt.scatter(location_list[kk][0], location_list[kk][1], c='r')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('tmpFinal.png')
        plt.close('all')

        return pp


    ##########################################


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
    for idx_dow in [1,2,4]: #range(1):

        nonCompleteBoard = 0
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for idx_test in range(len(files_list)):
            print(idx_test)

            # Original
            oCam_board = \
                np.load(files_list[idx_test].replace('/benchmarkData/GoPro/','/results/GoPro_original/').replace('.JPG','_OCamCalib.npy'))

            original_board = getHeatmapCorners(files_list[idx_test])

            if idx_dow==0:
                dwn_board=np.copy(original_board)
            else:
                dwn_board = getHeatmapCorners(files_list[idx_test].replace('/GoPro/', '/GoPro_dwn_{0}/'.format(idx_dow)))
                dwn_board /= [0.75, 0.5, 0.25, 1/3, 1/6][idx_dow]

            if dwn_board.shape[0]!=0:
                # Take predictions that are a board
                oCamdistance = np.zeros([oCam_board.shape[0], dwn_board.shape[0]])
                for kk in range(oCam_board.shape[0]):
                    for jj in range(dwn_board.shape[0]):
                        oCamdistance[kk, jj] = np.linalg.norm(oCam_board[kk, :2] - dwn_board[jj, :2])
                oCammini = np.min(oCamdistance,axis=0)
                oCamIdx = oCammini < 5
                dwn_board = dwn_board[oCamIdx]

                if dwn_board.shape[0]==54:
                    original_openCV, _ = OpenCV_corners(files_list[idx_test])
                    CVdistance = np.zeros([original_openCV.shape[0], dwn_board.shape[0]])
                    for kk in range(original_openCV.shape[0]):
                        for jj in range(dwn_board.shape[0]):
                            CVdistance[kk, jj] = np.linalg.norm(original_openCV[kk, :2] - dwn_board[jj, :2])

                    correspondance = np.argmin(CVdistance, axis=1)
                    dwn_board = dwn_board[correspondance,:]

                    imgpoints.append(dwn_board[:,np.newaxis,:2].astype(np.float32))
                    objpoints.append(objp)
                else:
                    nonCompleteBoard += 1
            else:
                nonCompleteBoard+=1

            # Intermadiate calibration
            if idx_test%30==0 and len(objpoints)>=20:
                print("--------Intermadiate Calibration")
                _, gray = OpenCV_corners(files_list[0].replace('/GoPro/', '/GoPro_dwn_{0}/'.format(idx_dow)))
                if not alternative:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                    tot_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        tot_error += error

                    print("------------------------------------------------------------------")
                    print(idx_dow)
                    print('----NonComplete ', nonCompleteBoard)
                    print("total error: ", tot_error / len(objpoints))

                else:
                    error_list = []
                    for kk in range(40):

                        idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                        img_pts = [imgpoints[i] for i in idx_choice]
                        obj_pts = [objpoints[i] for i in idx_choice]

                        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                        tot_error = 0
                        for i in range(20):
                            imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                            error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                            tot_error += error

                        error_list.append(tot_error / 20)

                    print("------------------------------------------ Heatmap --------")
                    print(idx_dow)
                    print('----NonComplete ', nonCompleteBoard)
                    print(np.mean(error_list), np.std(error_list))


        ### Final Calibration
        _, gray = OpenCV_corners(files_list[0].replace('/GoPro/','/GoPro_dwn_{0}/'.format(idx_dow)))
        if not alternative:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error

            print("------------------------------------------------------------------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print("total error: ", tot_error/len(objpoints))

        else:
            error_list = []
            print('----NonComplete ', nonCompleteBoard, len(files_list) - nonCompleteBoard)
            for kk in range(40):

                idx_choice = np.random.choice(len(objpoints), size=20, replace=False)

                img_pts = [imgpoints[i] for i in idx_choice]
                obj_pts = [objpoints[i] for i in idx_choice]

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

                tot_error = 0
                for i in range(20):
                    imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    tot_error += error

                error_list.append(tot_error/20)

            print("------------------------------------------ Heatmap --------")
            print(idx_dow)
            print('----NonComplete ', nonCompleteBoard)
            print(np.mean(error_list), np.std(error_list))













