import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import create_dataset as cd
import utils as utils

version = "torch_full_v6_"
config = {'learning_rate':  0.0005, #0.001,
          'batch_size': 4, 'background_size': 4, 'niter': 100000, 'n_channel':1,
          'max_occurence': 37}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dimPred = 2 + 1
n_points = 1

##### Generate training dataset
print('>>>> Loading dataset')
X_data, y_location, y_dimension, file_list = cd.testDataset()



Y_label = np.zeros([len(y_location), config['max_occurence'], dimPred])
for kk in range(len(y_location)):
    Y_label[kk, :y_location[kk].shape[0], :2] = y_location[kk][:, 0, :]
    Y_label[kk, :y_location[kk].shape[0], 2] = 1




#####################################
### 1. Our Approach



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 13, 1, padding=6)
        self.dowsampling1 = nn.Conv2d(32, 32, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 1, 1)

        self.dowsampling2 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(64, n_points*dimPred, 3, 1,padding=1)

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


true_positive,false_negative,false_positive = 0,0,0

accuracy = []
import time
t = time.time()
for idx_test in range(X_data.shape[0]):
    print(idx_test, X_data.shape[0])

    prediction = model(torch.tensor(X_data[idx_test:idx_test+1], device=device).float())


    pp = prediction.cpu().data.numpy()
    pp = pp[0,pp[0,:,2]>0.2,:]
    pp = pp[:,[1, 0, 2]]


    yy = Y_label[idx_test][Y_label[idx_test][:, 2] > 0.2, :]


    if pp.shape[0]!=0:
        # quick cleaning: delete potential duplicates
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

        if False:
            plt.figure(figsize=(15, 15))
            for jj in range(1):
                plt.subplot(1,1,jj+1)
                plt.imshow(X_data[idx_test+jj])

                # Plot predictions
                for kk in range(config['n_channel']):
                    for ii in range(pp.shape[0]):
                        plt.scatter(pp[ii, 0], pp[ii, 1], alpha=pp[ii, 2+kk],c='r')  # edgecolors="r", facecolors='r')

                plt.axis('equal')
                plt.gca().axis('off')
                plt.tight_layout()
            plt.savefig('results/' + str(idx_test) + '.png')
            plt.close('all')


        distance = np.zeros([pp.shape[0], yy.shape[0]])
        for kk in range(pp.shape[0]):
            for jj in range(yy.shape[0]):
                distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])


        mini = np.min(distance,axis=0)
        mini = mini[mini<3] #non-detection

        accuracy+=mini.tolist()
        true_positive += np.sum(np.min(distance,axis=0)<=3)
        false_negative += np.sum(np.min(distance,axis=0)>3)

        assert((pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))>=0)
        false_positive += (pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))

    else:
        false_negative += yy.shape[0]

    if idx_test%20==0 and False:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        print(idx_test, np.mean(accuracy),recall, precision,2*precision*recall/(precision+recall))
elapsed = time.time() - t


print("--------------------------------")
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(idx_test+1, np.round(np.mean(accuracy),3),np.round(recall*100,1), np.round(precision*100,1),
                np.round(2*precision*recall/(precision+recall)*100,1))



print(1000/elapsed, " images per seconds")






#####################################
### 2. OpenCV

import cv2
true_positive,false_negative,false_positive = 0,0,0
accuracy = []
import time
t = time.time()
for idx_test in range(X_data.shape[0]):
    print(idx_test, X_data.shape[0])

    gray = cv2.cvtColor(np.copy(X_data[idx_test]), cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners

    if False: # This unfortunately does not work
        ret, corners = cv2.findChessboardCorners(gray, tuple(y_dimension[idx_test].astype(np.int8)), None)

        if ret == True:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine the corner predictions

            pp = corners[:, 0, :]

        else:
            pp = np.zeros([0,2])

    else:
        #other option
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)  # refine the corner predictions

        pp = corners[:,0,:]
    if False:
        plt.figure(figsize=(15, 15))
        for jj in range(1):
            plt.subplot(1,1,jj+1)
            plt.imshow(X_data[idx_test+jj])

            # Plot predictions
            for kk in range(config['n_channel']):
                for ii in range(pp.shape[0]):
                    plt.scatter(pp[ii, 0], pp[ii, 1],c='r')

            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('tmp.png')


    yy = Y_label[idx_test][Y_label[idx_test][:,2]>0.2,:]

    distance = np.zeros([pp.shape[0], yy.shape[0]])
    for kk in range(pp.shape[0]):
        for jj in range(yy.shape[0]):
            distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])

    if distance.shape[0]!=0:
        mini = np.min(distance,axis=0)
        mini = mini[mini<3] #non-detection
        accuracy+=mini.tolist()

        true_positive += np.sum(np.min(distance,axis=0)<=3)
        false_negative += np.sum(np.min(distance,axis=0)>3)

        assert((pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))>=0)
        false_positive += (pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))
    else:
        true_positive+=0
        false_negative+= distance.shape[1]
        false_positive+=0

    if idx_test%20==0 and False:
        precision = (true_positive+1e-12) / (true_positive + false_positive+1e-12)
        recall = true_positive / (true_positive + false_negative)

        print(idx_test, np.mean(accuracy),recall, precision,2*precision*recall/(precision+recall))

elapsed = time.time() - t


print("--------------------------------")
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(idx_test+1, np.round(np.mean(accuracy),3),np.round(recall*100,1), np.round(precision*100,1),
                    np.round(2*precision*recall/(precision+recall)*100,1))

print(1000/elapsed, " images per seconds")




#####################################
### 3. Rochade

import cv2
true_positive,false_negative,false_positive = 0,0,0
accuracy = []
count_empty =0
ROCHADE_idx = []
for idx_test in range(X_data.shape[0]):
    print(idx_test, X_data.shape[0])



    pp = np.loadtxt(file_list[idx_test].replace('.png','_rochade_refined_5.txt').replace('testData','results'))
    pp-= 1 #matlab stuff

    yy = Y_label[idx_test][Y_label[idx_test][:,2]>0.2,:]


    distance = np.zeros([pp.shape[0], yy.shape[0]])
    for kk in range(pp.shape[0]):
        for jj in range(yy.shape[0]):
            distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])

    if distance.shape[0]!=0:
        count_empty += 1
        mini = np.min(distance,axis=0)
        mini = mini[mini<3] #non-detection
        accuracy+=mini.tolist()

        true_positive += np.sum(np.min(distance,axis=0)<=3)
        false_negative += np.sum(np.min(distance,axis=0)>3)

        assert((pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))>=0)
        false_positive += (pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))
        ROCHADE_idx.append(idx_test)
    else:
        true_positive+=0
        false_negative+= distance.shape[1]
        false_positive+=0

    if idx_test%20==0:
        precision = (true_positive+1e-12) / (true_positive + false_positive+1e-12)
        recall = true_positive / (true_positive + false_negative)

        print(idx_test, np.mean(accuracy),recall, precision,2*precision*recall/(precision+recall))


print("--------------------------------")
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(idx_test+1, np.round(np.mean(accuracy),3),np.round(recall*100,1), np.round(precision*100,1),
                    np.round(2*precision*recall/(precision+recall)*100,1))


time = np.loadtxt('Checkerboard/results/timings.rochade_raw.txt')
print(1000/np.sum(time[:,1]), " images per seconds")



#####################################
### 4. Heatmap baselines

_subpixel = True # gaussian fitting or argmax


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 13, 1, padding=6)
        self.conv2 = nn.Conv2d(32, 32, 1, 1)
        self.conv3 = nn.Conv2d(32, 1, 3, 1,padding=1)
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


true_positive,false_negative,false_positive = 0,0,0

accuracy = []

import time
t = time.time()
for idx_test in range(X_data.shape[0]):
    print(idx_test, X_data.shape[0])


    prediction = model(torch.tensor(X_data[idx_test:idx_test+1], device=device).float())
    pp = prediction.cpu().data.numpy()


    if False:
        plt.figure(figsize=(15, 15))
        for jj in range(1):
            plt.subplot(1,1,jj+1)
            plt.imshow(X_data[idx_test+jj])
            plt.imshow(pp, alpha=0.8, cmap='Reds')


            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('tmp.png')
        plt.close('all')


        plt.figure(figsize=(15, 15))
        for jj in range(1):
            plt.subplot(1,1,jj+1)
            plt.imshow(X_data[idx_test+jj])

            location_list = utils.argmaxPrediction(np.copy(pp))

            for kk in range(len(location_list)):
                plt.scatter(location_list[kk][0], location_list[kk][1], c='r')

            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('tmp.png')
        plt.close('all')

    if _subpixel:
        pp = np.array(utils.subpixelPrediction(np.copy(pp)))
    else:
        pp = np.array(utils.argmaxPrediction(np.copy(pp)))


    yy = Y_label[idx_test][Y_label[idx_test][:,2]>0.2,:]

    distance = np.zeros([pp.shape[0], yy.shape[0]])
    for kk in range(pp.shape[0]):
        for jj in range(yy.shape[0]):
            distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])

    if distance.shape[0]!=0:

        mini = np.min(distance,axis=0)
        mini = mini[mini<3] #non-detection

        accuracy+=mini.tolist()
        true_positive += np.sum(np.min(distance,axis=0)<=3)
        false_negative += np.sum(np.min(distance,axis=0)>3)

        assert((pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))>=0)
        false_positive += (pp.shape[0]-np.sum(np.min(distance,axis=0)<=3))

    else:
        true_positive+=0
        false_negative+= distance.shape[1]
        false_positive+=0

    if idx_test%20==0 and False:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        print(idx_test, np.mean(accuracy),recall, precision,2*precision*recall/(precision+recall))

elapsed = time.time() - t

print("--------------------------------")
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(idx_test+1, np.round(np.mean(accuracy),3),np.round(recall*100,1), np.round(precision*100,1),
                np.round(2*precision*recall/(precision+recall)*100,1))


print(1000/elapsed, " images per seconds")

