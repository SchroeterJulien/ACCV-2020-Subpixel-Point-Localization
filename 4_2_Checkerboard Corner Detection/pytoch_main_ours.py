import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import create_dataset as cd
import utils as utils
import SmoothedLosses as losses

if not os.path.exists("plt"):
    os.makedirs("plt")
if not os.path.exists("models"):
    os.makedirs("models")


version = "torch_full_v6_"
config = {'learning_rate':  0.0005, #0.001,
          'batch_size': 4, 'background_size': 4, 'niter': 100000, 'n_channel':1,
          'max_occurence': 37}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dimPred = 2 + 1
n_points = 1

##### Generate training dataset
print('>>>> Loading dataset')
X_data, _, y_location = cd.generateDataset(20000, True)

Y_label = np.zeros([len(y_location), config['max_occurence'], dimPred])
for kk in range(len(y_location)):
    Y_label[kk, :y_location[kk].shape[0], :2] = y_location[kk][:, 0, :]
    Y_label[kk, :y_location[kk].shape[0], 2] = 1

print('>>>> Loading backgrounds')
# Load background image
background_images = []
files_list = glob.glob('Checkerboard/background/*')
for file in files_list:
    im = cv2.imread(file)
    background_images.append(im)

##### Load test dataset
print('>>>> Loading test data')

## MESA
mesa_images = []
files_list = glob.glob('Checkerboard/benchmarkData/Mesa/*.png')
for kk in range(4):
    im = cv2.imread(files_list[kk])
    mesa_images.append(im)

mesa_images = np.stack(mesa_images,axis=0)

## GoPro
gopro_images = []
files_list = glob.glob('Checkerboard/benchmarkData/GoPro/*.JPG')
for kk in range(4):
    im = cv2.imread(files_list[kk])
    gopro_images.append(im)

gopro_images = np.stack(gopro_images,axis=0)

## uEye
ueye_images = []
files_list = glob.glob('Checkerboard/benchmarkData/uEye/*.png')
for kk in range(4):
    im = cv2.imread(files_list[kk])
    ueye_images.append(im)

ueye_images = np.stack(ueye_images,axis=0)

### Define network
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
model.train()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate']) #Adam


# Loss placeholders
loss_window = {'loss': np.zeros([100]), 'max': np.zeros([100]), 'centroid': np.zeros([100]), 'count': np.zeros([100])}
list_loss = {'loss': [], 'max': [], 'centroid': [], 'count': []}

iter = 0
while iter <= config['niter']:
    if iter%10==0:
        print(iter, np.round(max(min(0.1 * 5000 / max(iter-25000,1),0.1),0.01),3))

    # Create batch
    batch_x = np.zeros([config['batch_size']+config['background_size'], X_data.shape[1], X_data.shape[2], 3], np.uint8)
    batch_y= np.zeros([config['batch_size']+config['background_size'], config['max_occurence'], 2+config['n_channel']])

    idx_batch = np.random.randint(0, len(X_data), config['batch_size'])
    batch_x[:config['batch_size']] = np.transpose(X_data[idx_batch, :],(0,2,1,3))
    batch_y[:config['batch_size']] = Y_label[idx_batch]

    # Add background images:
    for kk in range(config['background_size']):
        idx = np.random.randint(len(background_images))
        xmin = np.random.randint(background_images[idx].shape[0]-100)
        ymin = np.random.randint(background_images[idx].shape[1]-100)
        tmp = utils.random_transform(background_images[idx][xmin:xmin+X_data.shape[1], ymin:ymin+X_data.shape[2]])
        batch_x[config['batch_size']+kk,:tmp.shape[0], :tmp.shape[1]] = tmp


    optimizer.zero_grad()
    prediction = model(torch.tensor(batch_x, device=device).float())
    loss_centroid = 0.7 * losses.torch_loss_centroid.apply(prediction, torch.tensor(batch_y, device=device).float()) + 1

    loss_count =  0.1 * losses.CountingLoss(prediction, torch.tensor(batch_y, device=device).float(),
                                            threshold=max(min(0.1 * 5000 / max(iter-25000,1),0.1),0.01))
    if iter>25000:
        loss = loss_centroid +  min((iter-25000)/25000,1) * loss_count
    else:
        loss = loss_centroid
    loss.backward()
    optimizer.step()

    # Save loss
    loss_window['loss'][1:] = loss_window['loss'][:-1]
    loss_window['loss'][0] = loss.cpu().data.numpy()
    list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

    loss_window['centroid'][1:] = loss_window['centroid'][:-1]
    loss_window['centroid'][0] = loss_centroid.cpu().data.numpy()
    list_loss['centroid'].append(np.median(loss_window['centroid'][loss_window['centroid'] != 0]))

    loss_window['count'][1:] = loss_window['count'][:-1]
    loss_window['count'][0] = loss_count.cpu().data.numpy()
    list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))

    loss_window['max'][1:] = loss_window['max'][:-1]
    loss_window['max'][0] = np.max(prediction[:,:,2:].cpu().data.numpy())
    list_loss['max'].append(np.median(loss_window['max'][loss_window['max'] != 0]))


    # Figures and stuff
    if iter % 500 == 0 and iter > 0:
        print("-----", iter)
        plt.figure(figsize=(15,9))
        plt.subplot(1, 4, 1)
        plt.plot(np.log(list_loss['loss']), 'k')
        plt.subplot(1, 4, 2)
        plt.plot(np.log(list_loss['centroid']), 'k')
        plt.subplot(1, 4, 3)
        plt.plot(np.log(list_loss['count']), 'k')
        plt.subplot(1, 4, 4)
        plt.plot(list_loss['max'], 'k')
        plt.ylim([0,1.05])
        plt.savefig('plt/' + version + 'loss.png')
        plt.close('all')


        def crop_image(img, img2):
            x_min = max(np.min(np.where(np.sum(img, axis=(1, 2)) != 0)) - 1, 0)
            x_max = np.max(np.where(np.sum(img, axis=(1, 2)) != 0)) + 1
            y_min = max(np.min(np.where(np.sum(img, axis=(0, 2)) != 0)) - 1, 0)
            y_max = np.max(np.where(np.sum(img, axis=(0, 2)) != 0)) + 1

            return img[x_min:x_max, y_min:y_max], img2[x_min:x_max, y_min:y_max], [x_min, x_max, y_min, y_max]


        pp = prediction.cpu().data.numpy()
        plt.figure(figsize=(15,15))
        for jj in range(4):
            plt.subplot(2,2,jj+1)
            img, map, lim = crop_image(batch_x[jj], batch_x[jj])
            plt.imshow(img)

            # Plot label
            for kk in range(config['n_channel']):
                for ii in range(batch_y.shape[1]):
                    plt.scatter(batch_y[jj, ii, 1]-lim[2], batch_y[jj, ii, 0]-lim[0], alpha=0.5 * batch_y[jj, ii, 2 + kk],
                                c='b')

            # Plot predictions
            for kk in range(config['n_channel']):
                for ii in range(pp.shape[1]):
                    if pp[jj, ii, 2 + kk] > 0.1:
                        plt.scatter(pp[jj, ii, 1]-lim[2], pp[jj, ii, 0]-lim[0], alpha=pp[jj, ii, 2 + kk], c='r')

            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'results.png')
        plt.close('all')


        pp = prediction.cpu().data.numpy()
        plt.figure(figsize=(15,15))
        for jj in range(4):
            jj=jj+1
            plt.subplot(2,2,jj)
            plt.imshow(batch_x[-jj])

            # Plot predictions
            for kk in range(config['n_channel']):
                count = 0
                for ii in range(pp.shape[1]):
                    if pp[-jj, ii, 2 + kk] > 0.1:
                        plt.scatter(pp[-jj, ii, 1]-lim[2], pp[-jj, ii, 0]-lim[0], alpha=pp[-jj, ii, 2 + kk], c='r')
                        count+=1
                    if count>50:
                        break
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'background.png')
        plt.close('all')


    if iter % 5000 == 0 and iter > 0:

        # MESA
        print("--- MESA")
        pp= model(torch.tensor(mesa_images, device=device).float()).cpu().data.numpy()
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(mesa_images[kk])

            for jj in range(config['n_channel']):
                for ii in range(pp.shape[1]):
                    if pp[kk, ii, 2 + jj] > 0.2:
                        plt.scatter(pp[kk, ii, 1]-lim[2], pp[kk, ii, 0]-lim[0], alpha=pp[kk, ii, 2 + jj], c='r')


            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_mesa.png')
        plt.close('all')


        # GoPro
        print("--- GoPro")
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            pp = model(torch.tensor(gopro_images[kk:kk+1], device=device).float()).cpu().data.numpy()

            plt.subplot(2, 2, kk + 1)
            plt.imshow(gopro_images[kk])

            # highest probabilities first
            idx = np.argsort(-pp[0,:,2])
            pp = pp[:,idx,:]
            count=0
            for jj in range(config['n_channel']):
                for ii in range(pp.shape[1]):
                    if pp[0, ii, 2 + jj] > 0.2:
                        plt.scatter(pp[0, ii, 1]-lim[2], pp[0, ii, 0]-lim[0], alpha=pp[0, ii, 2 + jj], c='r')
                        count+=1
                    if count>100:
                        break
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_gopro.png')
        plt.close('all')


        # uEye
        print("--- uEye")
        pp = model(torch.tensor(ueye_images, device=device).float()).cpu().data.numpy()
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(ueye_images[kk])

            # highest probabilities first
            idx = np.argsort(-pp[0,:,2])
            pp = pp[:,idx,:]

            count=1
            for jj in range(config['n_channel']):
                for ii in range(pp.shape[1]):
                    if pp[kk, ii, 2 + jj] > 0.2:
                        plt.scatter(pp[kk, ii, 1]-lim[2], pp[kk, ii, 0]-lim[0], alpha=pp[kk, ii, 2 + jj], c='r')
                        count+=1
                    if count>100:
                        break
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_ueye.png')
        plt.close('all')


        torch.save(model.state_dict(), "models/" + version + "model.ckpt")

    iter += 1
