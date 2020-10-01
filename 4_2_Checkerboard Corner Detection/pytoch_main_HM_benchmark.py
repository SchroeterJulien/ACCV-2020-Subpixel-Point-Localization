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


if not os.path.exists("plt"):
    os.makedirs("plt")
if not os.path.exists("models"):
    os.makedirs("models")


version = "torch_benchmark_v6_"
config = {'learning_rate':  0.0005, #0.001,
          'batch_size': 4, 'background_size': 4, 'niter': 100000, 'n_channel':1,
          'max_occurence': 37}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Generate training dataset
#data, labels = cd.generateDataset(1000, True)
#data, labels = cd.generateDataset(500, True)
print('load dataset')
data, labels, _ = cd.generateDataset(20000, False)

background_images = []
files_list = glob.glob('Checkerboard/background/*')
for file in files_list:
    im = cv2.imread(file)
    background_images.append(im)

print('load dataset')
##### Load banchmarks

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

model = Net().to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


# Loss placeholders
loss_window = {'loss': np.zeros([100]),  'max': np.zeros([100])} # 'count': np.zeros([25]), 'soft': np.zeros([25])
list_loss = {'loss': [], 'max': []} # 'soft': [], 'count': [],

iter = 0

while iter <= config['niter']:

    if iter%100==0:
        print(iter)

    # Create batch
    batch_x = np.zeros([config['batch_size']+config['background_size'], data.shape[1], data.shape[2], 3], np.uint8)
    batch_y= np.zeros([config['batch_size']+config['background_size'], data.shape[1], data.shape[2]])

    idx_batch = np.random.randint(0, len(data), config['batch_size'])
    batch_x[:config['batch_size']] = data[idx_batch]
    batch_y[:config['batch_size']] = labels[idx_batch]

    # Add background images:
    for kk in range(config['background_size']):
        idx = np.random.randint(len(background_images))
        xmin = np.random.randint(background_images[idx].shape[0]-100)
        ymin = np.random.randint(background_images[idx].shape[1]-100)
        tmp = utils.random_transform(background_images[idx][xmin:xmin+data.shape[1], ymin:ymin+data.shape[2]])
        batch_x[config['batch_size']+kk,:tmp.shape[0], :tmp.shape[1]] = tmp

    optimizer.zero_grad()
    prediction = model(torch.tensor(batch_x, device=device).float())
    loss = torch.mean((prediction - torch.tensor(batch_y, device=device).float())**2)
    loss.backward()
    optimizer.step()

    # Save loss
    loss_window['loss'][1:] = loss_window['loss'][:-1]
    loss_window['loss'][0] = loss.cpu().data.numpy()
    list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

    loss_window['max'][1:] = loss_window['max'][:-1]
    loss_window['max'][0] = np.max(prediction.cpu().data.numpy())
    list_loss['max'].append(np.median(loss_window['max'][loss_window['max'] != 0]))


    # Figures and stuff
    if iter % 500 == 0 and iter > 0:
        print("-----", iter)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(np.log(list_loss['loss']), 'k')
        plt.ylim([np.min(np.log(list_loss['loss'])),np.log(list_loss['loss'])[min(200,iter)]])
        plt.subplot(1, 2, 2)
        plt.plot(list_loss['max'], 'k')
        plt.ylim([0,1.05])
        plt.savefig('plt/' + version + 'loss.png')
        plt.close('all')


        def crop_image(img, img2):
            x_min = max(np.min(np.where(np.sum(img, axis=(1, 2)) != 0)) - 1, 0)
            x_max = np.max(np.where(np.sum(img, axis=(1, 2)) != 0)) + 1
            y_min = max(np.min(np.where(np.sum(img, axis=(0, 2)) != 0)) - 1, 0)
            y_max = np.max(np.where(np.sum(img, axis=(0, 2)) != 0)) + 1

            return img[x_min:x_max, y_min:y_max], img2[x_min:x_max, y_min:y_max]

        plt.figure(figsize=(15,15))
        for kk in range(4):
            plt.subplot(2,2,kk+1)
            img, map = crop_image(batch_x[kk], prediction[kk].cpu().data.numpy())
            plt.imshow(img)
            plt.imshow(map, alpha=0.8, cmap='Reds')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'results.png')
        plt.close('all')

        plt.figure(figsize=(15,15))
        for kk in range(4):
            plt.subplot(2,2,kk+1)
            img, map = crop_image(batch_x[kk], batch_y[kk])
            plt.imshow(img)
            plt.imshow(map, alpha=0.8, cmap='Reds')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()

        plt.savefig('plt/' + version + 'rgt.png')
        plt.close('all')


    if iter % 5000 == 0 and iter > 0:

        # MESA
        print("--- MESA")
        pp= model(torch.tensor(mesa_images, device=device).float()).cpu().data.numpy()
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(mesa_images[kk])
            plt.imshow(pp[kk], alpha=0.8, cmap='Reds')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_mesa.png')
        plt.close('all')

        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(mesa_images[kk])
            location_list = utils.subpixelPrediction(pp[kk])
            for kk in range(len(location_list)):
                plt.scatter(location_list[kk][0], location_list[kk][1], c='r')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()

        plt.savefig('plt/' + version + 'benchmark_mesa_2.png')
        plt.close('all')


        # GoPro
        print("--- GoPro")
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            pp = model(torch.tensor(gopro_images[kk:kk+1], device=device).float()).cpu().data.numpy()
            plt.subplot(2, 2, kk + 1)
            plt.imshow(gopro_images[kk])
            plt.imshow(pp, alpha=0.8, cmap='Reds')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_gopro.png')
        plt.close('all')

        plt.figure(figsize=(15, 15))
        for kk in range(4):
            pp = model(torch.tensor(gopro_images[kk:kk+1], device=device).float()).cpu().data.numpy()
            plt.subplot(2, 2, kk + 1)
            plt.imshow(gopro_images[kk])
            location_list = utils.subpixelPrediction(pp)
            for kk in range(len(location_list)):
                plt.scatter(location_list[kk][0], location_list[kk][1], c='r')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_gopro_2.png')
        plt.close('all')

        # uEye
        print("--- uEye")
        pp = model(torch.tensor(ueye_images, device=device).float()).cpu().data.numpy()
        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(ueye_images[kk])
            plt.imshow(pp[kk], alpha=0.8, cmap='Reds')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_ueye.png')
        plt.close('all')

        plt.figure(figsize=(15, 15))
        for kk in range(4):
            plt.subplot(2, 2, kk + 1)
            plt.imshow(ueye_images[kk])
            location_list = utils.subpixelPrediction(pp[kk])
            for kk in range(len(location_list)):
                plt.scatter(location_list[kk][0], location_list[kk][1], c='r')
            plt.axis('equal')
            plt.gca().axis('off')
            plt.tight_layout()
        plt.savefig('plt/' + version + 'benchmark_ueye_2.png')
        plt.close('all')

        torch.save(model.state_dict(), "models/" + version + "model.ckpt")

    iter += 1