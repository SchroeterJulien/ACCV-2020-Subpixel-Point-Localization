import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_SmoothedLosses as losses



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Directories
if not os.path.exists('plt'):
    os.makedirs('plt')
if not os.path.exists('preds'):
    os.makedirs('preds')
if not os.path.exists('models'):
    os.makedirs('models')

# Functions as in Nehme et al.
def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    return (im - min_val)/(max_val - min_val)

def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

# Set seed for reproducibility
np.random.seed(123)

# Configuration
config = {'version': "Ours_v1",
          'learning_rate':  0.001, 'count_factor':  0.1,
          'batch_size': 16, 'niter': 50000, 'n_channel':1, 'max_occurence': 100}

n_points = 2

# Load training data and divide it to training and validation sets
path2data = "data"
patches = np.load(os.path.join(path2data, "our_data_26.npy"))
labels = np.load(os.path.join(path2data, "our_labels_26.npy"))

X_train, X_test = patches[:], patches[-100:] # for final run all the data is included in the training set!
y_train, y_test = labels[:], labels[-100:]

print('Number of Training Examples: %d' % X_train.shape[0])
print('Number of Validation Examples: %d' % X_test.shape[0])


# Setting type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

####### Similar normalization as in Nehme et al.
mean_train = np.zeros(X_train.shape[0], dtype=np.float32)
std_train = np.zeros(X_train.shape[0], dtype=np.float32)
for i in range(X_train.shape[0]):
    X_train[i, :, :] = project_01(X_train[i, :, :])
    mean_train[i] = X_train[i, :, :].mean()
    std_train[i] = X_train[i, :, :].std()

# resulting normalized training images
mean_val_train = mean_train.mean()
std_val_train = std_train.mean()
X_train_norm = np.zeros(X_train.shape, dtype=np.float32)
for i in range(X_train.shape[0]):
    X_train_norm[i, :, :] = normalize_im(X_train[i, :, :], mean_val_train, std_val_train)

# patch size
psize = X_train_norm.shape[1]

# Reshaping
X_data = X_train_norm.reshape(X_train.shape[0], psize, psize, 1)
Y_label = y_train

# Test normalization
mean_test = np.zeros(X_test.shape[0], dtype=np.float32)
std_test = np.zeros(X_test.shape[0], dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test[i, :, :] = project_01(X_test[i, :, :])
    mean_test[i] = X_test[i, :, :].mean()
    std_test[i] = X_test[i, :, :].std()

# resulting normalized test images
mean_val_test = mean_test.mean()
std_val_test = std_test.mean()
X_test_norm = np.zeros(X_test.shape, dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test_norm[i, :, :] = normalize_im(X_test[i, :, :], mean_val_test, std_val_test)

# Reshaping
X_test = X_test_norm.reshape(X_test.shape[0], psize, psize, 1)
Y_test = y_test

# save normalization factor
np.save("models/" + config['version'] + "_meanstd.npy", np.array([mean_val_test, std_val_test]))

# Load test image stack
from getTestData import getTest
test_stacks = getTest(config['version'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        vv=32
        self.conv1 = nn.Conv2d(1, vv, 7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(vv, vv, 5, 1, padding=2)
        self.conv5 = nn.Conv2d(vv, vv, 5, 1, padding=2)
        self.conv6 = nn.Conv2d(vv, n_points*(2 + config['n_channel']), 3, 1,padding=1)
        torch.nn.init.constant_(self.conv6.bias, -3)


    def forward(self, input):

        # Difference with Nehme et al.: no batch-norm for simplicity
        input = input.permute([0,3,1,2])

        conv1 = F.relu(self.conv1(input))
        conv4 = F.relu(self.conv4(conv1))
        conv5 = F.relu(self.conv5(conv4))

        x = torch.sigmoid(self.conv6(conv5))

        # Prediction Offsets
        final_y = x.shape[3]
        original_shape_y = input.shape[3]
        final_x = x.shape[2]
        original_shape_x = input.shape[2]

        x_offset, y_offset = np.meshgrid(
            np.linspace(0, original_shape_y -original_shape_y / final_y, final_y),
            np.linspace(0, original_shape_x - original_shape_x / final_x, final_x))

        x_clone = x.clone() #to allow for in-place operations
        x_clone = x_clone.reshape(x_clone.shape[0], (2 + config['n_channel']), -1, x_clone.shape[2], x_clone.shape[3])

        x_clone[:, 0, :, :, :] *= original_shape_x / final_x # local offset
        x_clone[:, 1, :, :, :] *= original_shape_y / final_y # local offset
        x_clone[:, 0,:,:, :] += torch.tensor(y_offset[np.newaxis,np.newaxis,:,:], device=device).float() - 0.5
        x_clone[:, 1,:,:,:] += torch.tensor(x_offset[np.newaxis,np.newaxis,:,:], device=device).float() - 0.5

        return x_clone.reshape(x_clone.shape[0], (2 + config['n_channel']), -1).permute(0,2,1)


# Define Graph
model = Net().to(device)
model.train()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


# Loss placeholders
loss_window = {'loss': np.zeros([100]), 'max': np.zeros([100]), 'centroid': np.zeros([100]), 'count': np.zeros([100])}
list_loss = {'loss': [], 'max': [], 'centroid': [], 'count': []}

iter = 0
while iter <= config['niter']:
    if iter%100==0:
        print(iter)

    # Create batch
    idx_batch = np.random.randint(0, len(X_data), config['batch_size'])

    batch_x = X_data[idx_batch, :,:,:].astype( np.float32)
    batch_y= Y_label[idx_batch,:,:][:,:,[1,0,2]]

    # Initialize gradient
    optimizer.zero_grad()

    # Predictions
    prediction = model(torch.tensor(batch_x, device=device).float())


    # Continuous Heatmap-matching loss
    smoothing_lambda = 0.2
    loss_cHM = losses.torch_loss_cHM.apply(prediction, torch.tensor(batch_y, device=device).float(), smoothing_lambda)

    # Loss Counting
    loss_count = config['count_factor']* losses.CountingLoss(prediction, torch.tensor(batch_y, device=device).float(), threshold=0)
    # l1: torch.mean(torch.abs(prediction))
    # None: 0 * loss_cHM

    # Full Loss
    loss = loss_cHM + loss_count

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Save loss
    loss_window['loss'][1:] = loss_window['loss'][:-1]
    loss_window['loss'][0] = loss.cpu().data.numpy()
    list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

    loss_window['centroid'][1:] = loss_window['centroid'][:-1]
    loss_window['centroid'][0] = loss_cHM.cpu().data.numpy()
    list_loss['centroid'].append(np.median(loss_window['centroid'][loss_window['centroid'] != 0]))

    loss_window['count'][1:] = loss_window['count'][:-1]
    loss_window['count'][0] = loss_count.cpu().data.numpy()
    list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))

    loss_window['max'][1:] = loss_window['max'][:-1]
    loss_window['max'][0] = np.max(prediction[:,:,2:].cpu().data.numpy())
    list_loss['max'].append(np.median(loss_window['max'][loss_window['max'] != 0]))


    # Plot loss history
    if iter % 1000 == 0 and iter > 0:
        print("-----", iter, "|| lambda:", np.round(smoothing_lambda,2))
        plt.figure(figsize=(15,9))

        plt.subplot(1, 4, 1)
        plt.plot(np.log(list_loss['loss']), 'k')
        plt.xlabel('Loss', fontweight='bold',fontsize=14)
        plt.ylim([np.min(np.log(list_loss['loss']))-0.05,
                  np.log(list_loss['loss'])[max(min(iter-100,500),0)]])

        plt.subplot(1, 4, 2)
        plt.plot(np.log(list_loss['centroid']), 'k')
        plt.xlabel('Centroid', fontweight='bold',fontsize=14)
        plt.ylim([np.min(np.log(list_loss['centroid']))-0.05,
                  np.log(list_loss['centroid'])[max(min(iter-100,500),0)]])

        plt.subplot(1, 4, 3)
        plt.plot(np.log(list_loss['count']), 'k')
        plt.xlabel('Count', fontweight='bold',fontsize=14)
        plt.ylim([np.min(np.log(list_loss['count']))-0.05,
                  np.log(list_loss['count'])[max(min(iter-100,500),0)]])

        plt.subplot(1, 4, 4)
        plt.plot(list_loss['max'], 'k')
        plt.xlabel('Max', fontweight='bold',fontsize=14)
        plt.ylim([0.0,1.05])

        plt.savefig('plt/' + config['version'] + 'loss.png')
        plt.close('all')


    if iter % 10000 == 0 and iter > 0:

        #### Render Test Stack
        Images_norm = test_stacks[0]

        pred = []
        for img_index in range(Images_norm.shape[0]):
            prediction = model(torch.tensor(Images_norm[img_index:img_index + 1], device=device).float())
            pred.append(prediction.cpu().data.numpy())

        pp = np.concatenate(pred, axis=0)

        # Figure - Full stacked
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 1, 1)
        plt.imshow(np.sum(Images_norm[:, :, :, 0], axis=0))

        pp_final = np.reshape(pp, [-1, pp.shape[2]])
        np.save("preds/" +config['version'] + "predictions.npy", pp_final)


        pp_final_relevant = pp_final[pp_final[:, 2] > 0.1, :]
        plt.scatter(pp_final_relevant[:, 1] - 0.5, pp_final_relevant[:, 0] - 0.5, alpha=0.8, c='r', s=1)

        plt.axis('equal')
        plt.gca().axis('off')
        plt.tight_layout()
        plt.savefig('plt/' + config['version'] + '_final_image.png')
        plt.close('all')

        # Save point predictions for comparison: use CompareLocalization.jar to compute metrics against the ground-truth
        pp_list = []
        for kk in range(pp.shape[0]):
            pp_tmp = pp[kk]
            pp_tmp = pp_tmp[pp_tmp[:, 2] > 0.1, :]

            rows, cols = pp_tmp[:, 1], pp_tmp[:, 0]
            preds = np.zeros([len(rows), 3])
            preds[:, 0] = kk + 1
            preds[:, 1] = 8 * 12.5 * rows
            preds[:, 2] = 8 * 12.5 * cols

            pp_list.append(preds)

        np.savetxt("preds/ours_" + config['version'] + ".csv", np.concatenate(pp_list, axis=0), delimiter=",")


        # Save Model
        torch.save(model.state_dict(), "models/" + config['version'] + "_" + str(iter) +  "model.ckpt")

    iter += 1
