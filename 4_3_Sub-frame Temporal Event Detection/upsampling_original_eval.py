from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import sys

import os
if not os.path.exists("results"):
    os.makedirs("results")

# Get the arguments from the command-line except the filename
argv = sys.argv[1:]
if len(argv)>0:
    step = int(argv[0])
    print('>>> step: ' + str(step))
else:
    step = 1
if len(argv)>1:
    run_idx = int(argv[1])
else:
    run_idx = 10

if len(argv)>2:
    split = int(argv[2])
else:
    split = 1

_video_interpolation = False

if not _video_interpolation:
    version_name = 'original_step_' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)
else:
    version_name = 'interpolation_step_' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)
print(version_name)

def eval(model, split, seq_length, n_cpu, disp, steps=1):
    print("------in function")
    if not _video_interpolation:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    else:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/'.replace('videos_160','videos_downsampled_' + str(steps) + 'x'),
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)
        steps = 1 

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    idx_keep = np.arange(0, seq_length, steps)
    idx_erase = np.delete(np.arange(0, seq_length), idx_keep)
    correct = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']

        if steps>1:
            #### Downsample video (temporally)
            images[:, idx_erase, :, :, :] = images[:, np.repeat(idx_keep, steps - 1)[:len(idx_erase)], :, :, :]

        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        if i==176:
            print('hello')
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)

    return PCE


if __name__ == '__main__':

    seq_length = 65
    n_cpu = 8

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)


    #save_dict = torch.load('models/swingnet_1800.pth.tar')
    save_dict = torch.load('models_original/' + version_name + '_10000.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True, steps=step)
    print('Average PCE: {}'.format(PCE))

    with open('results/results.txt', 'a') as f:
        f.write(version_name)
        f.write(",%s" % str(PCE))
        f.write("\n")


