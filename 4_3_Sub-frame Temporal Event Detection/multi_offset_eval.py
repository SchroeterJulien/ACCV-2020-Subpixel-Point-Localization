import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from multi_offset_model import EventDetector
from util import offset_correct_preds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
if not os.path.exists("results"):
    os.makedirs("results")


# Get the arguments from the command-line except the filename
import sys
argv = sys.argv[1:]
if len(argv)>0:
    step = int(argv[0])
    print('>>> step: ' + str(step))
else:
    step = 8

if len(argv)>1:
    run_idx = int(argv[1])
else:
    run_idx = 65

if len(argv)>2:
    split = int(argv[2])
else:
    split = 2


version_name = 'multi_v4' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)


def eval(model, split, seq_length, n_cpu, disp, steps=1):
    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    delta = []
    tolerance = []
    ground_truth = []
    predictions = []
    predictions_original = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'][:,::steps,:,:,:], sample['labels']
        assert(images.shape[0]==1)

        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        seq_length_new = int(np.ceil(seq_length/steps))
        batch = 0
        while batch * (seq_length_new-1) < images.shape[1]-1:
            if (batch + 1) * (seq_length_new-1) + 1 > images.shape[1]:
                image_batch = images[:, batch * (seq_length_new-1):, :, :, :]
            else:
                image_batch = images[:, batch * (seq_length_new-1):(batch + 1) * (seq_length_new-1)+1, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = logits.cpu().data.numpy()[0]
            else:
                local_preds = logits.cpu().data.numpy()[0]
                local_preds[:,0]+= batch * steps * (seq_length_new-1)
                probs = np.concatenate([probs, local_preds], 0)
            batch += 1
        gt, pp, deltas, tol, c, original_pp = offset_correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
            #print(i, np.round(deltas))

        correct.append(c)
        tolerance.append(tol)
        delta.append(deltas)
        ground_truth.append(gt)
        predictions.append(pp)
        predictions_original.append(original_pp)

    delta = np.array(delta)
    tolerance = np.array(tolerance)
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    predictions_original = np.array(predictions_original)


    np.savez_compressed('results/' + version_name + '.npz', array1=np.array(delta), array2=np.array(predictions),
                                                    array3=np.array(tolerance), array4=np.array(ground_truth),
                                                    array5=np.array(predictions_original))


    print(np.round(np.mean(np.array(correct),0),3))
    print(np.round(np.sqrt(np.mean(np.square(delta/tolerance[:,np.newaxis]),0)),1))
    print(np.round(np.std(np.array(delta)/np.array(tolerance)[:,np.newaxis],0),3))
    print(np.round(np.median(np.array(delta)/np.array(tolerance)[:,np.newaxis],0),3))
    print(np.round(np.mean(np.abs(np.array(delta)/np.array(tolerance)[:,np.newaxis]),0),3))
    PCE = np.mean(correct)


    return PCE


if __name__ == '__main__':

    seq_length = 65
    n_cpu = 0

    print(step)
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False,
                          step=step)

    save_dict = torch.load('models/multi_' + version_name + '_10000.pth.tar', map_location=lambda storage, location: storage)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True, steps=step)
    print('Average PCE: {}'.format(PCE))

    with open('results/results_ours_' + str(split) + '.txt', 'a') as f:
        f.write(version_name)
        f.write(",%s" % str(step))
        f.write(",%s" % str(PCE))
        f.write("\n")

