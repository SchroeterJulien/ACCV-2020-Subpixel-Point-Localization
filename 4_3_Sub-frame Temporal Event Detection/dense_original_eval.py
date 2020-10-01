import sys
argv = sys.argv[1:]

import os
if len(argv)>3:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(argv[3]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dense_original_model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds

import os
if not os.path.exists("results"):
    os.makedirs("results")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the arguments from the command-line except the filename
if len(argv)>0:
    step = int(argv[0])
    print('>>> step: ' + str(step))
else:
    step = 16

if len(argv)>1:
    run_idx = int(argv[1])
else:
    run_idx = 65

if len(argv)>2:
    split = int(argv[2])
else:
    split = 1

version_name = 'dense' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)

def eval(model, split, seq_length, n_cpu, disp):
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
    predictions = []
    predictions_original = []
    ground_truth = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'][:,::step,:,:,:], sample['labels']

        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        seq_length_new = int(np.ceil(seq_length/step))
        batch = 0
        while batch * (seq_length_new-1) < images.shape[1]-1:
            if (batch + 1) * (seq_length_new-1) + 1 > images.shape[1]:
                image_batch = images[:, batch * (seq_length_new-1):, :, :, :]
            else:
                image_batch = images[:, batch * (seq_length_new-1):(batch + 1) * (seq_length_new-1)+1, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs[:-1], F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        gt, pp, deltas, tol, c, original = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)

        correct.append(c)
        tolerance.append(tol)
        delta.append(deltas)
        predictions.append(pp)
        ground_truth.append(gt)
        predictions_original.append(original)

    np.savez_compressed('results/' + version_name + '.npz', array1=np.array(delta), array2=np.array(predictions),
                                                    array3=np.array(tolerance), array4=np.array(ground_truth),
                                                    array5=np.array(predictions_original))

    print(np.round(np.mean(np.array(correct),0),3))
    print(np.round(np.sqrt(np.mean(np.square(np.array(delta)/np.array(tolerance)[:,np.newaxis]),0)),3))
    print(np.round(np.std(np.array(delta)/np.array(tolerance)[:,np.newaxis],0),3))
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
                          dropout=False,
                          step=step,
                          length=seq_length)

    save_dict = torch.load('finalModels/' + version_name + '_10000.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    with open('results/original_dense.txt', 'a') as f:
        f.write(version_name)
        f.write(",%s" % str(PCE))
        f.write("\n")


