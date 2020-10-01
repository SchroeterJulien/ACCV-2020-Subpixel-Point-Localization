from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

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

if __name__ == '__main__':

    # training configuration

    iterations = 10000
    it_save = 2000  # save model every 100 iterations
    n_cpu = 0
    seq_length = 65
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()

    if not _video_interpolation:
        version_name = 'original_step_' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)
        dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True,
                     step=step)

    else:
        version_name = 'interpolation_step_' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)
        dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/'.replace('videos_160','videos_downsampled_' + str(step) + 'x'),
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)
        step = 1



    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0

    while i < iterations:
        for sample in data_loader:

            images, labels = sample['images'].cuda(), sample['labels'].cuda()

            if step>1:
            #### Downsample video (temporally)
                idx_keep = np.arange(0, seq_length, step)
                idx_erase = np.delete(np.arange(0, seq_length),idx_keep)
                images[:,idx_erase,:,:,:] = images[:,np.repeat(idx_keep, step - 1)[:len(idx_erase)],:,:,:]

            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models_original/{}_{}.pth.tar'.format(version_name, i))
            if i == iterations:
                break
