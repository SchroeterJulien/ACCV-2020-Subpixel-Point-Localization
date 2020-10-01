import sys
argv = sys.argv[1:]

import os
if len(argv)>3:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(argv[3]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from dataloader import GolfDB, Normalize, ToTensor
from dense_original_model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the arguments from the command-line except the filename
if len(argv)>0:
    step = int(argv[0])
    print('>>> step: ' + str(step))
else:
    step = 20

if len(argv)>1:
    run_idx = int(argv[1])
else:
    run_idx = 9

if len(argv)>2:
    split = int(argv[2])
else:
    split = 1

print(split)
version_name = 'dense_original' + '_' + str(step) + '_' + str(run_idx) + '_' + str(split)


if __name__ == '__main__':
    # training configuration
    iterations = 10000
    it_save = 10000  # save model every 100 iterations
    n_cpu = 8
    seq_length = 65
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False,
                          step=step,
                          length=seq_length)
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True,
                     step=step)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'][:,::step,:,:,:].to(device), sample['labels'].to(device)

            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            if i % 10 == 0:
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/{}_{}.pth.tar'.format(version_name, i))
            if i == iterations:
                break





